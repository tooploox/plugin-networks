#!/usr/bin/env python3
import argparse
import glob
import json
import os
import warnings
import shutil
from datetime import datetime

import numpy as np
import torch
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events, create_supervised_evaluator
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pluginnet.common.metrics import log_results
from pluginnet.common.model import create_supervised_trainer, CheckpointManager


warnings.filterwarnings('ignore', module='PIL.TiffImagePlugin')
warnings.filterwarnings('ignore', message='indexing with dtype torch.uint8 is now deprecated, please use a '
                                          'dtype torch.bool instead.')


def task_factory(task):
    if task == 'sun397':
        from pluginnet.sun397.sun397 import create_model as sun397_create_model
        return sun397_create_model
    elif task == 'sun397_pe':
        from pluginnet.sun397.partial_evidence import create_model as sun397_pe_create_model
        return sun397_pe_create_model
    else:
        raise ValueError('No such task %s' % task)


def create_summary_writer(model, data_loader, log_dir, run_id):
    log_dir = os.path.join(log_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def load_model(all_params, state_dicts):
    for k in all_params.keys():
        all_params[k].load_state_dict(state_dicts[k])


def load_conf(conf_file):
    with open(conf_file, 'r') as fd:
        conf = json.load(fd)
        if conf['clip_gradient'] == -1:
            conf['clip_gradient'] = np.inf
        return conf


def find_recent_output_dir(tag, output):
    path = sorted(glob.glob(os.path.join(output, '%s*' % tag)))[-1]
    return os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch training script for SUN397 dataset')
    parser.add_argument('conf_file')
    parser.add_argument('output_dir', help='Model save directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('-T', '--tensor-board-dir', help='Tensor board log dir', default='runs')
    parser.add_argument('--restart', help='Restart', default=False, action='store_true')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--eval', default=False, action='store_true', help='checkpoint file')

    args = parser.parse_args()

    conf = load_conf(args.conf_file)
    train_set, val_set, net, criterion, metrics_dict, (score_name, score_function) = task_factory(conf['task'])(conf)

    if args.restart:
        run_id = find_recent_output_dir(conf['tag'], args.output_dir)
    else:
        run_id = '%s_%s' % (conf['tag'], datetime.now().strftime('%Y%m%d%H%M'))
    output_dir = os.path.join(args.output_dir, run_id)

    checkpoint_handler = CheckpointManager(output_dir, 'model', score_name=score_name,
                                           score_function=score_function, extra={'conf': conf, 'args': vars(args)})
    shutil.copy(args.conf_file, os.path.join(output_dir, 'conf.json'))
    loader_pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=loader_pin_memory,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=loader_pin_memory,
                                             drop_last=False)

    writer = create_summary_writer(net, train_loader, args.tensor_board_dir, run_id)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device, gradient_clip=conf['clip_gradient'])
    train_evaluator = create_supervised_evaluator(net,
                                                  metrics=metrics_dict,
                                                  device=device)

    evaluator = create_supervised_evaluator(net,
                                            metrics=metrics_dict,
                                            device=device)

    step_scheduler = StepLR(optimizer, step_size=conf['lr_step'], gamma=conf['lr_decay'])
    scheduler = LRScheduler(step_scheduler)
    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    all_params = {'model': net, 'optimizer': optimizer,
                  'lr_scheduler': step_scheduler}
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, all_params)
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )
    log_interval = 10

    # load checkpoint
    if args.restart and checkpoint_handler.is_checkpoint_available():
        state_dicts = checkpoint_handler.load_last()
        load_model(all_params, state_dicts)
    elif args.checkpoint is not None:
        state_dicts = checkpoint_handler.load(args.checkpoint)
        load_model(all_params, state_dicts)

    @trainer.on(Events.EPOCH_STARTED)
    def setup_engine(engine):
        if engine.state.epoch == 1:
            engine.state.epoch = checkpoint_handler.epoch_ + 1

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        train_evaluator.run(train_loader)
        log_results(engine, train_evaluator, "Training", writer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        checkpoint_handler.epoch_ = engine.state.epoch
        evaluator.run(val_loader)
        log_results(engine, evaluator, "Validation", writer)
        pbar.n = pbar.last_print_n = 0

    if args.eval:
        evaluator.run(val_loader)
        log_results(evaluator, evaluator, "Validation", writer)
    else:
        trainer.run(train_loader, max_epochs=conf['epochs'])
    pbar.close()
    print("END")


if __name__ == '__main__':
    main()
