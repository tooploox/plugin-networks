#!/usr/bin/env python3
import argparse
import os
import torch
from pluginnet.sun397.testing import create_mode_pe as sun397_create_mode_pe
from pluginnet.sun397.testing import create_mode_base as sun397_create_mode_base
from ignite.engine import Events, create_supervised_evaluator
from tqdm import tqdm
from pluginnet.common.metrics import log_results
import json


tasks = {'sun397_pe': sun397_create_mode_pe, 'sun397': sun397_create_mode_base}


def load_conf(conf_file):
    with open(conf_file, 'r') as fd:
        conf = json.load(fd)
        return conf


def main():
    parser = argparse.ArgumentParser(description='PyTorch training script for SUN397 dataset')
    parser.add_argument('model_dir')
    parser.add_argument('split', choices=['train', 'val', 'test'])
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size')

    args = parser.parse_args()

    conf = load_conf(os.path.join(args.model_dir, 'conf.json'))
    conf['split'] = args.split
    conf['base_model_file'] = os.path.join(args.model_dir, 'model_best_state_dict.pth.tar')
    test_set, net, metrics_dict, aggregator = tasks[conf['task']](conf)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                                              drop_last=False)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    evaluator = create_supervised_evaluator(net,
                                            metrics=metrics_dict,
                                            device=device)

    desc = "ITERATION"
    pbar = tqdm(
        initial=0, leave=False, total=len(test_loader),
        desc=desc.format(0)
    )
    log_interval = 10

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(test_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc
            pbar.update(log_interval)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        log_results(engine, evaluator, "Testing")
        pbar.n = pbar.last_print_n = 0

    evaluator.add_event_handler(Events.ITERATION_COMPLETED, aggregator)
    evaluator.run(test_loader)

    aggregator.save_results(os.path.join(args.model_dir, '%s.h5' % args.split))


if __name__ == '__main__':
    main()
