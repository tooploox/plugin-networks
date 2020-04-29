from ignite.engine import _prepare_batch, Engine
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import torch
import shutil
import glob
import torch.nn as nn


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item(),
                              gradient_clip=np.inf):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


class CheckpointManager(object):

    def __init__(self, dir_name, prefix, score_name, score_function, extra=None, save_freq=1):
        self.best_score_ = -1
        if extra is None:
            self.extra_ = {}
        else:
            self.extra_ = extra
        self.dir_name_ = dir_name
        self.prefix_ = prefix
        self.score_name_ = score_name
        self.get_score_ = score_function
        self.epoch_ = 0
        self.save_freq_ = save_freq
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def _update_best_score(self, score):
        if score > self.best_score_:
            self.best_score_ = score
            return True
        else:
            return False

    def __call__(self, engine, to_save):
        if self.epoch_ % self.save_freq_ != 0:
            return
        score = self.get_score_(engine)
        is_best = self._update_best_score(score)
        out_dict = {
            'epoch': self.epoch_,
            'best_score_value': self.best_score_,
            'score_value': score,
            'score_name': self.score_name_,
            'extra': self.extra_
        }
        state_dicts = {}
        for name, obj in to_save.items():
            state_dicts[name] = obj.state_dict()
        out_dict['state_dicts'] = state_dicts
        self.save_(out_dict, is_best)

    def save_(self, out_dict, is_best):
        fn = '{prefix}_{epoch:02d}_{score_name}_{score_val:.2f}.pth.tar'.format(prefix=self.prefix_,
                                                                                epoch=out_dict['epoch'],
                                                                                score_name=out_dict['score_name'],
                                                                                score_val=out_dict['score_value'])
        cur_path = os.path.join(self.dir_name_, fn)
        best_path = os.path.join(self.dir_name_, 'model_best.pth.tar')
        torch.save(out_dict, cur_path)
        if is_best:
            shutil.copyfile(cur_path, best_path)
            torch.save(out_dict['state_dicts']['model'],
                       os.path.join(self.dir_name_, 'model_best_state_dict.pth.tar'))

    def load(self, model_path):
        params = torch.load(model_path, map_location=lambda storage, location: self.move_to_device_(storage))
        self.best_score_ = params['best_score_value']
        self.epoch_ = params['epoch']
        return params['state_dicts']

    def load_last(self):
        if len(glob.glob(os.path.join(self.dir_name_, '*.pth.tar'))) > 0:
            last_checkpoint_file = sorted(glob.glob(os.path.join(self.dir_name_, 'model_[0-9]*.pth.tar')))[-1]
            return self.load(last_checkpoint_file)
        else:
            raise FileNotFoundError('No checkpoints')

    def is_checkpoint_available(self):
        return len(glob.glob(os.path.join(self.dir_name_, '*.pth.tar'))) > 0

    @staticmethod
    def move_to_device_(variable):
        if torch.cuda.is_available():
            return variable.cuda()
        else:
            return variable


def create_plugin_net(definition):
    layers = []
    for i in range(len(definition) - 1):
        if i > 0:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(definition[i], definition[i + 1]))
    return nn.Sequential(*layers)


def build_plugins(plugins_def):
    """
    Builds Plugin Networks based on definition
    :param plugins_def: dictionary of the following form:
    { 'layer_name: [plugin_layer0_size, plugin_layer1_size,...], ...} layer_size means number of neurons.
    Layer_name indicates to which layer from base model Plugin Network is attached.
    :return: list of Pluign Networks of form [('layer_name', Plugin Network)]
    """
    ret = []
    for layer, definition in plugins_def.items():
        plugin_net = create_plugin_net(definition)
        ret.append((layer, plugin_net))
    return ret


class Operator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, base_network_ouput, plugin_ouput):
        raise NotImplementedError()


class AdditiveOperator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, base_network_ouput, plugin_output):
        return base_network_ouput + plugin_output


class MultiplicativeOperator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, base_network_output, plugin_ouput):
        return base_network_output + plugin_ouput


class AffineOperator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, base_network_ouput, plugin_output):
        a = plugin_output[:, :int(plugin_output.shape[1] / 2)]
        b = plugin_output[:, int(plugin_output.shape[1] / 2):]
        return a * base_network_ouput + b


def operator_factory(operator_name):
    operators = {'additive': AdditiveOperator,
                 'multiplicative': MultiplicativeOperator,
                 'affine': AdditiveOperator}
    return operators[operator_name]()
