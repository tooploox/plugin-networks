from pluginnet.sun397.sun397 import sun397_data_transform_val, CrossEntropyLossOneHot
from pluginnet.sun397.metrics import create_softmax_metrics_dictionary
from pluginnet.sun397.partial_evidence import build_plugins, AlexNetPartialEvidence, SUN397PE
from pluginnet.sun397.sun397 import SUN397
import torchvision
import torch
import torchvision.models as models
import pandas as pd


def create_mode_pe(conf):
    data_set, net, metrics_dict, aggregator = _create__alexnet_cross_entropy_model_pe(conf)
    return data_set, net, metrics_dict, aggregator


def create_mode_base(conf):
    data_set, net, metrics_dict, aggregator = _create__alexnet_cross_entropy_model_base(conf)
    return data_set, net, metrics_dict, aggregator


def _create__alexnet_cross_entropy_model_base(conf):
    def target_transform(x):
        return x[-397:]

    if conf['split'] == 'train':
        conf['split_file'] = conf['split_file_train']
    else:
        conf['split_file'] = conf['split_file_test']

    data_set = get_dataset_base(conf['dataset_root'], conf['split_file'], conf['split'],
                                conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                target_transform=target_transform, seed=conf.get('seed'))
    net = models.__dict__['alexnet'](num_classes=397)
    state_dict = torch.load(conf['base_model_file'])
    net.load_state_dict(state_dict)
    criterion = CrossEntropyLossOneHot()
    aggregator = PredictionAggregatorSUN397(data_set.samples)
    return data_set, net, create_softmax_metrics_dictionary(criterion), aggregator


def _create__alexnet_cross_entropy_model_pe(conf):
    def target_transform(x): return x[-397:]

    if conf['split'] == 'train':
        conf['split_file'] = conf['split_file_train']
    else:
        conf['split_file'] = conf['split_file_test']

    data_set = get_dataset(conf['dataset_root'], conf['split_file'], conf['split'],
                           conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                           target_transform=target_transform, seed=conf.get('seed'))
    base_net = models.__dict__['alexnet'](num_classes=397)
    plugins = build_plugins(conf['plugins'])
    net = AlexNetPartialEvidence(base_net, plugins)
    state_dict = torch.load(conf['base_model_file'])
    net.load_state_dict(state_dict)
    criterion = CrossEntropyLossOneHot()
    aggregator = PredictionAggregatorSUN397(data_set.samples)
    return data_set, net, create_softmax_metrics_dictionary(criterion), aggregator


def get_dataset(dataset_root, split_file, split, hierarchy_file, data_transform=sun397_data_transform_val,
                use_fraction=0, target_transform=None, seed=0):
    if seed is None:
        seed = 0

    loader_f = torchvision.datasets.folder.pil_loader

    if split == 'train':
        data_set = SUN397PE(dataset_root, split_file, hierarchy_file, split='train',
                            validation_size=0, transform=data_transform,
                            target_transform=target_transform, loader=loader_f,
                            use_fraction=use_fraction, random_seed=seed)
    else:
        data_set = SUN397PE(dataset_root, split_file, hierarchy_file, split=split,
                            validation_size=10, transform=data_transform,
                            target_transform=target_transform)
    return data_set


def get_dataset_base(dataset_root, split_file, split, hierarchy_file, data_transform=sun397_data_transform_val,
                     use_fraction=0, target_transform=None, seed=0):
    if seed is None:
        seed = 0

    loader_f = torchvision.datasets.folder.pil_loader

    if split == 'train':
        data_set = SUN397(dataset_root, split_file, hierarchy_file, split='train',
                          validation_size=0, transform=data_transform,
                          target_transform=target_transform, loader=loader_f,
                          use_fraction=use_fraction, random_seed=seed)
    else:
        data_set = SUN397(dataset_root, split_file, hierarchy_file, split=split,
                          validation_size=10, transform=data_transform,
                          target_transform=target_transform)
    return data_set


class PredictionAggregatorSUN397(object):

    def __init__(self, files=None):
        self.predictions = []
        self.ground_truth = []
        self.files = files

    def __call__(self, engine):
        self.add_result_(engine)

    def add_result_(self, engine):
        out = engine.state.output
        self.predictions.extend(out[0].detach().cpu().numpy())
        self.ground_truth.extend(out[1].detach().cpu().numpy())

    def save_results(self, file_name):
        predictions = pd.DataFrame(self.predictions)
        gt = pd.DataFrame(self.ground_truth)
        if self.files is not None:
            predictions = predictions.set_index(pd.Index(self.files))
            gt = gt.set_index(pd.Index(self.files))
        results = pd.concat([predictions, gt], axis=1, keys=['predictions', 'ground_truth'])
        results.to_hdf(file_name, key='results', mode='w')
