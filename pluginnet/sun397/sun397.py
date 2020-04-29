import torch.utils.data
import os.path
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from pluginnet.sun397.metrics import create_metrics_dictionary, create_softmax_metrics_dictionary
from pluginnet.sun397.metrics import get_map_score
"""
This module contains the code which processes SUN397 dataset
"""

sun397_data_transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

sun397_data_transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


def get_dataset(dataset_root, split_file_train, split_file_test, hierarchy_file,
                data_transform_train=sun397_data_transform_train, data_transform_val=sun397_data_transform_val,
                use_fraction=0, target_transform=None, seed=0):
    if seed is None:
        seed = 0

    loader_f = torchvision.datasets.folder.pil_loader

    train_set = SUN397(dataset_root, split_file_train, hierarchy_file, split='train',
                       validation_size=0, transform=data_transform_train,
                       target_transform=target_transform, loader=loader_f,
                       use_fraction=use_fraction, random_seed=seed)

    val_set = SUN397(dataset_root, split_file_test, hierarchy_file, split='val',
                     validation_size=10, transform=data_transform_val,
                     target_transform=target_transform, random_seed=seed)

    test_set = SUN397(dataset_root, split_file_test, hierarchy_file, split='test',
                      validation_size=10, transform=data_transform_val,
                      target_transform=target_transform, random_seed=seed)
    return train_set, val_set, test_set


class SUN397(torch.utils.data.Dataset):
    """
    Custom loader for SUN397 dataset.
    """

    def __init__(self, root, split_file, hierarchy_file, loader=torchvision.datasets.folder.pil_loader,
                 split='train', transform=None, target_transform=None, validation_size=10, random_seed=0,
                 use_fraction=0):
        self.root = root
        self.split_file = split_file
        self.split = split
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.targets = []

        self.samples = pd.read_csv(split_file, header=None)
        labels = pd.read_csv(hierarchy_file, delimiter=';', header=[0, 1], index_col=0)
        level3_label_index = dict([(k, l) for l, k in enumerate(labels.index)])

        random_state = np.random.RandomState(random_seed)
        val_idx = self._get_val_indices(self.samples, level3_label_index, random_state, validation_size)
        if split == 'val':
            self.samples = self.samples.iloc[val_idx]
        elif split == 'test':
            train_idx = set(range(self.samples.shape[0])) - set(val_idx)
            train_idx = sorted(train_idx)
            self.samples = self.samples.iloc[train_idx]
        elif split == 'train':
            if use_fraction is not None and use_fraction > 1:
                tr_idx = self._get_val_indices(self.samples, level3_label_index, random_state, use_fraction)
                self.samples = self.samples.iloc[tr_idx]
            pass
        else:
            raise ValueError('Wrong split name: {}. You can use only "val" or "train".'.format(split))

        self.samples = self.samples.values.squeeze().tolist()

        for sample in self.samples:
            one_hot_level1_level2_labels = labels.loc[os.path.dirname(sample)].values
            level3_label = level3_label_index[os.path.dirname(sample)]
            level3_one_hot_label = np.zeros(labels.shape[0]).astype(np.float32)
            level3_one_hot_label[level3_label] = 1
            self.targets.append(np.hstack([one_hot_level1_level2_labels, level3_one_hot_label]).astype(np.float32))

    @staticmethod
    def _get_val_indices(samples, level3_label_index, random_state, count_per_class=10):
        groupped = samples.apply(lambda x: pd.Series([level3_label_index[os.path.dirname(x[0])], 1]), axis=1)
        it = groupped.groupby(0).count()
        cum_sum = 0
        indices = []
        for i in it.iterrows():
            indices.extend(random_state.choice(range(cum_sum, cum_sum + i[1][1]), count_per_class, replace=False))
            cum_sum += i[1][1]
        return indices

    def __getitem__(self, index):
        path = os.path.join(self.root, self.samples[index][1:])
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_num_classes(self):
        if self.target_transform is not None:
            targets_local = [self.target_transform(t) for t in self.targets]
        else:
            targets_local = self.targets
        return targets_local[0].shape[0]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Split file: {}\n'.format(self.split_file)
        fmt_str += '    Split: {}\n'.format(self.split)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def _alex_surgery(model, num_classes, train_only_last_layer):
    in_features = model.classifier[-1].in_features
    layers = list(model.classifier.children())[:-1]
    layers.extend([nn.Linear(in_features, num_classes)])
    model.classifier = nn.Sequential(*layers)

    if train_only_last_layer:
        for p in model.parameters():
            p.requires_grad = False
        for l in list(model.classifier.children())[-2:]:
            for p in l.parameters():
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
    return model


class CrossEntropyLossOneHot(nn.CrossEntropyLoss):

    def forward(self, input, target):
        _, t = target.max(1)
        return super().forward(input, t)


def create_alexnet_places365(model_file, num_classes=416, train_only_last_layer=False):
    model = models.__dict__['alexnet'](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return _alex_surgery(model, num_classes, train_only_last_layer)


def set_train_only_last(net):
    for p in net.parameters():
        p.requires_grad = False
    for l in list(net.classifier.children())[-2:]:
        for p in l.parameters():
            p.requires_grad = True
    return net


def create_model(conf):
    if conf['base_model_type'] == 'places365' and conf['criterion'] == 'BCEWithLogits':
        train_set, val_set, net, criterion, metrics_dict = _create__places365_logitloss_model(conf)
    elif conf['base_model_type'] == 'places365' and conf['criterion'] == 'cross_entropy':
        train_set, val_set, net, criterion, metrics_dict = _create__places365_cross_entropy_model(conf)
    elif conf['base_model_type'] == 'alexnet' and conf['criterion'] == 'BCEWithLogits':
        train_set, val_set, net, criterion, metrics_dict = _create__alexnet_logit_model(conf)
    elif conf['base_model_type'] == 'alexnet' and conf['criterion'] == 'cross_entropy':
        train_set, val_set, net, criterion, metrics_dict = _create__alexnet_cross_entropy_model(conf)
    else:
        raise ValueError('Incorrect configuration cannot create model')

    if conf['train_only_last_layer']:
        net = set_train_only_last(net)

    return train_set, val_set, net, criterion, metrics_dict, ('mAP', get_map_score)


def _create__places365_logitloss_model(conf):
    train_set, val_set, _ = get_dataset(conf['dataset_root'], conf['split_file_train'], conf['split_file_test'],
                                        conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                        seed=conf.get('seed'))
    net = create_alexnet_places365(conf['base_model_file'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(([3] * 3) + ([16] * 16) + ([397] * 397)))
    return train_set, val_set, net, criterion, create_metrics_dictionary(criterion)


def _create__alexnet_logit_model(conf):
    train_set, val_set, _ = get_dataset(conf['dataset_root'], conf['split_file_train'], conf['split_file_test'],
                                        conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                        seed=conf.get('seed'))
    net = models.__dict__['alexnet'](num_classes=416)
    state_dict = torch.load(conf['base_model_file'])
    net.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(([3] * 3) + ([16] * 16) + ([397] * 397)))
    return train_set, val_set, net, criterion, create_metrics_dictionary(criterion)


def _create__alexnet_cross_entropy_model(conf):
    def target_transform(x): return x[-397:]

    train_set, val_set, _ = get_dataset(conf['dataset_root'], conf['split_file_train'], conf['split_file_test'],
                                        conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                        target_transform=target_transform, seed=conf.get('seed'))
    net = models.__dict__['alexnet'](num_classes=397)
    state_dict = torch.load(conf['base_model_file'])
    net.load_state_dict(state_dict)
    criterion = CrossEntropyLossOneHot()
    return train_set, val_set, net, criterion, create_softmax_metrics_dictionary(criterion)


def _create__places365_cross_entropy_model(conf):
    def target_transform(x): return x[-397:]

    train_set, val_set, _ = get_dataset(conf['dataset_root'], conf['split_file_train'], conf['split_file_test'],
                                        conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                        target_transform=target_transform, seed=conf.get('seed'))
    net = create_alexnet_places365(conf['base_model_file'], num_classes=397)
    criterion = CrossEntropyLossOneHot()
    return train_set, val_set, net, criterion, create_softmax_metrics_dictionary(criterion)
