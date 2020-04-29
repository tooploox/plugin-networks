from pluginnet.sun397.sun397 import sun397_data_transform_val, sun397_data_transform_train, \
    CrossEntropyLossOneHot, SUN397
from pluginnet.sun397.metrics import create_softmax_metrics_dictionary
from pluginnet.common.model import build_plugins
import torchvision
import torch
import torchvision.models as models
import torch.nn as nn
from pluginnet.sun397.metrics import get_map_score
from pluginnet.common.model import operator_factory


"""
This module contains all the code specific to Partial Evidence processing for SUN397 dataset.
"""


class SUN397PE(SUN397):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        partial_evidence = self.targets[index][:3]
        return (partial_evidence, sample), target


def get_dataset(dataset_root, split_file_train, split_file_test, hierarchy_file,
                data_transform_train=sun397_data_transform_train, data_transform_val=sun397_data_transform_val,
                use_fraction=0, target_transform=None, seed=0):
    if seed is None:
        seed = 0

    loader_f = torchvision.datasets.folder.pil_loader

    train_set = SUN397PE(dataset_root, split_file_train, hierarchy_file, split='train',
                         validation_size=0, transform=data_transform_train,
                         target_transform=target_transform, loader=loader_f,
                         use_fraction=use_fraction, random_seed=seed)

    val_set = SUN397PE(dataset_root, split_file_test, hierarchy_file, split='val',
                       validation_size=10, transform=data_transform_val,
                       target_transform=target_transform, random_seed=seed)

    test_set = SUN397PE(dataset_root, split_file_test, hierarchy_file, split='test',
                        validation_size=10, transform=data_transform_val,
                        target_transform=target_transform, random_seed=seed)
    return train_set, val_set, test_set


def create_model(conf):
    train_set, val_set, net, criterion, metrics_dict = _create__alexnet_cross_entropy_model(conf)
    return train_set, val_set, net, criterion, metrics_dict, ('mAP', get_map_score)


def _create__alexnet_cross_entropy_model(conf):
    def target_transform(x): return x[-397:]

    train_set, val_set, _ = get_dataset(conf['dataset_root'], conf['split_file_train'], conf['split_file_test'],
                                        conf['hierarchy_file'], use_fraction=conf.get('train_set_size'),
                                        target_transform=target_transform, seed=conf.get('seed'))
    base_net = models.__dict__['alexnet'](num_classes=397)
    state_dict = torch.load(conf['base_model_file'])
    base_net.load_state_dict(state_dict)
    plugins = build_plugins(conf['plugins'])
    net = AlexNetPartialEvidence(base_net, plugins)
    criterion = CrossEntropyLossOneHot()
    return train_set, val_set, net, criterion, create_softmax_metrics_dictionary(criterion)


class AlexNetPartialEvidence(nn.Module):
    """
    AlexNet model modified to work with Plugin Networks
    """

    def __init__(self, base_model, plugins, operator='additive'):
        """

        :param base_model: Base standard AlexNet model
        :param plugins: list of Plugin Netowrks to be attached. List should contain tuples:
        (layer_name, Plugin Netowrk model). Layer_name indicates to which layer from base model Plugin Network is attached.
        Plugin network model, can be any neural network.
        :param operator: fusion operator (see in paper).
        """
        super(AlexNetPartialEvidence, self).__init__()
        self.operator = operator_factory(operator)
        for p in base_model.parameters():
            p.requires_grad = False
        self.layers = []

        self.conv1 = list(base_model.features.children())[0]
        self.layers.append(('conv1', self.conv1))
        self.relu1 = list(base_model.features.children())[1]
        self.layers.append(('relu1', self.relu1))
        self.maxpool1 = list(base_model.features.children())[2]
        self.layers.append(('maxpool1', self.maxpool1))
        self.conv2 = list(base_model.features.children())[3]
        self.layers.append(('conv2', self.conv2))
        self.relu2 = list(base_model.features.children())[4]
        self.layers.append(('relu2', self.relu2))
        self.maxpool2 = list(base_model.features.children())[5]
        self.layers.append(('maxpool2', self.maxpool2))
        self.conv3 = list(base_model.features.children())[6]
        self.layers.append(('conv3', self.conv3))
        self.relu3 = list(base_model.features.children())[7]
        self.layers.append(('relu3', self.relu3))
        self.conv4 = list(base_model.features.children())[8]
        self.layers.append(('conv4', self.conv4))
        self.relu4 = list(base_model.features.children())[9]
        self.layers.append(('relu4', self.relu4))
        self.conv5 = list(base_model.features.children())[10]
        self.layers.append(('conv5', self.conv5))
        self.relu5 = list(base_model.features.children())[11]
        self.layers.append(('relu5', self.relu5))
        self.maxpool3 = list(base_model.features.children())[12]
        self.layers.append(('maxpool3', self.maxpool3))

        self.dropout1 = list(base_model.classifier.children())[0]
        self.layers.append(('dropout1', self.dropout1))
        self.linear1 = list(base_model.classifier.children())[1]
        self.layers.append(('linear1', self.linear1))
        self.relu6 = list(base_model.classifier.children())[2]
        self.layers.append(('relu6', self.relu6))
        self.dropout2 = list(base_model.classifier.children())[3]
        self.layers.append(('dropout2', self.dropout2))
        self.linear2 = list(base_model.classifier.children())[4]
        self.layers.append(('linear2', self.linear2))
        self.relu7 = list(base_model.classifier.children())[5]
        self.layers.append(('relu7', self.relu7))
        self.linear3 = list(base_model.classifier.children())[6]
        self.layers.append(('linear3', self.linear3))
        # self.sigmoid = nn.Sigmoid()
        # self.layers.append(('sigmoid', self.sigmoid))

        self.plugins = plugins
        for i, (_, plugin) in enumerate(self.plugins):
            self.add_module('plugin%d' % i, plugin)
        self.plugins_dict = dict(self.plugins)

    def forward(self, x_in):
        partial_evidence, img = x_in
        x = img

        for layer_name, layer in self.layers:
            if layer_name == 'dropout1':
                x = x.view(x.size(0), 256 * 6 * 6)
            x = layer(x)

            if layer_name in self.plugins_dict.keys():
                plugin_layer = self.plugins_dict[layer_name]
                plugin_output = plugin_layer(partial_evidence)
                if layer_name[:4] == 'conv':
                    plugin_output = plugin_output.view(x.size(0), -1, 1, 1)

                # Here fusion with plugin happens
                x = self.operator(x, plugin_output)
        activations = x
        return activations

    def get_trainable_params(self):
        for _, p in self.plugins:
            for param in p.parameters():
                yield param
