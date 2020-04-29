from ignite.metrics import Loss, Metric
import torchnet
import numpy as np
import torch.nn as nn


def filter_level1(output):
    y_pred, target = output
    y_pred = y_pred[:, :3]
    target = target[:, :3]
    return y_pred, target


def filter_level2(output):
    y_pred, target = output
    y_pred = y_pred[:, 3:19]
    target = target[:, 3:19]
    return y_pred, target


def filter_level3(output):
    y_pred, target = output
    y_pred = y_pred[:, 19:]
    target = target[:, 19:]
    return y_pred, target


class MCAcc(Metric):

    def __init__(self, output_transform=lambda x: x):
        self.meter = torchnet.meter.classerrormeter.ClassErrorMeter(accuracy=True)
        super().__init__(output_transform)

    def update(self, output):
        y_pred, target_one_hot = output
        if len(target_one_hot.shape) == 1:
            target = target_one_hot
        elif len(target_one_hot.shape) == 2:
            target = np.argmax(target_one_hot.cpu().numpy(), axis=1)
        else:
            raise ValueError("Wrong target size %s" % str(target_one_hot.shape))
        self.meter.add(y_pred.detach().cpu().numpy(), target)

    def reset(self):
        self.meter.reset()

    def compute(self):
        return self.meter.value(k=1)


class mAP(Metric):

    def __init__(self, output_transform=lambda x: x, single_label=True):
        self._single_label = single_label
        self.meter = torchnet.meter.mapmeter.mAPMeter()
        super().__init__(output_transform)

    def update(self, output):
        y_pred, target_one_hot = output
        if self._single_label:
            y_pred = nn.Sigmoid()(y_pred)
        else:
            y_pred = nn.Softmax()(y_pred)
        self.meter.add(y_pred.detach().cpu().numpy(), target_one_hot)

    def reset(self):
        self.meter.reset()

    def compute(self):
        return self.meter.value() * 100


def create_metrics_dictionary(criterion):
    return {'level1_accuracy': MCAcc(filter_level1),
            'level2_accuracy': MCAcc(filter_level2),
            'level3_accuracy': MCAcc(filter_level3),
            'level1_mAP': mAP(filter_level1),
            'level2_mAP': mAP(filter_level2),
            'level3_mAP': mAP(filter_level3, single_label=False),
            'level1_IOU': IOU(filter_level1),
            'level2_IOU': IOU(filter_level2),
            'level3_IOU': IOU(filter_level3),
            'nll': Loss(criterion)}


def create_softmax_metrics_dictionary(criterion):
    return {'level3_accuracy': MCAcc(),
            'level3_mAP': mAP(single_label=False),
            'nll': Loss(criterion)}


def get_map_score(engine):
    metrics = engine.state.metrics
    mAP_level3 = metrics['level3_mAP']
    return mAP_level3
