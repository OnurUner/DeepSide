from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize, gamma=0.1):
    # decay learning rate by 'gamma' for every 'stepsize'
    lr = base_lr * (gamma ** (epoch // stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param


def calc_multiclass_accuracy(out, labels):
    outputs = torch.argmax(out, dim=1)
    return torch.sum(torch.eq(outputs, labels)).item() / float(labels.size(0))


def calc_binary_accuracy(out, labels):
    outputs = out.data.clone()
    outputs[outputs >= 0.5] = 1.0
    outputs[outputs < 0.5] = 0.0
    return torch.mean(torch.eq(outputs, labels).type(torch.FloatTensor))
