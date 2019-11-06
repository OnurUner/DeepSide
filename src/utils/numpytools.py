# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, fbeta_score


def dropColumns(y_true, y_prob, min_count):
    del_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i]) <= min_count:
            del_list.append(i)
    return np.delete(y_true, del_list, axis=1), np.delete(y_prob, del_list, axis=1)


def hammingLoss(out, y_test):
    threshold = np.arange(0.1,1.0,0.1)
    h_losses = []
    for thr in threshold:
        y_temp = np.array(out)
        y_temp[y_temp >= thr] = 1
        y_temp[y_temp < thr] = 0
        h_losses.append(hamming_loss(y_test, y_temp))
    return np.min(h_losses)


def np_accuracy(y_true, y_pred):
    _y_pred = y_pred.copy()
    _y_pred[_y_pred >= 0.5] = 1.0
    _y_pred[_y_pred < 0.5] = 0.0
    return accuracy_score(y_true, _y_pred)


def f2_score(y_true, y_pred):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    return fbeta_score(y_true, y_pred, beta=2, average='samples')