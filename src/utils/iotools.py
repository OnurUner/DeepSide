from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
import shutil
import pandas as pd

import torch

from .numpytools import np_accuracy


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(fpath)

    if is_best:
        torch.save(state, osp.join(fpath, 'best_model.pth.tar'))


def experiment_acc_csv(y_true, y_prob, drug_ids, exp_ids, meta_df):
    assert len(drug_ids) == len(exp_ids), "Drug ids count does not match with the count of experiment ids"
    columns = ["exp_id", "drug_id", "pert_idose", "pert_itime", "cell_id", "accuracy"]
    df = pd.DataFrame(columns=columns)
    df.set_index("exp_id", inplace=True)
    for i, (exp_id, drug_id) in enumerate(zip(exp_ids, drug_ids)):
        acc = np_accuracy(y_true[i, :], y_prob[i, :])
        dose = meta_df.loc[exp_id, "pert_idose"]
        time = meta_df.loc[exp_id, "pert_itime"]
        cell = meta_df.loc[exp_id, "cell_id"]
        df.loc[exp_id] = [drug_id, dose, time, cell, acc]
    return df


def create_drug_per_adr_csv(y_true, y_prob, drug_ids, adr_names):
    pass

