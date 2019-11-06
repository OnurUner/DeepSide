import os
import sys
import time
import pickle
import datetime
import argparse
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import src.data_manager as data_manager
from src.data_manager.utils import load_meta
from src.utils.torchtools import count_num_param
from src.utils.numpytools import hammingLoss, dropColumns, f2_score
from src.utils.logger import Logger
from src.utils.iotools import experiment_acc_csv

from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='DeepSide TEST')

# Miscs
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--log_path', type=str, metavar='PATH',
                    default='./../log/mlpsidenet/GE_03:08AM on May 07, 2019')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pos-count', type=int, default=-1,
                    help="prune side effects if less than pos count")
parser.add_argument('--test-signatures', default=True,
                    help='Test with only strongest experiments')
parser.add_argument('--export-csv', default=False,
                    help='Export micro auc/accuracy per drug results to csv')
parser.add_argument('--export-pickle', default=True,
                    help='Export prediction/ground truths')
# global variables
args = parser.parse_args()

sys.path.insert(0, os.path.join(args.log_path, 'src'))
import src.models as models


def test(cfg, model, testloader, use_gpu):
    y_true = []
    y_prob = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if type(data) is list:
                if len(data) == 2:
                    data_ge, data_cs = data
                    if use_gpu:
                        data_ge = data_ge.cuda()
                        data_cs = data_cs.cuda()
                        labels = labels.cuda()
                    outputs = model(data_ge, data_cs)
                elif len(data) == 3:
                    data_ge, data_cs, data_meta = data
                    if use_gpu:
                        data_ge = data_ge.cuda()
                        data_cs = data_cs.cuda()
                        data_meta = data_meta.cuda()
                        labels = labels.cuda()
                    outputs = model(data_ge, data_cs, data_meta)
            else:
                if use_gpu:
                    data, labels = data.cuda(), labels.cuda(non_blocking=True)

                if cfg["name"] == 'mlpontology':
                    ontology_vectors = testloader.dataset.ontology_vectors
                    ontology_vectors = torch.from_numpy(ontology_vectors).type(torch.FloatTensor)
                    ontology_vectors = ontology_vectors.cuda()
                    outputs = model(data, ontology_vectors)
                else:
                    outputs = model(data)
            y_true.append(labels.data.cpu().numpy())
            y_prob.append(outputs.data.cpu().numpy())

    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    return y_true, y_prob


if __name__ == '__main__':
    results = {}
    macro_auc_scores = []
    macro_map_scores = []
    micro_auc_scores = []
    micro_map_scores = []
    fbeta_scores = []
    hamming_losses = []
    y_true_list = []
    y_pred_list = []
    result_df_list = []
    drug_id_list = []

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    checkpoint = torch.load(osp.join(args.log_path, "fold0", "best_model.pth.tar"))
    dataset_config = checkpoint["dataset_config"]
    model_config = checkpoint["model_config"]
    dataset_config["test_signatures"] = args.test_signatures

    dataset = data_manager.init_dataset(**dataset_config)
    model = models.init_model(**model_config)

    sys.stdout = Logger(osp.join(args.log_path, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("==========\nModel Args:{}\n==========".format(model_config))
    print("==========\nDataset Args:{}\n==========".format(dataset_config))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Model size: {:.3f} M".format(count_num_param(model)))

    for fold_i, (_, test_dataset) in enumerate(dataset):
        best_path = osp.join(args.log_path, "fold" + str(fold_i), "best_model.pth.tar")
        checkpoint = torch.load(best_path)
        model_config = checkpoint["model_config"]
        testloader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            drop_last=False
        )

        model = models.init_model(**model_config)
        model_dict = checkpoint['state_dict']
        model.load_state_dict(model_dict, False)
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['auc']
        print("Loaded checkpoint from '{}'".format(best_path))
        print("- start_epoch: {}\n- auc: {}".format(start_epoch, best_auc))

        if use_gpu:
            model = nn.DataParallel(model).cuda()

        y_true, y_prob = test(model_config, model, testloader, use_gpu)
        y_pred_list.append(y_prob)
        y_true_list.append(y_true)
        drug_id_list.extend(test_dataset.drug_ids)

        if args.export_csv:
            meta_df = load_meta()
            result_df_list.append(
                experiment_acc_csv(y_true, y_prob, test_dataset.drug_ids, test_dataset.exp_ids, meta_df))

        if args.pos_count > 0:
            y_true, y_prob = dropColumns(y_true, y_prob, args.pos_count)

        macro_auc_score = np.mean(roc_auc_score(y_true, y_prob, average=None))
        micro_auc_score = roc_auc_score(y_true, y_prob, average="micro")
        macro_map_score = np.mean(average_precision_score(y_true, y_prob, average=None))
        micro_map_score = average_precision_score(y_true, y_prob, average="micro")
        hamming_loss_score = hammingLoss(y_prob, y_true)
        f2score = f2_score(y_true, np.array(y_prob))

        macro_auc_scores.append(macro_auc_score)
        micro_auc_scores.append(micro_auc_score)
        macro_map_scores.append(macro_map_score)
        micro_map_scores.append(micro_map_score)
        hamming_losses.append(hamming_loss_score)
        fbeta_scores.append(f2score)
        print("=======================")

    results["macro_auc_scores"] = macro_auc_scores
    results["micro_auc_scores"] = micro_auc_scores
    results["macro_map_scores"] = macro_map_scores
    results["micro_map_scores"] = micro_map_scores
    results["hamming_loss"] = hamming_losses
    results["y_pred_list"] = y_pred_list
    results["y_true_list"] = y_true_list
    results["f2_scores"] = fbeta_scores

    print( "Macro Mean AUC:", np.mean(macro_auc_scores))
    print( "Micro Mean AUC:", np.mean(micro_auc_scores))
    print( "Macro MAP:", np.mean(macro_map_scores))
    print( "Micro MAP:", np.mean(micro_map_scores))
    print( "Hamming loss:", np.mean(hamming_losses))
    print( "F2 Score:", np.mean(fbeta_scores))

    y_true_arr = np.vstack(y_true_list)
    y_pred_arr = np.vstack(y_pred_list)

    print("F2 Score:", f2_score(y_true_arr, np.array(y_pred_arr)))
    if args.export_csv:
        result_df = pd.concat(result_df_list)
        result_df.to_csv(os.path.join(args.log_path, "per_pert_test_results.csv"))

    if args.export_pickle:
        results_dict = {"y_true": y_true_arr,
                        "y_pred": y_pred_arr,
                        "adr_names": test_dataset.adr_names,
                        "drug_ids": drug_id_list}
        with open(osp.join(args.log_path, 'predictions.pickle'), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
