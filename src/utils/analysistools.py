import os

import pickle
import numpy as np
import pandas as pd
from PIL import Image
from pandas import ExcelWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score

from numpytools import hammingLoss
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, classification_report


current_path = os.path.dirname(os.path.abspath(__file__))
cs_pickle_path = os.path.join(current_path, "../../log/mlpsidenet/CS_08:50PM on May 06, 2019/predictions.pickle")
ge_pickle_path = os.path.join(current_path, "../../log/mm_joint_attn/BEST_GE_CS_META_02:43AM on May 06, 2019/predictions.pickle")
# ge_pickle_path = os.path.join(current_path, "../../log/mm_sum/GE_CS_META_04:09PM on February 25, 2019/predictions.pickle")
# ge_pickle_path = os.path.join(current_path, "../../log/mm_joint/GE_CS_META_09:19PM on April 01, 2019/predictions.pickle")
cs_ontology_pickle_path =  os.path.join(current_path, "../../log/mlpontology/CS_02:00AM on March 19, 2019/predictions.pickle")


def load_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        pickle_file = pickle.load(f)
    
    y_true = pickle_file['y_true']
    y_prob = pickle_file['y_pred']
    adr_names = pickle_file['adr_names']
    drug_ids = pickle_file['drug_ids']
    
    return y_true, y_prob, adr_names, drug_ids


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def get_recall(results):
    y_true = results['y_true']
    y_prob = results['y_pred']
    adr_names = results['adr_names']
    drug_ids = results['drug_ids']
    
    y_pred = np.array(y_prob)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    results = np.equal(y_true, y_pred).astype(np.uint8)
    results = results * y_true
    results_csv = pd.DataFrame(data=results, index=drug_ids, columns=adr_names)
    return results_csv


def write_prob_pred_truth(writer, y_true, y_prob, adr_names, drug_ids, predef):
    y_pred = np.array(y_prob)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    
    true_df = pd.DataFrame(data=y_true, index=drug_ids, columns=adr_names)
    prob_df = pd.DataFrame(data=y_prob, index=drug_ids, columns=adr_names)
    pred_df = pd.DataFrame(data=y_pred, index=drug_ids, columns=adr_names)
    
    true_df.to_excel(writer, predef+' Ground Truth')
    prob_df.to_excel(writer, predef+' Probabilities')
    pred_df.to_excel(writer, predef+' Binary Predictions')


def write_class_based_true_predictions(cs_probs, ge_probs, y_true, adr_names, drug_ids, prename1, prename2):
    data_shape = cs_probs.shape
    cs_true_preds = []
    ge_true_preds = []
    for row_i in range(data_shape[0]):
        for col_i in range(data_shape[1]):
            if y_true[row_i, col_i] == 1:
                if cs_probs[row_i, col_i] > 0.8 and ge_probs[row_i, col_i] < 0.4:
                    cs_true_preds.append((drug_ids[row_i], adr_names[col_i], cs_probs[row_i, col_i], ge_probs[row_i, col_i]))
                elif ge_probs[row_i, col_i] > 0.8 and cs_probs[row_i, col_i] < 0.4:
                    ge_true_preds.append((drug_ids[row_i], adr_names[col_i], ge_probs[row_i, col_i], cs_probs[row_i, col_i]))
    
    cs_truths = pd.DataFrame({'Drug IDs': [item[0] for item in cs_true_preds],
                              'ADR Name': [item[1] for item in cs_true_preds],
                              prename1 + ' Probability': [item[2] for item in cs_true_preds],
                              prename2 + ' Probability': [item[3] for item in cs_true_preds]})

    ge_truths = pd.DataFrame({'Drug IDs': [item[0] for item in ge_true_preds],
                              'ADR Name': [item[1] for item in ge_true_preds],
                              prename2 + ' Probability': [item[2] for item in ge_true_preds],
                              prename1 + ' Probability': [item[3] for item in ge_true_preds]})
    return cs_truths, ge_truths


def write_class_based_most_probable_predictions(cs_probs, ge_probs, y_true, adr_names, drug_ids, prename1, prename2):
    data_shape = cs_probs.shape
    cs_true_preds = []
    ge_true_preds = []
    for row_i in range(data_shape[0]):
        for col_i in range(data_shape[1]):
            if y_true[row_i, col_i] == 0:
                if cs_probs[row_i, col_i] > 0.80 and ge_probs[row_i, col_i] < 0.5:
                    cs_true_preds.append(
                        (drug_ids[row_i], adr_names[col_i], cs_probs[row_i, col_i], ge_probs[row_i, col_i]))
                elif ge_probs[row_i, col_i] > 0.80 and cs_probs[row_i, col_i] < 0.5:
                    ge_true_preds.append(
                        (drug_ids[row_i], adr_names[col_i], ge_probs[row_i, col_i], cs_probs[row_i, col_i]))

    cs_truths = pd.DataFrame({'Drug IDs': [item[0] for item in cs_true_preds],
                              'ADR Name': [item[1] for item in cs_true_preds],
                              prename1 + ' Probability': [item[2] for item in cs_true_preds],
                              prename2 + ' Probability': [item[3] for item in cs_true_preds]})

    ge_truths = pd.DataFrame({'Drug IDs': [item[0] for item in ge_true_preds],
                              'ADR Name': [item[1] for item in ge_true_preds],
                              prename2 + ' Probability': [item[2] for item in ge_true_preds],
                              prename1 + ' Probability': [item[3] for item in ge_true_preds]})
    return cs_truths, ge_truths


def get_accuracy(y_true, y_prob, names, type='DRUG'):
    if type == 'DRUG':
        it_len = y_true.shape[0]
        name = 'Drug IDS'
    elif type == 'SE':
        it_len = y_true.shape[1]
        name = 'Side Effect'

    y_pred = np.array(y_prob)
    y_pred[y_pred>=0.5] = 1.
    y_pred[y_pred<0.5] = 0.
    acc_list = []
    precision_list = []
    recall_list = []
    fscore_list = []
    support_list = []
    f2score_list = []
    for i in range(it_len):
        if type == 'DRUG':
            acc_list.append(accuracy_score(y_true[i], y_pred[i]))
            precision, recall, fscore, support = precision_recall_fscore_support(y_true[i], y_pred[i], average=None)
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)
            support_list.append(support)
            # f2score_list.append(f2_score(y_true[i], y_pred[i]))
            f2score_list.append(0)
        elif type == 'SE':
            acc_list.append(accuracy_score(y_true[:, i], y_pred[:, i]))
            precision, recall, fscore, support = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average=None)
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)
            support_list.append(support)
            # f2score_list.append(f2_score(y_true[:, i], y_pred[:, i]))
            f2score_list.append(0)

    column_names = [name, 'Accuracy', 'Precision Class 0', 'Precision Class 1', 'Recall Class 0', 'Recall Class 1',
                    'FScore Class 0', 'FScore Class 1', 'F2 Score', 'Support Class 0', 'Support Class 1']
    acc_df = pd.DataFrame({column_names[0]: [item for item in names],
                           column_names[1]: [item for item in acc_list],
                           column_names[2]: [item[0] for item in precision_list],
                           column_names[3]: [item[1] for item in precision_list],
                           column_names[4]: [item[0] for item in recall_list],
                           column_names[5]: [item[1] for item in recall_list],
                           column_names[6]: [item[0] for item in fscore_list],
                           column_names[7]: [item[1] for item in fscore_list],
                           column_names[8]: [item for item in f2score_list],
                           column_names[9]: [item[0] for item in support_list],
                           column_names[10]: [item[1] for item in support_list]})
    acc_df = acc_df[column_names]
    acc_df = acc_df.set_index(name)
    acc_df = acc_df.sort_values(by=['Accuracy'], ascending=False)
    return acc_df


def side_effect_accuracy(y_true, y_prob, names):
    y_pred = np.array(y_prob)
    y_pred[y_pred>=0.5] = 1.
    y_pred[y_pred<0.5] = 0.
    acc_list = []
    for i in range(y_true.shape[1]):
        acc_list.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    acc_df = pd.DataFrame({'Drug IDS': [item for item in names],
                           'Accuracy': [item for item in acc_list]})
    acc_df = acc_df.sort_values(by=['Accuracy'], ascending=False)
    return acc_df


def cs_vs_ge_cs_meta():
    writer = ExcelWriter('CS_vs_GE-CS-META.xlsx')

    cs_true, cs_prob, cs_adr_names, cs_drug_ids = load_pickle(cs_pickle_path)
    ge_true, ge_prob, ge_adr_names, ge_drug_ids = load_pickle(ge_pickle_path)
    assert (cs_drug_ids == ge_drug_ids) and (cs_adr_names == ge_adr_names), "Unordered drugs and adrs"

    cs_drug_acc = get_accuracy(cs_true, cs_prob, cs_drug_ids, 'DRUG')
    ge_drug_acc = get_accuracy(ge_true, ge_prob, ge_drug_ids, 'DRUG')

    ge_drug_variation = ge_drug_acc - cs_drug_acc
    ge_drug_variation['Support Class 0'] = ge_drug_acc['Support Class 0']
    ge_drug_variation['Support Class 1'] = ge_drug_acc['Support Class 1']

    cs_side_acc = get_accuracy(cs_true, cs_prob, cs_adr_names, 'SE')
    ge_side_acc = get_accuracy(ge_true, ge_prob, ge_adr_names, 'SE')

    ge_side_variation = ge_side_acc - cs_side_acc
    ge_side_variation['Support Class 0'] = ge_side_acc['Support Class 0']
    ge_side_variation['Support Class 1'] = ge_side_acc['Support Class 1']

    ge_drug_variation.to_excel(writer, 'GE Drug Based Variations')
    ge_side_variation.to_excel(writer, 'GE SE Based Variations')

    cs_drug_acc.to_excel(writer, 'CS Model Drug Based Metrics')
    ge_drug_acc.to_excel(writer, 'GE Model Drug Based Metrics')

    cs_side_acc.to_excel(writer, 'CS Model SE Based Metrics')
    ge_side_acc.to_excel(writer, 'GE Model SE Based Metrics')

    cs_true_df, ge_true_df = write_class_based_true_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'CS', 'GE_META_CS')
    cs_probable_df, ge_probable_df = write_class_based_most_probable_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'CS', 'GE_META_CS')

    write_prob_pred_truth(writer, cs_true, cs_prob, cs_adr_names, cs_drug_ids, 'CS')
    write_prob_pred_truth(writer, ge_true, ge_prob, ge_adr_names, ge_drug_ids, 'GE_META_CS')
    cs_true_df.to_excel(writer, 'Best Results of CS')
    ge_true_df.to_excel(writer, 'Best Results of GE_META_CS')

    cs_probable_df.to_excel(writer, 'Most Probable FP CS')
    ge_probable_df.to_excel(writer, 'Most Probable FP GE_META_CS')
    writer.save()


def cs_vs_cs_ontology():
    writer = ExcelWriter('CS_vs_CS-Ontology.xlsx')

    cs_true, cs_prob, cs_adr_names, cs_drug_ids = load_pickle(cs_pickle_path)
    ge_true, ge_prob, ge_adr_names, ge_drug_ids = load_pickle(cs_ontology_pickle_path)
    assert (cs_drug_ids == ge_drug_ids), "Unordered drugs and adrs"

    cs_true, cs_prob = calc_metrics(cs_true, cs_prob, cs_adr_names, ge_adr_names)
    _, _ = calc_metrics(ge_true, ge_prob, ge_adr_names, cs_adr_names)
    cs_adr_names = ge_adr_names

    cs_drug_acc = get_accuracy(cs_true, cs_prob, cs_drug_ids, 'DRUG')
    ge_drug_acc = get_accuracy(ge_true, ge_prob, ge_drug_ids, 'DRUG')

    cs_side_acc = get_accuracy(cs_true, cs_prob, cs_adr_names, 'SE')
    ge_side_acc = get_accuracy(ge_true, ge_prob, ge_adr_names, 'SE')

    cs_drug_acc.to_excel(writer, 'CS Model Drug Based Metrics')
    ge_drug_acc.to_excel(writer, 'GE Model Drug Based Metrics')

    cs_side_acc.to_excel(writer, 'CS Model SE Based Metrics')
    ge_side_acc.to_excel(writer, 'GE Model SE Based Metrics')

    cs_true_df, ge_true_df = write_class_based_true_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'CS', 'CS-Ontology')
    cs_probable_df, ge_probable_df = write_class_based_most_probable_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'CS', 'CS-Ontology')

    write_prob_pred_truth(writer, cs_true, cs_prob, cs_adr_names, cs_drug_ids, 'CS')
    write_prob_pred_truth(writer, ge_true, ge_prob, ge_adr_names, ge_drug_ids, 'CS-Ontology')
    cs_true_df.to_excel(writer, 'Best Results of CS')
    ge_true_df.to_excel(writer, 'Best Results of CS-Ontology')

    cs_probable_df.to_excel(writer, 'Most Probable FP CS')
    ge_probable_df.to_excel(writer, 'Most Probable FP CS-Ontology')
    writer.save()


def ge_cs_meta_vs_cs_ontology():
    writer = ExcelWriter('GE-CS-META_vs_CS-Ontology.xlsx')

    cs_true, cs_prob, cs_adr_names, cs_drug_ids = load_pickle(ge_pickle_path)
    ge_true, ge_prob, ge_adr_names, ge_drug_ids = load_pickle(cs_ontology_pickle_path)
    assert (cs_drug_ids == ge_drug_ids), "Unordered drugs and adrs"

    cs_true, cs_prob = calc_metrics(cs_true, cs_prob, cs_adr_names, ge_adr_names)
    cs_adr_names = ge_adr_names

    cs_true_df, ge_true_df = write_class_based_true_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'GE_META_CS', 'CS-Ontology')
    cs_probable_df, ge_probable_df = write_class_based_most_probable_predictions(cs_prob, ge_prob, cs_true, cs_adr_names, cs_drug_ids, 'GE_META_CS', 'CS-Ontology')

    write_prob_pred_truth(writer, cs_true, cs_prob, cs_adr_names, cs_drug_ids, 'GE_META_CS')
    write_prob_pred_truth(writer, ge_true, ge_prob, ge_adr_names, ge_drug_ids, 'CS-Ontology')
    cs_true_df.to_excel(writer, 'Best Results of GE_META_CS')
    ge_true_df.to_excel(writer, 'Best Results of CS-Ontology')

    cs_probable_df.to_excel(writer, 'Most Probable FP GE_META_CS')
    ge_probable_df.to_excel(writer, 'Most Probable FP CS-Ontology')
    writer.save()


def calc_metrics(true, prob, adr_names, filter_names):
    del_index = []
    for i, adr in enumerate(adr_names):
        if adr not in filter_names:
            del_index.append(i)
    y_true = np.delete(true, del_index, axis=1)
    y_prob = np.delete(prob, del_index, axis=1)
    macro_auc_score = np.mean(roc_auc_score(y_true, y_prob, average=None))
    micro_auc_score = roc_auc_score(y_true, y_prob, average="micro")
    macro_map_score = np.mean(average_precision_score(y_true, y_prob, average=None))
    micro_map_score = average_precision_score(y_true, y_prob, average="micro")
    hamming_loss_score = hammingLoss(y_prob, y_true)

    print "Macro Mean AUC:", macro_auc_score
    print "Micro Mean AUC:", micro_auc_score
    print "Macro MAP:", macro_map_score
    print "Micro MAP:", micro_map_score
    print "Hamming loss:", hamming_loss_score
    return y_true, y_prob


if __name__ == '__main__':
    cs_vs_ge_cs_meta()
    # pickle_path = "/home/onurcan/Desktop/workspace/bilkent/DeepSide/log/mlpsidenet/CS_02:15AM on April 04, 2019/predictions.pickle"
    # writer = ExcelWriter('adr_groups.xlsx')

    # cs_true, cs_prob, cs_adr_names, cs_drug_ids = load_pickle(pickle_path)

    # cs_drug_acc = get_accuracy(cs_true, cs_prob, cs_drug_ids, 'DRUG')
    # cs_side_acc = get_accuracy(cs_true, cs_prob, cs_adr_names, 'SE')

    # cs_drug_acc.to_excel(writer, 'CS Model Drug Based Metrics')
    # cs_side_acc.to_excel(writer, 'CS Model SE Based Metrics')

    # write_prob_pred_truth(writer, cs_true, cs_prob, cs_adr_names, cs_drug_ids, 'CS')
    #  writer.save()