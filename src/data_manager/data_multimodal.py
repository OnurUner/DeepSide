import numpy as np
import pandas as pd
from .preprocess import scale

from .drugdataset import DrugDataset
from .utils import get_meta, load_cs, generate_folds
from .utils import load_l5, prune_labels, dict_to_df, load_meta, load_signatures


def get_ge_cs_meta(n_fold=3, train_signatures=False, test_signatures=True, **kwargs):
    dataset = load_l5()
    cs = load_cs()
    meta = load_meta()
    strongest_sig = load_signatures()
    ge = dataset["drug_expression"]
    exp = dataset["drug_perturbations"]
    labels = dataset["label_df"]
    labels = prune_labels(labels, 11, 1)

    drug_ids = set()
    for cs_id in cs.index.values:
        if cs_id in labels.index.values:
            drug_ids.add(cs_id)

    drug_ids = list(drug_ids)
    cs = cs.loc[drug_ids]
    cs = pd.DataFrame((cs.values).astype(np.float32), columns=cs.columns, index=cs.index)

    train_fold, test_fold, labels = generate_folds(drug_ids, labels, n_fold)

    for fold_i in range(n_fold):
        train_dataset = []
        test_dataset = []
        train_ids = train_fold[fold_i]
        test_ids = test_fold[fold_i]

        train_ge_data, train_exps, train_drugs = dict_to_df(ge, exp, train_ids)
        test_ge_data, test_exps, test_drugs = dict_to_df(ge, exp, test_ids)
        exp_ids = train_exps + test_exps
        meta_data = meta.loc[exp_ids]
        meta_data = get_meta(meta_data)
        meta_data = (meta_data).astype(np.float32)

        train_ge_data = scale(train_ge_data, just_transform=False)
        test_ge_data = scale(test_ge_data, just_transform=True)

        train_drug_ids = []
        test_drug_ids = []

        for i in range(train_ge_data.shape[0]):
            x_ge = train_ge_data[i, :]
            x_cs = cs.loc[train_drugs[i]].values
            x_meta = meta_data[exp_ids.index(train_exps[i]), :]

            if train_signatures:
                strong_exp = strongest_sig.loc[train_drugs[i], "strongest sig_id"]
                if train_exps[i] != strong_exp:
                    continue

            x = (x_ge, x_meta, x_cs)
            y = labels.loc[train_drugs[i]].values.astype(np.float32)
            train_dataset.append((x, y))
            train_drug_ids.append(train_drugs[i])

        for i in range(test_ge_data.shape[0]):
            x_ge = test_ge_data[i, :]
            x_cs = cs.loc[test_drugs[i]].values
            x_meta = meta_data[exp_ids.index(test_exps[i]), :]

            if test_signatures:
                strong_exp = strongest_sig.loc[test_drugs[i], "strongest sig_id"]
                if test_exps[i] != strong_exp:
                    continue

            x = (x_ge, x_meta, x_cs)
            y = labels.loc[test_drugs[i]].values.astype(np.float32)
            test_dataset.append((x, y))
            test_drug_ids.append(test_drugs[i])

        train = DrugDataset(train_dataset, labels.shape[1], train_drug_ids, train_exps, list(labels.columns.values))
        test = DrugDataset(test_dataset, labels.shape[1], test_drug_ids, test_exps, list(labels.columns.values))
        yield train, test