import os
import numpy as np
import pandas as pd
from .preprocess import scale, normalize

from .drugdataset import DrugDataset
from .utils import get_meta, load_cs, load_labels, generate_folds
from .utils import load_l5, prune_labels, dict_to_df, load_meta, load_signatures
from .utils import get_ontology_vectors, get_adr_groups, transform_label
from .utils import go_path, cd_lincs_path

current_path = os.path.dirname(os.path.abspath(__file__))
ontology_filepath = os.path.join(current_path, "ontology_vectors.pickle")


class DataType:
    GE, CS, GO, CD, GE_CS, GE_CS_META, GE_META = range(7)


datatype_factory = {
    'CS': DataType.CS,
    'GE': DataType.GE,
    'GO': DataType.GO,
    'CD': DataType.CD,
    'GE_CS': DataType.GE_CS,
    'GE_CS_META': DataType.GE_CS_META,
    'GE_META': DataType.GE_META
}


def get_cs(n_fold=3, parent_labels=True, **kwargs):
    dataset = load_l5()
    cs = load_cs()
    labels = dataset["label_df"]
    labels = prune_labels(labels, 11, 0.8)

    drug_ids = set()
    for cs_id in cs.index.values:
        if cs_id in labels.index.values:
            drug_ids.add(cs_id)

    drug_ids = list(drug_ids)
    cs = cs.loc[drug_ids]

    cs = pd.DataFrame((cs.values).astype(np.float32), columns=cs.columns, index=cs.index)

    train_fold, test_fold, labels = generate_folds(drug_ids, labels, n_fold)
    ontology_vectors, drop_adrs, exist_names, adr_ids = get_ontology_vectors(ontology_filepath, list(labels.columns.values))
    # labels = labels.drop(drop_adrs, axis=1)
    column_names = list(labels.columns.values)
    if parent_labels:
        se_ids, all_ids, column_names = get_adr_groups(labels)

    for fold_i in range(n_fold):
        train_dataset = []
        test_dataset = []
        train_ids = train_fold[fold_i]
        test_ids = test_fold[fold_i]

        for _id in train_ids:
            x = cs.loc[_id].values
            y = labels.loc[_id].values.astype(np.float32)
            if parent_labels:
                y = transform_label(all_ids, se_ids, y, list(labels.columns.values))
            train_dataset.append((x, y))

        for _id in test_ids:
            x = cs.loc[_id].values
            y = labels.loc[_id].values.astype(np.float32)
            if parent_labels:
                y = transform_label(all_ids, se_ids, y, list(labels.columns.values))
            test_dataset.append((x, y))

        if parent_labels:
            output_size = y.shape[0]
            # adr_names =
        else:
            output_size = labels.shape[1]
        train = DrugDataset(train_dataset, output_size, train_ids, None, column_names, ontology_vectors=ontology_vectors)
        test = DrugDataset(test_dataset, output_size, test_ids, None, column_names, ontology_vectors=ontology_vectors)
        yield train, test


def get_processed_data(n_fold=3, data_type="GO", multimodal=False, **kwargs):
    data_type = datatype_factory[data_type]
    labels = load_l5()["label_df"]
    labels = prune_labels(labels, 11, 1)
    cs = load_cs()
    dataset = None
    if data_type == DataType.GO:
        dataset = pd.read_csv(go_path, index_col=0)
    elif data_type == DataType.CD:
        dataset = pd.read_csv(cd_lincs_path, index_col=0)

    drug_ids = set()
    for cs_id in cs.index.values:
        if cs_id in labels.index.values and cs_id in dataset.index.values:
            drug_ids.add(cs_id)

    drug_ids = list(drug_ids)
    train_fold, test_fold, labels = generate_folds(drug_ids, labels, n_fold)

    for fold_i in range(n_fold):
        train_dataset = []
        test_dataset = []
        train_ids = train_fold[fold_i]
        test_ids = test_fold[fold_i]
        train_df = dataset.loc[train_ids]
        test_df = dataset.loc[test_ids]

        train_df = scale(train_df, just_transform=False)
        test_df = scale(test_df, just_transform=True)

        train_df = pd.DataFrame(train_df, index=train_ids)
        test_df = pd.DataFrame(test_df, index=test_ids)

        for id in train_ids:
            x = train_df.loc[id].values.astype(np.float32)
            y = labels.loc[id].values.astype(np.float32)
            train_dataset.append((x, y))

        for id in test_ids:
            x = test_df.loc[id].values.astype(np.float32)
            y = labels.loc[id].values.astype(np.float32)
            test_dataset.append((x, y))

        train = DrugDataset(train_dataset, labels.shape[1], train_ids, None, list(labels.columns.values))
        test = DrugDataset(test_dataset, labels.shape[1], test_ids, None, list(labels.columns.values))
        yield train, test


def get_ge(n_fold=3, data_type="GE_META", multimodal=False, train_signatures=False, test_signatures=False, **kwargs):
    data_type = datatype_factory[data_type]
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
        print("GE Scaled")
        train_ge_data = scale(train_ge_data, just_transform=False)
        test_ge_data = scale(test_ge_data, just_transform=True)

        train_drug_ids = []
        test_drug_ids = []

        for i in range(train_ge_data.shape[0]):
            x_ge = train_ge_data[i, :]
            x_meta = meta_data[exp_ids.index(train_exps[i]), :]

            if train_signatures:
                strong_exp = strongest_sig.loc[train_drugs[i], "strongest sig_id"]
                if train_exps[i] != strong_exp:
                    continue
            if multimodal:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.GE_META:
                    x = (x_ge, x_meta)
            else:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.GE_META:
                    x = np.hstack([x_ge, x_meta]).astype(np.float32)
            y = labels.loc[train_drugs[i]].values.astype(np.float32)
            train_dataset.append((x, y))
            train_drug_ids.append(train_drugs[i])

        for i in range(test_ge_data.shape[0]):
            x_ge = test_ge_data[i, :]
            x_meta = meta_data[exp_ids.index(test_exps[i]), :]

            if test_signatures:
                strong_exp = strongest_sig.loc[test_drugs[i], "strongest sig_id"]
                if test_exps[i] != strong_exp:
                    continue
            if multimodal:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.GE_META:
                    x = (x_ge, x_meta)
            else:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.GE_META:
                    x = np.hstack([x_ge, x_meta]).astype(np.float32)
            y = labels.loc[test_drugs[i]].values.astype(np.float32)
            test_dataset.append((x, y))
            test_drug_ids.append(test_drugs[i])

        train = DrugDataset(train_dataset, labels.shape[1], train_drug_ids, train_exps, list(labels.columns.values))
        test = DrugDataset(test_dataset, labels.shape[1], test_drug_ids, test_exps, list(labels.columns.values))
        yield train, test


def get_dataset(n_fold=3, data_type="GE_CS", multimodal=False, 
                train_signatures=False, test_signatures=True,
                add_dmso=False, dmso_norm=False, concat_dmso=False, **kwargs):
    data_type = datatype_factory[data_type]
    dataset = load_l5(add_dmso, dmso_norm, concat_dmso)
    cs = load_cs(add_dmso)
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

        if add_dmso:
            if "DMSO" in test_ids:
                test_ids.remove("DMSO")
                print("DMSO removed from test ids")
            if "DMSO" not in train_ids:
                train_ids.append("DMSO")
                print("DMSO appended to train ids")
            assert "DMSO" in train_ids, "DMSO not exists"
            if "DMSO" in train_ids:
                print("Train set with DMSO")

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
                if train_drugs[i] == 'DMSO':
                    continue
                strong_exp = strongest_sig.loc[train_drugs[i], "strongest sig_id"]
                if train_exps[i] != strong_exp:
                    continue

            if not multimodal:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.CS:
                    x = x_cs
                elif data_type == DataType.GE_CS:
                    x = np.hstack([x_ge, x_cs]).astype(np.float32)
                elif data_type == DataType.GE_CS_META:
                    x = np.hstack([x_ge, x_cs, x_meta]).astype(np.float32)
                else:
                    print("MLP DATA ERROR")
            else:
                if data_type == DataType.GE_CS_META:
                    x = (np.hstack([x_ge, x_meta]).astype(np.float32), x_cs)
                elif data_type == DataType.GE_CS:
                    x = (x_ge, x_cs)
                else:
                    print("MULTIMODAL DATA ERROR")

            y = labels.loc[train_drugs[i]].values.astype(np.float32)
            train_dataset.append((x, y))
            train_drug_ids.append(train_drugs[i])

        for i in range(test_ge_data.shape[0]):
            x_ge = test_ge_data[i, :]
            x_cs = cs.loc[test_drugs[i]].values
            x_meta = meta_data[exp_ids.index(test_exps[i]), :]

            if test_signatures:
                if test_drugs[i] == 'DMSO':
                    continue
                strong_exp = strongest_sig.loc[test_drugs[i], "strongest sig_id"]
                if test_exps[i] != strong_exp:
                    continue

            if not multimodal:
                if data_type == DataType.GE:
                    x = x_ge
                elif data_type == DataType.CS:
                    x = x_cs
                elif data_type == DataType.GE_CS:
                    x = np.hstack([x_ge, x_cs]).astype(np.float32)
                elif data_type == DataType.GE_CS_META:
                    x = np.hstack([x_ge, x_cs, x_meta]).astype(np.float32)
                else:
                    print("MLP DATA ERROR")
            else:
                if data_type == DataType.GE_CS_META:
                    x = (np.hstack([x_ge, x_meta]).astype(np.float32), x_cs)
                elif data_type == DataType.GE_CS:
                    x = (x_ge, x_cs)
                else:
                    print("MULTIMODAL DATA ERROR")

            y = labels.loc[test_drugs[i]].values.astype(np.float32)
            test_dataset.append((x, y))
            test_drug_ids.append(test_drugs[i])

        train = DrugDataset(train_dataset, labels.shape[1], train_drug_ids, train_exps, list(labels.columns.values))
        test = DrugDataset(test_dataset, labels.shape[1], test_drug_ids, test_exps, list(labels.columns.values))
        yield train, test


