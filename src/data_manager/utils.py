import os
import cv2
import os.path as osp
import pickle
import numpy as np
import pandas as pd
from math import ceil
from sklearn.preprocessing import normalize as norml2

from src.data_manager import ontology_parser

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(cur_dir, "../../data/level5_dataset.pkl")
concat_dmso_dataset_path = os.path.join(cur_dir, "../../data/level5_dmso_concat_dataset.pkl")
add_dmso_dataset_path = os.path.join(cur_dir, "../../data/level5_with_dmso_dataset.pkl")
dmso_norm_dataset_path = os.path.join(cur_dir, "../../data/level5_dmso_normalized_dataset.pkl")
mc_path = os.path.join(cur_dir, "../../data/MLPCN_morplological_profiles.csv")
cs_path = os.path.join(cur_dir, "../../data/MACCS_bitmatrix.csv")
label_path = os.path.join(cur_dir, "../../data/SIDER_PTs.csv")
meta_path = os.path.join(cur_dir, "../../data/GSE92742_Broad_LINCS_sig_info.txt")
strongest_sig_path = os.path.join(cur_dir, "../../data/meta_LINCS_strongest_signatures.csv")
cd_lincs_path = os.path.join(cur_dir, "../../data/LINCS_Gene_Experssion_signatures_CD.csv")
go_path = os.path.join(cur_dir, "../../data/GO_transformed_signatures_PAEA.csv")
cs_images_folder = os.path.join(cur_dir, "../../data/cs_2d")
meta_smiles_path = os.path.join(cur_dir, "../../data/meta_SMILES.csv")


def load_pickle(path):
	dataset = None
	with open(path, 'rb') as handle:
		dataset = pickle.load(handle, encoding="latin1")
	return dataset


def load_labels(prune_count=0, association_thr=1, add_dmso=False):
	if add_dmso:
		dataset = load_pickle(add_dmso_dataset_path)
	else:
		dataset = load_pickle(dataset_path)
	label_df = dataset["label_df"]
	prune_labels(label_df, prune_count, association_thr)
	return label_df


def prune_labels(label_df, prune_count, association_thr):
	adr_names = list(label_df)
	del_adr_names = list()
	for adr_name in adr_names:
		total = np.sum(label_df.loc[:,adr_name].values)
		if total < prune_count or (float(total) / label_df.loc[:,adr_name].shape[0]) > association_thr:
			del_adr_names.append(adr_name)		
	return label_df.drop(del_adr_names, axis=1)


def load_l5(add_dmso=False, dmso_norm=False, concat_dmso=False):
	if add_dmso:
		return load_pickle(add_dmso_dataset_path)
	elif dmso_norm:	
		return load_pickle(dmso_norm_dataset_path)
	elif concat_dmso:
		return load_pickle(concat_dmso_dataset_path)
	else:
		return load_pickle(dataset_path)


def load_mc():
	return pd.read_csv(mc_path, index_col=0)


def load_cs(add_dmso=False):
	cs_df = pd.read_csv(cs_path, index_col=0)
	if add_dmso:
		cs_df.loc["DMSO"] = np.ones((cs_df.shape[1]))
	return cs_df


def load_meta():
	return pd.read_csv(meta_path, index_col=0, sep='\t')


def load_signatures():
	return pd.read_csv(strongest_sig_path, index_col=0)


def generate_folds(drug_list, label_df, n_fold):
	del_label_cols = set()
	fold_len = int(ceil(len(drug_list)/float(n_fold)))
	test_folds = []
	train_folds = []
	begin_index = 0
	end_index = begin_index + fold_len
	for i in range(n_fold):
		train_ids = drug_list[0:begin_index] + drug_list[end_index:]
		test_ids = drug_list[begin_index:end_index]
		
		train_folds.append(train_ids)
		test_folds.append(test_ids)
		begin_index += fold_len
		end_index += fold_len
		
		train_label_df = label_df.loc[train_ids,:]
		val_label_df = label_df.loc[test_ids,:]
				
		for column in train_label_df.columns:
			if np.sum(train_label_df.loc[:, column].values) == 0 or np.sum(val_label_df.loc[:, column].values) == 0:
				del_label_cols.add(column)

	pruned_labels = label_df.drop(del_label_cols, axis=1)
	print("Delete number of label columns:", len(del_label_cols))
	return train_folds, test_folds, pruned_labels


def dict_to_df(data_dict, exp_dict, drug_ids):
	data_list = []
	exp_list = []
	drug_id_list = []
	
	for d_id in drug_ids:
		for ex_i, ge_data in enumerate(data_dict[d_id]):
			data_list.append(ge_data)
			exp_list.append(exp_dict[d_id][ex_i])
			drug_id_list.append(d_id)
	return np.vstack(data_list), exp_list, drug_id_list


def onehot(values, val):
	data = np.zeros((1, len(values)))
	index = values.index(val)
	data[0,index] = 1.0
	return data


def get_meta(df):
	cells = sorted(np.unique(df.loc[:,"cell_id"].values).tolist())
	doses = sorted(np.unique(df.loc[:,"pert_idose"].values).tolist())
	times = sorted(np.unique(df.loc[:,"pert_itime"].values).tolist())
	data = []
	for i in range(df.shape[0]):
		cell = df.loc[df.index[i],"cell_id"]
		dose = df.loc[df.index[i],"pert_idose"]
		time = df.loc[df.index[i],"pert_itime"]
		
		np_cell = onehot(cells, cell)
		np_dose = onehot(doses, dose)
		np_time = onehot(times, time)
		data.append(np.hstack([np_cell, np_dose, np_time]))
		#data.append(np.hstack([np_cell]))
	return np.vstack(data)
		

def get_ontology_vectors(ontology_filepath, adr_names):
	if os.path.isfile(ontology_filepath):
		with open(ontology_filepath, 'rb') as handle:
			f_content = pickle.load(handle)
			ontology_vectors = f_content["ontology_vectors"]
			drop_list = f_content["drop_list"]
			exist_names = f_content["exist_names"]
			adr_ids = f_content["adr_ids"]
	else:
		ontology_vectors, drop_list, exist_names, adr_ids = ontology_parser.get_ontology_vectors(adr_names)
		ontology_vectors = np.hstack(ontology_vectors)
		f_content = dict()
		f_content["ontology_vectors"] = ontology_vectors
		f_content["drop_list"] = drop_list
		f_content["exist_names"] = exist_names
		f_content["adr_ids"] = adr_ids
		with open(ontology_filepath, 'wb') as handle:
			pickle.dump(f_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# ontology_vectors = ontology_vectors.astype(np.float32)
	# ontology_vectors[ontology_vectors == 0] = 0.1111111
	# ontology_vectors[ontology_vectors == 1] = 0.8888889
	ontology_vectors = norml2(ontology_vectors.astype(np.float32), axis=0)
	return ontology_vectors, drop_list, exist_names, adr_ids


def get_adr_groups(labels_df):
	adr_groups, id_names = ontology_parser.get_adr_groups(labels_df)
	se_ids = {}
	for adr_id in adr_groups:
		for se_name in adr_groups[adr_id]:
			if se_name not in se_ids:
				se_ids[se_name] = []
			se_ids[se_name].append(adr_id)

	all_ids = adr_groups.keys()
	all_ids.sort()

	id2name = {}
	group_names = []
	for adr_name in se_ids:
		for se_id in se_ids[adr_name]:
			id2name[se_id] = adr_name
	for adr_id in all_ids:
		group_names.append(id2name[adr_id])

	return se_ids, all_ids, group_names


def transform_label(all_ids, se_ids, value, adr_names):
	y = np.zeros(len(all_ids)).astype(np.float32)
	for i in range(value.shape[0]):
		if value[i] == 1:
			for se_id in se_ids[adr_names[i].lower().replace(',', '')]:
				index = all_ids.index(se_id)
				y[index] = 1
	return y


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = cv2.imread(
                img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img