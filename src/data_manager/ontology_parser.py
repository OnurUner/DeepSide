import xml.etree.ElementTree
import pandas as pd
import numpy as np
import os

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, "..", "..", "data")
ontology_path = os.path.join(data_path, "ADReCS_ADR_info.xml")
label_path = os.path.join(data_path, "SIDER_PTs.csv")


def get_adr_ids():
    e = xml.etree.ElementTree.parse(ontology_path).getroot()
    label_df = pd.read_csv(label_path, index_col=0)
    adr_names = label_df.columns.values

    adr_ids = {}

    for adr_name in adr_names:
        for adr in e._children:
            adr_id = None
            for adr_child in adr._children:
                if adr_child.tag == "ADReCS_ID":
                    adr_id = adr_child.text

                if adr_child.tag == "ADR_TERM":
                    if adr_name.lower().replace(',', '') == adr_child.text.lower().replace(',', ''):
                        adr_name_ = adr_name.lower().replace(',', '')
                        if adr_name_ not in adr_ids:
                            adr_ids[adr_name_] = []
                        adr_ids[adr_name_].append(adr_id)
                        break
    return adr_ids


def get_ontology_vectors(adr_queries):
    adr_ids = get_adr_ids()
    max_nodes = [-1] * 4

    for id_list in adr_ids.values():
        for id in id_list:
            id_tree = id.split(".")
            for i, node_id in enumerate(id_tree):
                if max_nodes[i] < int(node_id):
                    max_nodes[i] = int(node_id)

    exist_names = []
    ontology_vectors = []
    drop_adr_names = []
    for adr_query in adr_queries:
        query = adr_query.lower().replace(',', '')
        if query not in adr_ids:
            print(adr_query)
            drop_adr_names.append(adr_query)
            continue

        tree_level_vectors = []
        for max_node in max_nodes:
            tree_level_vectors.append(np.zeros(max_node))

        for adr_index, adr_id in enumerate(adr_ids[query]):
            id_tree = adr_id.split(".")
            for node_i, id in enumerate(id_tree):
                i = int(id) - 1
                #				if node_i == 0:
                #					tree_level_vectors[node_i][i] = 1
                #				elif node_i == 1:
                #					tree_level_vectors[node_i][i] = 10
                #				elif node_i == 2:
                #					tree_level_vectors[node_i][i] = 100
                #				elif node_i == 3:
                #					tree_level_vectors[node_i][i] = 1000
                #				tree_level_vectors[node_i][i] = node_i + 1
                tree_level_vectors[node_i][i] += 1

        exist_names.append(adr_query)
        v = np.hstack(tree_level_vectors)
        ontology_vectors.append(v.reshape(-1, 1))
    return ontology_vectors, drop_adr_names, exist_names, adr_ids


def get_labels(prune_count=0, association_thr=1):
    label_df = pd.read_csv(label_path, index_col=0)
    adr_names = list(label_df)
    del_adr_names = list()
    for adr_name in adr_names:
        total = np.sum(label_df.loc[:, adr_name].values)
        if total < prune_count or (float(total) / label_df.loc[:, adr_name].shape[0]) > association_thr:
            del_adr_names.append(adr_name)
    return label_df.drop(del_adr_names, axis=1)


def get_adr_groups(label_df):
    e = xml.etree.ElementTree.parse(ontology_path).getroot()
    adr_names = label_df.columns.values
    adr_groups = {}

    adr_ids = {}
    id_names = {}
    for adr_name in adr_names:
        for adr in e._children:
            adr_id = None
            for adr_child in adr._children:
                if adr_child.tag == "ADReCS_ID":
                    adr_id = adr_child.text
                    id_names[adr_id] = None

                if adr_child.tag == "ADR_TERM":
                    id_names[adr_id] = adr_child.text
                    if adr_name.lower().replace(',', '') == adr_child.text.lower().replace(',', ''):
                        adr_name_ = adr_name.lower().replace(',', '')
                        if adr_name_ not in adr_ids:
                            adr_ids[adr_name_] = []
                        adr_ids[adr_name_].append(adr_id)
                        break

    for adr_name in adr_ids:
        for adr_id in adr_ids[adr_name]:
            parent = adr_id[:5]
            if parent not in adr_groups:
                adr_groups[parent] = []
            adr_groups[parent].append(adr_name)
    return adr_groups, id_names


if __name__ == "__main__":
    e = xml.etree.ElementTree.parse(ontology_path).getroot()
    label_df = get_labels(prune_count=11)
    adr_names = label_df.columns.values

    adr_ids = {}
    id_names = {}
    for adr_name in adr_names:
        for adr in e._children:
            adr_id = None
            for adr_child in adr._children:
                if adr_child.tag == "ADReCS_ID":
                    adr_id = adr_child.text
                    id_names[adr_id] = None

                if adr_child.tag == "ADR_TERM":
                    id_names[adr_id] = adr_child.text
                    if adr_name.lower().replace(',', '') == adr_child.text.lower().replace(',', ''):
                        adr_name_ = adr_name.lower().replace(',', '')
                        if adr_name_ not in adr_ids:
                            adr_ids[adr_name_] = []
                        adr_ids[adr_name_].append(adr_id)
                        break

    header = ["ADR NAME", "1.BRANCH", "2.BRANCH", "3.BRANCH", "LEAF"]
    csv_rows = []
    for adr_name in adr_ids:
        for adr_id in adr_ids[adr_name]:
            row = []
            row.append(adr_name)
            row.append(id_names[adr_id[:2]])
            row.append(id_names[adr_id[:5]])
            row.append(id_names[adr_id[:8]])
            row.append(id_names[adr_id])
            csv_rows.append(row)






