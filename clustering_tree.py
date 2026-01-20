import torch
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import codingTree_utils as cu
import os

def get_node_partition_index(hierarchical_partition):
    node_partition_index = []
    temp = None
    for k, partition in enumerate(hierarchical_partition):
        if k == 0:
            temp = partition
        else:
            temp = torch.sparse.mm(temp, partition)
        node_partition_index.append(temp._indices()[1]) 

    return node_partition_index

def create_embedding(dataset, tree_path, adj_train_coo, tree_height, big, big_sample, K_for_graph_compression=None):
    if K_for_graph_compression == None:
        path = '../data/HG_Data/' + dataset + '/encoding/' + tree_path + '/' + str(tree_height) + '.npy'
    else:   
        path = '../data/HG_Data/' + dataset + '/encoding/' + tree_path + '/H' + str(tree_height) + '_K' + str(K_for_graph_compression) + '.npy'
    if os.path.exists(path):
        entro_encoding = np.load(path)
    else:
        encoding_tree = cu.build_k_coding_tree(dataset, tree_path, adj_train_coo, tree_height, K_for_graph_compression)   
        nodes = encoding_tree.tree_node.values()
        node_num = adj_train_coo.shape[0]
        if big:
            a=0
        else:
            gnn_high = -np.ones(shape=adj_train_coo.shape, dtype=int)
            for node in nodes:
                if node.high == 1:
                    continue
                else:
                    part = node.partition
                    for i in part:
                        for j in part:
                            if i == j or gnn_high[i][j] != -1:
                                a = 0
                            else:
                                gnn_high[i][j] = node.high-2
            gnn_high[gnn_high == -1] = tree_height-1

        if big:
            use_tree_high = []
            basic_high = int(tree_height / big_sample)
            if big_sample > 1:
                for i in range(big_sample-1):
                    use_tree_high.append(basic_high*(i+1))
            use_tree_high.append(tree_height-1)
            nodes = encoding_tree.tree_node.values()
            entro_encoding = []
            for high in use_tree_high:
                for node in nodes:
                    if node.high == (high-1):
                        part = node.partition   
                        entro = node.entropy-0.00001
                        if entro == 0:
                            continue
                        else:
                            entro_array = np.full(node_num, entro)
                            entro_array[part] = 0
                            entro_encoding.append(entro_array)
                    else:
                        continue
            entro_encoding = np.array(entro_encoding).T
            transform =VarianceThreshold()
            entro_encoding =transform.fit_transform(entro_encoding)
        else:
            entro_encoding = np.zeros(shape=(node_num, node_num))
            nodes = encoding_tree.tree_node.values()
            for node in nodes:
                part = node.partition
                entro = node.entropy
                entro_array = np.full(node_num, entro)
                entro_array[part] = 0
                for id in part:
                    entro_encoding[id] = entro_encoding[id] + entro_array
            entro_encoding = np.array(entro_encoding, dtype=np.double).T
        np.save(path, entro_encoding)
    return entro_encoding


