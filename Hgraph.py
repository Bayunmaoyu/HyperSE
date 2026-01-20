import numpy as np
import random
import torch
import math
from bisect import bisect_left
from sample import *
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import os

def hypergraph_to_laplacian(hypergraph, train_nodes, nodes_id_dict):
    hyperedges_id = list(hypergraph.keys())
    H = np.zeros(shape=(len(nodes_id_dict), len(hyperedges_id)))
    
    for i in range(len(hyperedges_id)):
        nodes = list(hypergraph[hyperedges_id[i]][0])
        for node in nodes:
            H[nodes_id_dict[node]][i] = 1
    H = sp.csr_matrix(H)
    hypergraph_adj = H.tocoo()
    D = H @ H.T
    D = sp.diags(D.diagonal())  
    L = D - H @ H.T
    D_inv_sqrt = sp.diags(1 / np.sqrt(D.diagonal()))
    L_norm = -(D_inv_sqrt @ L @ D_inv_sqrt)
    L_norm = L_norm.tocoo()
    return hypergraph_adj, L_norm


def hypergraph_to_graph(dataset, hypergraph, K_for_graph_compression):
    new_nodes = list(hypergraph.keys())  
    path = '../data/HG_Data/' + dataset + '/linegraph/K' + str(K_for_graph_compression) + '.npz'
    if os.path.exists(path):
        loaded = np.load(path)
        row = loaded['row']
        col = loaded['col']
        data = loaded['data']
        shape = tuple(loaded['shape'])
        linegraph_adj = sp.coo_matrix((data, (row, col)), shape=shape)
    else:
        row = []
        col = []
        data = []
        min_time = 100000000
        max_time = 0
        for i in range(len(new_nodes)):
            time = hypergraph[new_nodes[i]][1]
            if time < min_time:
                min_time = time
            if time > max_time:
                max_time = time
        all_time = max_time - min_time
        for i in range(len(new_nodes)):
            node_row = []
            node_col = []
            node_data = []
            for j in range(len(new_nodes)):
                edge1 = hypergraph[new_nodes[i]][0]
                edge2 = hypergraph[new_nodes[j]][0]
                time1 = hypergraph[new_nodes[j]][1]
                time2 = hypergraph[new_nodes[j]][1]
                common_nodes = set(edge1) & set(edge2)
                if common_nodes:
                    time_w = 1 - ((time2 - time1) / all_time)
                    weight = (len(common_nodes) / len(set(edge1).union(set(edge2)))) * time_w
                    node_row.append(i)
                    node_col.append(j)
                    node_data.append(weight)
            node_row = np.array(node_row)
            node_col = np.array(node_col)
            node_data = np.array(node_data)
            if len(node_col) <= K_for_graph_compression:
                row.extend(node_row)
                col.extend(node_col)
                data.extend(node_data)
            else:
                top_k_indices = np.argsort(node_data)[-K_for_graph_compression:][::-1]
                row.extend(node_row[top_k_indices])
                col.extend(node_col[top_k_indices])
                data.extend(node_data[top_k_indices])            
        linegraph_adj = sp.coo_matrix((data, (row, col)), shape=(len(new_nodes), len(new_nodes)))
        np.savez(path, row=linegraph_adj.row, col=linegraph_adj.col, data=linegraph_adj.data, shape=linegraph_adj.shape)
    return linegraph_adj, new_nodes