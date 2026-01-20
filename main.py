from utils import *
from Hgraph import *
from load_dataset import *
import pickle
from module import *
import numpy as np
import random
from log import *
from train import *
import clustering_tree as ct
import nni

args, sys_argv = get_args()
DATA = args.data    
Line_big = args.linegraph_big
Line_sample = args.linegraph_big_sample
max_early_stop = args.max_early_stop
Hyper_big = args.hypergraph_big
Hyper_sample = args.hypergraph_big_sample
epochs = args.epochs
seed = args.seed
GPU = args.gpu

param = {
    'lr1': 5e-4,
    'lr2': 1e-4,
    'hypergraph_tree_height': 5,
    'linegraph_tree_height': 20,
    'node_out_channels': 128, 
    'dropout': 0.1,
    'time_channels': 64,
    'hyperEdge_hidden_channels': 128,
    'decoder_hid_channels': 128,
    'K_for_graph_compression': 3,
    'pre_epochs': 50,
}

tuner_params = nni.get_next_parameter()
param.update(tuner_params)


lr1 = args.lr1 = param['lr1']
lr2 = args.lr2 = param['lr2']
Hyper_treehigh = args.hypergraph_tree_height = param['hypergraph_tree_height']
Line_treehigh = args.linegraph_tree_height = param['linegraph_tree_height']
node_out_channels = args.node_out_channels = param['node_out_channels']
dropout = args.dropout = param['dropout']
time_channels = args.time_channels = param['time_channels']
hyperEdge_hidden_channels = args.hyperEdge_hidden_channels = param['hyperEdge_hidden_channels']
decoder_hid_channels = args.decoder_hid_channels = param['decoder_hid_channels']
K_for_graph_compression = args.K_for_graph_compression = param['K_for_graph_compression']
pre_epochs = args.pre_epochs = param['pre_epochs']

set_random_seed(seed)

### Load Data 
n_v, v_simplices, ts, dataset_name =  load_dataset(DATA)

### Generte basic hypergraph modelling (he_info) 
full_he_info = generate_he_info(n_v, ts, v_simplices)
total_node_set = set(np.unique(np.array(v_simplices)))
num_total_unique_nodes = len(total_node_set)
num_total_hyperedges = len(n_v)


# split and pack the data by generating valid train/val/test mask according to the "mode"
ts_l = np.array(ts)
val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))
if args.mode == 't':
    valid_train_he_ids = np.where(ts_l <= val_time)[0] + 1 
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time))[0] + 1
    valid_test_he_ids = np.where(ts_l > test_time)[0] + 1

else:
    assert(args.mode == 'i')
    hes_ids_after_val_time = np.where((ts_l > val_time))[0] + 1 
    he_nodes_after_val_time = set().union(*[full_he_info[i][0] for i in hes_ids_after_val_time]) 
    mask_node_set = set(random.sample(he_nodes_after_val_time, int(0.1 * num_total_unique_nodes))) 
    he_has_masked_nodes = np.array([len(full_he_info[i][0] & mask_node_set) > 0 for i in range(1, num_total_hyperedges+1)])

    valid_train_he_ids = np.where((ts_l <= val_time)  & ~(he_has_masked_nodes))[0]+1
    valid_val_he_ids = np.where((ts_l > val_time) & (ts_l <= test_time) & ~(he_has_masked_nodes))[0]+1
    valid_test_he_ids = np.where((ts_l > test_time) & (he_has_masked_nodes))[0]+1
    
    he_is_all_masked_nodes = np.array([len(full_he_info[i][0] & mask_node_set) == min(len(full_he_info[i][0]), len(mask_node_set)) for i in range(1, num_total_hyperedges+1)])
    valid_test_all_new_he_ids = np.where((ts_l > test_time) & (he_is_all_masked_nodes))[0]+1
    valid_test_new_old_he_ids = np.setdiff1d(valid_test_he_ids, valid_test_all_new_he_ids)
    
# split data according to the mask
train_data = {key: full_he_info[key] for key in valid_train_he_ids}
val_data = {key: full_he_info[key] for key in valid_val_he_ids}
test_data = {key: full_he_info[key] for key in valid_test_he_ids}
if args.mode == 'i':
    test_all_new_data = {key: full_he_info[key] for key in valid_test_all_new_he_ids}
    test_new_old_data = {key: full_he_info[key] for key in valid_test_new_old_he_ids}
train_val_data = (train_data, val_data, test_data)

# create random samplers to generate train/val/test fake instances
train_nodes = set().union(*[train_data[i][0] for i in train_data])
val_nodes = set().union(*[val_data[i][0] for i in val_data])
test_nodes = set().union(*[test_data[i][0] for i in test_data])
train_rand_sampler = RandHyperEdgeSampler([train_nodes])
val_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes])
test_rand_sampler = RandHyperEdgeSampler([train_nodes, val_nodes, test_nodes])
rand_samplers = train_rand_sampler, val_rand_sampler, test_rand_sampler

nodes_id_dict = {}
nodes_list = list(total_node_set)

for i in range(len(nodes_list)):
    nodes_id_dict[nodes_list[i]] = i

hypergraph_adj, hypergraph_incidence_adj = hypergraph_to_laplacian(train_data, train_nodes, nodes_id_dict)
nodes_position_encoding = ct.create_embedding(dataset_name, 'hypergraph', hypergraph_incidence_adj, Hyper_treehigh, Hyper_big, Hyper_sample) * 100
linegraph_adj, edge_id_list = hypergraph_to_graph(dataset_name, train_data, K_for_graph_compression)
edges_position_encoding = ct.create_embedding(dataset_name, 'linegraph', linegraph_adj, Line_treehigh, Line_big, Line_sample, K_for_graph_compression)*10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_in_channels = nodes_position_encoding.shape[1]
max_node_num = max(n_v)
hyperEdge_in_channels = edges_position_encoding.shape[1]
decoder_in_channels = hyperEdge_hidden_channels

nodes_position_encoding = torch.from_numpy(nodes_position_encoding).float().to(device)
edges_position_encoding = torch.from_numpy(edges_position_encoding).float().to(device)
linegraph_adj = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((linegraph_adj.row, linegraph_adj.col))), torch.FloatTensor(linegraph_adj.data), linegraph_adj.shape).coalesce().to(device)
hyperedge_index = torch.LongTensor(np.vstack((hypergraph_adj.row, hypergraph_adj.col))).to(device)
decoder = Decoder(decoder_in_channels, decoder_hid_channels)
linegraphSE = LinegraphSE(hyperEdge_in_channels, hyperEdge_hidden_channels, decoder)
hyperSE = HyperSE(node_in_channels, node_out_channels, max_node_num, dropout, 
                time_channels, 
                hyperEdge_hidden_channels, 
                device, decoder)

linegraphSE = linegraphSE.to(device)
hyperSE = hyperSE.to(device)

optimizer1 = torch.optim.Adam(linegraphSE.parameters(), lr=args.lr1)
optimizer2 = torch.optim.Adam(hyperSE.parameters(), lr=args.lr2)
criterion = FocalLoss(alpha=0.25, gamma=2)


test_acc, test_ap, test_f1, test_auc = train_val_test(train_val_data, linegraphSE, hyperSE, optimizer1, optimizer2, rand_samplers, args, nodes_position_encoding, hyperedge_index, edges_position_encoding, 
            linegraph_adj, nodes_id_dict, edge_id_list, max_node_num, max_early_stop)

print('test auc: {}'.format(test_acc) + ', test ap: {}'.format(test_ap) + ', test f1: {}'.format(test_f1) + ', test acc: {}'.format(test_auc))
