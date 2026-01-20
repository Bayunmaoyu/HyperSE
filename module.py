import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import dhg
from clustering_tree import create_embedding
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge, GATConv, APPNP, LightGCN, EGConv, ClusterGCNConv, HypergraphConv




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinegraphSE(nn.Module):
    def __init__(self,  
                hyperEdge_in_channels, hyperEdge_hidden_channels, 
                decoder
                 ):
        super(LinegraphSE, self).__init__()
        self.logger = logging.getLogger(__name__)      
        self.hyperEdge_in_channels = hyperEdge_in_channels
        self.hyperEdge_hidden_channels = hyperEdge_hidden_channels

        self.hyperedge_encoder = HyperEdgeEncode(hyperEdge_in_channels, hyperEdge_hidden_channels)
        self.decoder = decoder

    def forward(self, edges_entropy_encoding, linegraph_adj, pos_data, edge_id_list):
        neg_edges_entropy_encoding = F.dropout(edges_entropy_encoding + torch.randn_like(edges_entropy_encoding) * 0.1, p=0.05)
        pos_edge_encoding = self.hyperedge_encoder(edges_entropy_encoding, linegraph_adj)
        neg_edge_encoding = self.hyperedge_encoder(neg_edges_entropy_encoding, linegraph_adj)
        pos_linegraph_edge_encoding = get_linegraph_edge_encoding(pos_data, pos_edge_encoding, edge_id_list)
        neg_linegraph_edge_encoding = get_linegraph_edge_encoding(pos_data, neg_edge_encoding, edge_id_list)
        pos_edge_score = self.decoder(pos_linegraph_edge_encoding)
        neg_edge_score = self.decoder(neg_linegraph_edge_encoding)
        print('pos_edge_score:'+str(pos_edge_score))
        print('neg_edge_score:'+str(neg_edge_score))
        return pos_edge_score, neg_edge_score
    
    def getencoding(self, edges_entropy_encoding, linegraph_adj, pos_data, edge_id_list):
        edge_encoding = self.hyperedge_encoder(edges_entropy_encoding, linegraph_adj)
        pos_linegraph_edge_encoding = get_linegraph_edge_encoding(pos_data, edge_encoding, edge_id_list)

        return pos_linegraph_edge_encoding

class HyperSE(nn.Module):
    def __init__(self,  
                node_in_channels, node_out_channels, max_node_num, dropout, 
                time_channels, 
                hyperEdge_hidden_channels, 
                device, decoder
                 ):
        super(HyperSE, self).__init__()
        self.logger = logging.getLogger(__name__)      

        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.max_node_num = max_node_num
        self.time_channels = time_channels
        self.hyperEdge_hidden_channels = hyperEdge_hidden_channels
        self.device = device
        self.node_encoder = NodeEncode(node_in_channels, node_out_channels, dropout)
        self.nodeset_encoder = NodeSetEncode(node_out_channels, max_node_num, time_channels, hyperEdge_hidden_channels, device)
        self.decoder = decoder
        self.triplet_loss = TripletLoss(margin=0.5)

    def forward(self, nodes_entropy_encoding, hypergraph_adj, pos_data, neg_data, nodes_id_dict, max_node_num, pos_linegraph_edge_encoding):
        nodes_encoding = self.node_encoder(nodes_entropy_encoding, hypergraph_adj)
        pos_nodes_set_encoding = get_nodes_set_encoding(pos_data, nodes_encoding, nodes_id_dict, max_node_num, self.device)
        pos_time = get_time(pos_data)
        pos_nodes_set_encoding = self.nodeset_encoder(pos_nodes_set_encoding, pos_time)
        neg_nodes_set_encoding = get_nodes_set_encoding(neg_data, nodes_encoding, nodes_id_dict, max_node_num, self.device)
        neg_time = get_time(neg_data)
        neg_nodes_set_encoding = self.nodeset_encoder(neg_nodes_set_encoding, neg_time)
        pos_score = self.decoder(pos_nodes_set_encoding)
        neg_score = self.decoder(neg_nodes_set_encoding)
        print('pos_score:'+str(torch.sum(pos_score)))
        print('neg_score:'+str(torch.sum(neg_score)))
        tri_loss = self.triplet_loss(pos_linegraph_edge_encoding, pos_nodes_set_encoding, neg_nodes_set_encoding)
        print('tri_loss:'+str(tri_loss))
        return pos_score, neg_score, tri_loss
    
    def score_for_test(self, nodes_entropy_encoding, hyperedge_index, pos_data, neg_data, nodes_id_dict, max_node_num):
        nodes_encoding = self.node_encoder(nodes_entropy_encoding, hyperedge_index)
        pos_nodes_set_encoding = get_nodes_set_encoding(pos_data, nodes_encoding, nodes_id_dict, max_node_num, self.device)
        pos_time = get_time(pos_data)
        pos_nodes_set_encoding = self.nodeset_encoder(pos_nodes_set_encoding, pos_time)
        neg_nodes_set_encoding = get_nodes_set_encoding(neg_data, nodes_encoding, nodes_id_dict, max_node_num, self.device)
        neg_time = get_time(neg_data)
        neg_nodes_set_encoding = self.nodeset_encoder(neg_nodes_set_encoding, neg_time)
        pos_score = self.decoder(pos_nodes_set_encoding)
        neg_score = self.decoder(neg_nodes_set_encoding)

        return pos_score, neg_score

def get_time(edge_data):
    ts = []
    edges = list(edge_data.keys())
    for edge in edges:
        ts.append(edge_data[edge][1])
    ts = np.array(ts)
    return ts

def get_linegraph_edge_encoding(edge_data, edge_encoding, edge_id_list):
    edges = list(edge_data.keys())
    index = []
    for edge in edges:
        index.append(edge_id_list.index(edge))
    pos_linegraph_edge_encoding = edge_encoding[index]

    return pos_linegraph_edge_encoding

def get_nodes_set_encoding(edges, nodes_encoding, nodes_id_dict, max_node_num, device):
    node_encoding_shape = nodes_encoding.shape[1]
    empty_node_encoding = torch.tensor(np.zeros(node_encoding_shape)).float().to(device)
    nodes_set_encodings = torch.empty(0, node_encoding_shape).to(device)
    for edge in edges.values():
        nodes = list(edge[0])
        nodes_num = len(nodes)
        index = []
        for node in nodes:
            index.append(nodes_id_dict[node])
        nodes_set_encoding = nodes_encoding[index]
        nodes_set_encoding = torch.sum(nodes_set_encoding, dim=0)
        nodes_set_encoding = torch.unsqueeze(nodes_set_encoding, 0)
        nodes_set_encodings = torch.cat([nodes_set_encodings, nodes_set_encoding], dim=0)
    return nodes_set_encodings

def dis_line_ori(nodes_set_encoding, linegraph_edge_encoding):
    similarity = (F.cosine_similarity(nodes_set_encoding, linegraph_edge_encoding, dim=1).detach() + 1) / 2
    return similarity

class NodeEncode(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, dropout):
        super().__init__()
        self.feat_encoder = HypergraphConv(node_in_channels, node_out_channels, dropout=dropout) 
        self.reset_parameters()

    def reset_parameters(self):
        self.feat_encoder.reset_parameters()
        
    def forward(self, nodes_entropy_encoding, hyperedge_index):
        nodes_encoding = self.feat_encoder(nodes_entropy_encoding, hyperedge_index)
        return nodes_encoding
  

class NodeSetEncode(nn.Module):
    def __init__(self, node_out_channels, max_node_num, time_dims, hyperEdge_hidden_channels, device):
        super().__init__()
        self.device = device
        self.time_encoder = TimeEncode(time_dims, device) 
        self.feat_encoder = nn.Linear(time_dims + node_out_channels, hyperEdge_hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, nodes_set_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([nodes_set_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class HyperEdgeEncode(nn.Module):
    def __init__(self, hyperEdge_in_channels, hyperEdge_hidden_channels):
        super().__init__()
        self.gcn1 = GCNConv(hyperEdge_in_channels, hyperEdge_in_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.gcn2 = GCNConv(hyperEdge_in_channels, hyperEdge_hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        
    def forward(self, edges_entropy_encoding, linegraph):
        edge_encoding = self.gcn1(edges_entropy_encoding, linegraph)
        edge_encoding = self.relu(edge_encoding)
        edge_encoding = self.drop(edge_encoding)
        edge_encoding = self.gcn2(edge_encoding, linegraph)
        return edge_encoding

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, device, factor=5):
        super(TimeEncode, self).__init__()
        self.device = device
        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        self.linear = torch.nn.Parameter(torch.zeros(1).float())
        self.linear_bias = torch.nn.Parameter(torch.zeros(1).float())

    def reset_parameters(self):
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        self.linear = torch.nn.Parameter(torch.zeros(1).float())
        self.linear_bias = torch.nn.Parameter(torch.zeros(1).float())

    def forward(self, ts):
        ts = ts[np.newaxis, :]
        ts = torch.tensor(ts).float().to(self.device)
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  
        map_ts = ts * self.basis_freq.view(1, 1, -1) 
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts) + (self.linear * ts) + self.linear_bias
        harmonic = torch.squeeze(harmonic)

        return harmonic 



class Decoder(torch.nn.Module):
    def __init__(self, decoderin_channels, decoder_hid_channels):
        super(Decoder, self).__init__()
        self.mlp_out = nn.Sequential(
            nn.Linear(decoderin_channels, decoder_hid_channels*20, bias=True),
            nn.ReLU(),
            nn.Linear(decoder_hid_channels*20, decoder_hid_channels*10, bias=True),
            nn.ReLU(),
            nn.Linear(decoder_hid_channels*10, decoder_hid_channels, bias=True),
            nn.ReLU(),
            nn.Linear(decoder_hid_channels, decoder_hid_channels, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(decoder_hid_channels, 1, bias=True)
        )

    def forward(self, hyperedge_out_encoding):
        h = self.mlp_out(hyperedge_out_encoding).squeeze()
        return torch.sigmoid(h)

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  

    def forward(self, anchor, positive, negative):
        d_positive = F.pairwise_distance(anchor, positive, p=2)
        d_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = torch.relu(d_positive - d_negative + self.margin)
        return torch.mean(losses)