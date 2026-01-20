import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import logging
from utils import *
import torch.nn as nn
from module import get_time
import nni

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_val_test(train_val_data, linegraphSE, hyperSE, optimizer1, optimizer2, rand_samplers, args, nodes_entropy_encoding, hypergraph_adj, edges_entropy_encoding, 
              linegraph_adj, nodes_id_dict, edge_id_list, max_node_num, max_early_stop):
    train_pos_data, val_pos_data, test_pos_data = train_val_data
    train_rand_sampler, val_rand_sampler, test_rand_sampler = rand_samplers
    train_size = len(train_pos_data)
    val_size = len(val_pos_data)
    test_size = len(test_pos_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    src_l, he_offset_l = construct_algo_data_given_he_ids(train_pos_data)
    train_data_time = get_time(train_pos_data)
    test_acc, test_ap, test_f1, test_auc = 0, 0, 0, 0
    early_stop = 0
    max_auc = 0
    val_src_l, val_he_offset_l = construct_algo_data_given_he_ids(val_pos_data)
    val_data_time = get_time(val_pos_data)
    val_neg_data = val_rand_sampler.sample(val_src_l, val_he_offset_l, val_data_time)
    test_src_l, test_he_offset_l = construct_algo_data_given_he_ids(test_pos_data)
    test_data_time = get_time(test_pos_data)
    test_neg_data = test_rand_sampler.sample(test_src_l, test_he_offset_l, test_data_time)
    test_true_label = np.concatenate([np.ones(test_size), np.zeros(test_size)])
    pos_label = torch.ones(train_size, dtype=torch.float, device=device, requires_grad=False)
    neg_label = torch.zeros(train_size, dtype=torch.float, device=device, requires_grad=False)

    linegraphSE.train()
    for epoch in range(args.pre_epochs):
        print('start {} pre_epoch'.format(epoch))
        optimizer1.zero_grad()
        pos_edge_score, neg_edge_score = linegraphSE(edges_entropy_encoding, linegraph_adj, train_pos_data, edge_id_list)
        loss1 = criterion(pos_edge_score, pos_label) + criterion(neg_edge_score, neg_label)
        print('pre_Epoch: '+str(epoch)+' loss1: '+str(loss1))
        loss1.backward()
        optimizer1.step()
    linegraphSE.eval()

    for epoch in range(args.epochs):
        print('start {} epoch'.format(epoch))
        train_neg_data = train_rand_sampler.sample(src_l, he_offset_l, train_data_time)
        pos_label = torch.ones(train_size, dtype=torch.float, device=device, requires_grad=False)
        neg_label = torch.zeros(train_size, dtype=torch.float, device=device, requires_grad=False)

        with torch.no_grad():
            pos_linegraph_edge_encoding = linegraphSE.getencoding(edges_entropy_encoding, linegraph_adj, train_pos_data, edge_id_list)

        hyperSE.train()
        optimizer2.zero_grad()
        pos_score, neg_score, tri_loss = hyperSE(nodes_entropy_encoding, hypergraph_adj, train_pos_data, train_neg_data, nodes_id_dict, 
                                       max_node_num, pos_linegraph_edge_encoding)
        loss2 = tri_loss
        print('Epoch: '+str(epoch)+' loss2: '+str(loss2))
        loss2.backward()
        optimizer2.step()
        hyperSE.eval()

        with torch.no_grad():
            linegraphSE.eval()
            hyperSE.eval()
            pos_score, neg_score = hyperSE.score_for_test(nodes_entropy_encoding, hypergraph_adj, val_pos_data, val_neg_data, nodes_id_dict, max_node_num)
            pred_score = np.concatenate([pos_score.cpu().detach().numpy(), neg_score.cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(val_size), np.zeros(val_size)])
            val_acc = (pred_label.flatten() == true_label).mean()
            val_ap = average_precision_score(true_label, pred_score)
            val_f1 = f1_score(true_label, pred_label)
            val_auc = roc_auc_score(true_label, pred_score)
            nni.report_intermediate_result(val_auc)
            print('epoch: {}:'.format(epoch) + ', val auc: {}'.format(val_auc) + ', val ap: {}'.format(val_ap) + ', val f1: {}'.format(val_f1) + ', val acc: {}'.format(val_acc))

            if val_auc > max_auc:
                max_auc = val_auc
                test_pos_score, test_neg_score = hyperSE.score_for_test(nodes_entropy_encoding, hypergraph_adj, test_pos_data, test_neg_data, nodes_id_dict, max_node_num)
                test_pred_score = np.concatenate([test_pos_score.cpu().detach().numpy(), test_neg_score.cpu().detach().numpy()])
                test_pred_label = test_pred_score > 0.5
                test_acc = (test_pred_label.flatten() == test_true_label).mean()
                test_ap = average_precision_score(test_true_label, test_pred_score)
                test_f1 = f1_score(test_true_label, test_pred_label)
                test_auc = roc_auc_score(test_true_label, test_pred_score) 
                print('epoch: {}:'.format(epoch) + ', test auc: {}'.format(test_auc) + ', test ap: {}'.format(test_ap) + ', test f1: {}'.format(test_f1) + ', test acc: {}'.format(test_acc))
                early_stop = 0          
            else:
                early_stop = early_stop+1

            if early_stop >= max_early_stop:
                break
    nni.report_final_result(test_auc)
    return test_acc, test_ap, test_f1, test_auc





