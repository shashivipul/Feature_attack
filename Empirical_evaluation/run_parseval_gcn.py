"""
Main script to run Parseval GCN
"""

import argparse
import os.path as osp

import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from utils import *
import pickle
import time
from pgd_attack import RandomNoise

from gcn import GCN, normalize_tensor_adj
from parseval_constraint import *

import time
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_exp = 1

    dataset = Planetoid("./data/", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    data = data.to(device)
    adj_true = to_dense_adj(data.edge_index)[0, :,:]
    norm_adj = normalize_tensor_adj(adj_true)

    l_acc = []
    l_time = []

    l_epoch = []
    l_loss_train = []
    l_acc_train = []
    l_acc_val = []

    for exp in range(num_exp):
        input_time = time.time()
        model_gcn = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model_gcn.parameters(), lr = args.lr, weight_decay = 5e-4)

        best_val_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_gcn, loss = parseval_train(model_gcn, optimizer, data, norm_adj)
            train_acc, val_acc, tmp_test_acc = test(model_gcn, data, norm_adj)
            if val_acc > best_val_acc:
                best_model = copy.deepcopy(model_gcn)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            if epoch % 5 == 0:
                
                l_epoch.append(epoch)
                l_loss_train.append(loss)
                l_acc_train.append(train_acc)
                l_acc_val.append(val_acc)

        output_time = time.time()

        l_acc.append(test_acc)
        l_time.append(output_time-input_time)

    print('Accuracy results for the GCN: {} - {}' .format(np.mean(l_acc) * 100, np.std(l_acc) * 100))
    print('Time for the GCN: {} - {}' .format(np.mean(l_time), np.std(l_time)))
