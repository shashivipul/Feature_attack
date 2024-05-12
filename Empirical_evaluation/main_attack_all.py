"""
Contains the complete implementation to reproduce the results of the attacks
results in the case of the "Cora", "CiteSeer" and "PubMed" datataset.
---
The implementation contains all the related benchmarks:
    - GCN
    - RGCN
    - ParsevalR
    - GCORN

For the GCN-K and the Air-GNN, refer to their official repo.
"""

import argparse
import os.path as osp

import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor

from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch_geometric.datasets import CitationFull

from utils import *
import pickle
import time
from pgd_attack import RandomNoise

from GCORN import GCORN, normalize_tensor_adj
from gcn import GCN, normalize_tensor_adj
from r_gcn import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--attack', type=str, default ="random", help='Type of attack')
    parser.add_argument('--budget', type=float, default=0.5, help='Attack budget')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_exp = 3


    if args.dataset == "CS":
        dataset = Coauthor(root="./data/", name="CS",
                                transform=T.NormalizeFeatures())


    if args.dataset == "cora_ml":
        dataset = CitationFull("./data/", args.dataset, transform=T.NormalizeFeatures())

    else:
        dataset = Planetoid("./data/", args.dataset, transform=T.NormalizeFeatures())



    data = dataset[0]

    data = data.to(device)
    adj_true = to_dense_adj(data.edge_index)[0, :,:]
    norm_adj = normalize_tensor_adj(adj_true)

    l_time_gcn = []
    l_time_parseval = []
    l_time_rgcn = []
    l_time_gcorn = []


    l_acc_GCORN = []
    l_acc_GCORN_attacked = []

    l_acc_gcn = []
    l_acc_gcn_attacked = []

    l_acc_parseval = []
    l_acc_parseval_attacked = []


    l_acc_rgcn = []
    l_acc_rgcn_attacked = []


    for exp in range(num_exp):
        # Generate random noise attack
        if args.attack == "random":
            random_noise = RandomNoise(args.budget)
            perturbed_x = random_noise.perturb(data)

        # Train GCORN
        input_time = time.time()
        model_ro = GCORN(dataset.num_features, args.hidden_channels,
                                                dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model_ro.parameters(), lr = args.lr)

        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_ro, loss = train(model_ro, optimizer, data, norm_adj)
            train_acc, val_acc, tmp_test_acc = test(model_ro, data, norm_adj)
            if val_acc > best_val_acc:
                best_model_ro = copy.deepcopy(model_ro)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        # Run accuracy on both normal data and attacked data
        output_time = time.time()
        acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_ro,
                                                data, norm_adj, perturbed_x)

        l_time_gcorn.append(output_time-input_time)
        l_acc_GCORN.append(acc_1)
        l_acc_GCORN_attacked.append(acc_2)

        # Train GCN
        input_time = time.time()
        model_gcn = GCN(dataset.num_features, args.hidden_channels,
                                            dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model_gcn.parameters(), lr = args.lr)

        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_gcn, loss = train(model_gcn, optimizer, data, norm_adj)
            train_acc, val_acc, tmp_test_acc = test(model_gcn, data, norm_adj)
            if val_acc > best_val_acc:
                best_model_gcn = copy.deepcopy(model_gcn)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        output_time = time.time()
        acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_gcn, data,
                                                        norm_adj, perturbed_x)

        l_time_gcn.append(output_time-input_time)
        l_acc_gcn.append(acc_1)
        l_acc_gcn_attacked.append(acc_2)

        # Train RGCN
        input_time = time.time()
        model_rgcn = RGCN(nnodes=norm_adj.shape[0], nfeat=data.x.shape[1],
                        nhid=16, nclass=dataset.num_classes , dropout=0.5,
                        device=device).to(device)
        optimizer = torch.optim.Adam(model_rgcn.parameters(), lr = args.lr)

        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_rgcn, loss = train(model_rgcn, optimizer, data, adj_true)
            train_acc, val_acc, tmp_test_acc = test(model_rgcn, data, adj_true)
            if val_acc > best_val_acc:
                best_model_rgcn = copy.deepcopy(model_rgcn)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        output_time = time.time()
        acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_rgcn,
                                                    data, adj_true, perturbed_x)

        l_time_rgcn.append(output_time-input_time)
        l_acc_rgcn.append(acc_1)
        l_acc_rgcn_attacked.append(acc_2)

        # Train Parseval GCN
        input_time = time.time()

        best_val_acc = final_test_acc = 0

        for alpha_val in [0.001, 0.01, 0.1, 1, 10]:

            model_gcn = GCN(dataset.num_features, args.hidden_channels,
                                                dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model_gcn.parameters(), lr = args.lr)


            for epoch in range(1, args.epochs + 1):
                model_gcn, loss = parseval_train(model_gcn, optimizer, data, norm_adj, alpha_val)
                train_acc, val_acc, tmp_test_acc = test(model_gcn, data, norm_adj)
                if val_acc > best_val_acc:
                    best_model_gcn = copy.deepcopy(model_gcn)
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc

        output_time = time.time()
        acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_gcn, data,
                                                        norm_adj, perturbed_x)

        l_time_parseval.append(output_time-input_time)
        l_acc_parseval.append(acc_1)
        l_acc_parseval_attacked.append(acc_2)



    print('For normal GCN: {} - {}' .format(np.mean(l_acc_gcn) * 100, np.std(l_acc_gcn) * 100))
    print('For normal GCN Peturbed : {} - {}' .format(np.mean(l_acc_gcn_attacked) * 100, np.std(l_acc_gcn_attacked) * 100))
    print('---')
    print('For Parseval: {} - {}' .format(np.mean(l_acc_parseval) * 100, np.std(l_acc_parseval) * 100))
    print('For Parseval Peturbed : {} - {}' .format(np.mean(l_acc_parseval_attacked) * 100, np.std(l_acc_parseval_attacked) * 100))
    print('---')
    print('For RGCN: {} - {}' .format(np.mean(l_acc_rgcn) * 100, np.std(l_acc_rgcn) * 100))
    print('For RGCN Peturbed : {} - {}' .format(np.mean(l_acc_rgcn_attacked) * 100, np.std(l_acc_rgcn_attacked) * 100))
    print('---')
    print('For GCORN: {} - {}' .format(np.mean(l_acc_GCORN) * 100, np.std(l_acc_GCORN) * 100))
    print('For GCORN Peturbed : {} - {}' .format(np.mean(l_acc_GCORN_attacked) * 100, np.std(l_acc_GCORN_attacked) * 100))
