"""
Contains the complete implementation to reproduce the results of the attacks
results in the case of the "OGB-Arxiv"
---
The implementation contains all the related benchmarks:
    - GCN
    - RGCN
    - ParsevalR
    - GCORN

For the GCN-K and the Air-GNN, refer to their official repo.
"""

import time
import argparse
import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, normalize, sparse_mx_to_torch_sparse_tensor
from utils import train, test, compute_acc_perturbation

from models import GCN, GCORN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from rgcn import *

from attack import RandomNoise

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Num Layers.')

    parser.add_argument('--attack', type=str, default ="random", help='Type of attack')
    parser.add_argument('--budget', type=float, default=0.5, help='Attack budget')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Budget is : {}' .format(args.budget))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_exp = 3

    # path = "/mimer/NOBACKUP/groups/naiss2023-22-288/"
    #

    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]

    #adj = to_torch_coo_tensor(add_self_loops(data.edge_index)[0])
    adj = to_scipy_sparse_matrix(add_self_loops(data.edge_index)[0])
    adj_true = to_scipy_sparse_matrix(add_self_loops(data.edge_index)[0])

    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features, labels = data.x, data.y.squeeze(1)
    #labels = torch.LongTensor(np.where(labels)[1])
    split_idx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idx['train'], split_idx['valid'], split_idx['test']

    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    l_time_gcn = []
    l_time_rgcn = []
    l_time_gcorn = []
    l_time_parseval = []

    l_acc_GCORN = []
    l_acc_GCORN_attacked = []

    l_acc_parseval = []
    l_acc_parseval_attacked = []

    l_acc_gcn = []
    l_acc_gcn_attacked = []

    l_acc_rgcn = []
    l_acc_rgcn_attacked = []

    evaluator = Evaluator(name='ogbn-arxiv')
    for exp in range(num_exp):
        # Generate random noise attack
        if args.attack == "random":
            random_noise = RandomNoise(args.budget)
            perturbed_x = random_noise.perturb(data).cuda()

        # for iter in range(20):
        #     print(iter)
        #     # Model and optimizer
        #     model_ro = GCORN(nfeat=features.shape[1],
        #                 nhid=args.hidden,
        #                 nclass=labels.max().item() + 1,
        #                 num_layers = args.num_layers,
        #                 dropout=args.dropout,
        #                 bjorck_iter = iter).cuda()
        #
        #     optimizer = torch.optim.Adam(model_ro.parameters(), lr=args.lr)

        # Model and optimizer
        input_time = time.time()
        model_ro = GCORN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    num_layers = args.num_layers,
                    dropout=args.dropout,
                    bjorck_iter = 15).cuda()

        optimizer = torch.optim.Adam(model_ro.parameters(), lr=args.lr)


        # Train GCORN model
        model_ro.reset_parameters()
        t_total = time.time()

        best_val = 0
        for epoch in range(args.epochs):
            loss = train(epoch, model_ro, optimizer, features, adj, labels, idx_train)
            result = test(model_ro, features, adj, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result

            if valid_acc > best_val:
                best_model_ro = copy.deepcopy(model_ro)
                best_val = valid_acc
                best_test = test_acc

        #print("Acc is GCORN : {}" .format(best_test))

        # Run accuracy on both normal data and attacked data
        output_time = time.time()
        acc_1, acc_2 = compute_acc_perturbation(best_model_ro, features, adj, perturbed_x, data, split_idx, evaluator)
        print("Acc is GCORN : {}" .format(acc_1))
        print("Acc attacked is GCORN : {}" .format(acc_2))

        l_time_gcorn.append(output_time-input_time)
        l_acc_GCORN.append(acc_1)
        l_acc_GCORN_attacked.append(acc_2)

        # Train RGCN model

        # Model and optimizer
        input_time = time.time()
        model_rgcn = RGCN(nnodes=adj_true.shape[0],
                        nfeat=data.x.shape[1], nhid=1024,
                        nclass=dataset.num_classes ,
                        dropout=0.5, device=device).to(device)

        optimizer = torch.optim.Adam(model_rgcn.parameters(), lr=args.lr)

        # Train GCORN model
        #model_rgcn.reset_parameters()

        best_val = 0
        for epoch in range(args.epochs):
            loss = train(epoch, model_rgcn, optimizer, features, adj_true, labels, idx_train)
            result = test(model_rgcn, features, adj_true, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result

            if valid_acc > best_val:
                best_model_rgcn = copy.deepcopy(model_rgcn)
                best_val = valid_acc
                best_test = test_acc


        output_time = time.time()
        acc_1, acc_2 = compute_acc_perturbation(best_model_rgcn, features, adj_true, perturbed_x, data, split_idx, evaluator)
        print("Acc RGCN is : {}" .format(acc_1))
        print("Attacked Acc RGCN is : {}" .format(acc_2))

        l_time_rgcn.append(output_time-input_time)
        l_acc_rgcn.append(acc_1)
        l_acc_rgcn_attacked.append(acc_2)

        # Train GCN model

        # Model and optimizer
        input_time = time.time()
        model_gcn = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    num_layers = args.num_layers,
                    dropout=args.dropout).cuda()

        optimizer = torch.optim.Adam(model_gcn.parameters(), lr=args.lr)

        # Train GCN model
        model_gcn.reset_parameters()

        best_val = 0
        for epoch in range(args.epochs):
            loss = train(epoch, model_gcn, optimizer, features, adj, labels, idx_train)
            result = test(model_gcn, features, adj, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result

            if valid_acc > best_val:
                best_model_gcn = copy.deepcopy(model_gcn)
                best_val = valid_acc
                best_test = test_acc


        output_time = time.time()
        acc_1, acc_2 = compute_acc_perturbation(best_model_gcn, features, adj, perturbed_x, data, split_idx, evaluator)
        print("Acc GCN is : {}" .format(acc_1))
        print("Attacked Acc GCN is : {}" .format(acc_2))

        l_time_gcn.append(output_time-input_time)
        l_acc_gcn.append(acc_1)
        l_acc_gcn_attacked.append(acc_2)


        # Train Parseval GCN model
        # Model and optimizer
        input_time = time.time()
        model_parseval = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    num_layers = args.num_layers,
                    dropout=args.dropout).cuda()

        optimizer = torch.optim.Adam(model_parseval.parameters(), lr=args.lr)

        # Train GCN model
        model_parseval.reset_parameters()

        best_val = 0
        for epoch in range(args.epochs):
            loss = train(epoch, model_parseval, optimizer, features, adj, labels, idx_train)
            result = test(model_parseval, features, adj, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result

            if valid_acc > best_val:
                best_model_parseval = copy.deepcopy(model_parseval)
                best_val = valid_acc
                best_test = test_acc


        output_time = time.time()
        acc_1, acc_2 = compute_acc_perturbation(best_model_parseval, features, adj, perturbed_x, data, split_idx, evaluator)
        print("Acc Parsreval is : {}" .format(acc_1))
        print("Attacked Acc Parsreval is : {}" .format(acc_2))

        l_time_parseval.append(output_time-input_time)
        l_acc_parseval.append(acc_1)
        l_acc_parseval_attacked.append(acc_2)

    print('*******')
    print('For Vanilla GCN: {} - {}' .format(np.mean(l_acc_gcn) * 100, np.std(l_acc_gcn) * 100))
    print('For Vanilla GCN Peturbed : {} - {}' .format(np.mean(l_acc_gcn_attacked) * 100, np.std(l_acc_gcn_attacked) * 100))
    print('---')
    print('For RGCN: {} - {}' .format(np.mean(l_acc_rgcn) * 100, np.std(l_acc_rgcn) * 100))
    print('For RGCN Peturbed : {} - {}' .format(np.mean(l_acc_rgcn_attacked) * 100, np.std(l_acc_rgcn_attacked) * 100))
    print('---')
    print('For Parseval: {} - {}' .format(np.mean(l_acc_parseval) * 100, np.std(l_acc_parseval) * 100))
    print('For Parseval Peturbed : {} - {}' .format(np.mean(l_acc_parseval_attacked) * 100, np.std(l_acc_parseval_attacked) * 100))
    print('---')
    print('For GCORN: {} - {}' .format(np.mean(l_acc_GCORN) * 100, np.std(l_acc_GCORN) * 100))
    print('For GCORN Peturbed : {} - {}' .format(np.mean(l_acc_GCORN_attacked) * 100, np.std(l_acc_GCORN_attacked) * 100))
    print('---')
    print('*******')

    print('Time for the GCN: {} - {}' .format(np.mean(l_time_gcn), np.std(l_time_gcn)))
    print('Time for the RGCN: {} - {}' .format(np.mean(l_time_rgcn), np.std(l_time_rgcn)))
    print('Time for the Parseval: {} - {}' .format(np.mean(l_time_parseval), np.std(l_time_parseval)))
    print('Time for the GCORN: {} - {}' .format(np.mean(l_time_gcorn), np.std(l_time_gcorn)))
    print('*******')
