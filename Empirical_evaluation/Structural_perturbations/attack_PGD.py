"""
Contains the complete implementation to reproduce the results of the "PGD"
Attack.
---
The implementation contains all the related benchmarks:
    - Normal GCN
    - GCNGuard
    - RGCN
    - GCN-Jaccard
    - Parseval GCN
    - Our proposed GCORN.

To use the benchmarks (GCNGuard, RGCN ...), please adapt the argument "defense"
in the "test" function. We provided an example of their use in the main section
of this file.
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack.topology_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN
import scipy

from deeprobust.graph.defense.gcorn import GCORN
from deeprobust.graph.defense.gcn_parseval import Parseval_GCN

from scipy.sparse import csr_matrix

from deeprobust.graph.defense.original_gcn import Original_GCN

# import args as arg

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora',
                'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,
                                                        help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD',
                            choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


from torch_geometric.datasets import Planetoid
from deeprobust.graph.data import Pyg2Dpr

data = Dataset(root='/tmp/', name=args.dataset)

def normalize_features(value):
    value = value - value.min()
    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return value

def test_gcorn(adj):
    """
    Main function to test our proposed GCORN defense approach
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested

    Output:
        acc_test: The resulting accuracy test
    """


    classifier = GCORN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
        dropout=0.5, device=device, bjorck_iter=25, bjorck_order=1)

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=200,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=False)
    classifier.eval()

    acc_val, _ = classifier.test(idx_val)
    acc_test, _ = classifier.test(idx_test)


    return acc_test.item()

def test_parseval(new_adj):
    """
    Main function to test the Parseval GCN
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested

    Output:
        acc_test: The resulting accuracy test
    """

    if not new_adj.is_sparse:
        new_adj = csr_matrix(new_adj.cpu())

    if not features.is_sparse:
        features_local = csr_matrix(features)

    best_acc_val = 0
    for alpha_val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:

        classifier = Parseval_GCN(nfeat=features_local.shape[1], nhid=16, nclass=labels.max().item() + 1,
            dropout=0.5, scale_param = alpha_val, num_passes=2, device=device)


        classifier = classifier.to(device)

        classifier.fit(features_local, new_adj, labels, idx_train, train_iters=200,
                       idx_val=idx_val,
                       idx_test=idx_test,
                       verbose=False, attention=False)
        classifier.eval()

        acc_val, _ = classifier.test(idx_val)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            acc_test, _ = classifier.test(idx_test)
            # best_test_val = acc_test
            # best_epsi = epsi


    return acc_test.item()

def test(new_adj, defense = "GCN"):
    """
    Main function to test the considered benchmarks
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested
        defense (str,): The considered defense method (Guard, Jaccard ..)

    Output:
        acc_test: The resulting accuracy test
    """

    if not new_adj.is_sparse:
        new_adj = csr_matrix(new_adj.cpu())

    if not features.is_sparse:
        features_local = csr_matrix(features)

    if defense == "GCN":
        classifier = GCN(nfeat=features_local.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "Guard":
        classifier = GCN(nfeat=features_local.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    else:
        classifier = globals()[defense](nnodes=new_adj.shape[0],
                nfeat=features_local.shape[1], nhid=16,
                nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features_local, new_adj, labels, idx_train,
                    train_iters=201, idx_val=idx_val, idx_test=idx_test,
                    verbose=False, attention=attention)

    classifier.eval()
    output = classifier.predict().cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


if __name__ == '__main__':
    """
    Main function containing the PGD implementation, please note that you
    need to uncomment the last part to use the other benchamarks
    """

    l_acc_gcn_non = []
    l_acc_gcn_att = []

    l_acc_jaccard_non = []
    l_acc_jaccard_att = []

    l_acc_gnnguard_non = []
    l_acc_gnnguard_att = []

    l_acc_bjorck_non = []
    l_acc_bjorck_att = []

    l_acc_rgcn_non = []
    l_acc_rgcn_att = []

    l_acc_gcn_svd_non = []
    l_acc_gcn_svd_att = []

    l_acc_parseval_non = []
    l_acc_parseval_att = []


    for exp in range(3):


        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
        sparse_adj, sparse_features = csr_matrix(adj), csr_matrix(features)

        # Train the surrogate model
        target_gcn = Original_GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5,
                  device=device)

        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train)

        # Setup Attack Model
        print('=== setup attack model ===')
        model = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='CE',
                                                                    device=device)
        model = model.to(device)

        # Attack the model
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj


        print('=== testing Normal GNN ===')
        acc_gcn_non_attacked = test(adj)
        acc_gcn_attacked = test(modified_adj)

        l_acc_gcn_non.append(acc_gcn_non_attacked)
        l_acc_gcn_att.append(acc_gcn_attacked)


        print('=== testing GNNGuard ===')
        attention = True
        acc_non_attacked = test(adj, defense="Guard")
        acc_attacked = test(modified_adj, defense="Guard")

        l_acc_gnnguard_non.append(acc_non_attacked)
        l_acc_gnnguard_att.append(acc_attacked)


        print('=== testing GCNJaccard ===')
        attention = False
        acc_jaccard_non_attacked = test(adj, defense = "GCNJaccard")
        acc_jaccard_attacked = test(modified_adj, defense = "GCNJaccard")

        l_acc_jaccard_non.append(acc_jaccard_non_attacked)
        l_acc_jaccard_att.append(acc_jaccard_attacked)
        #
        print('=== testing RGCN ===')
        attention = False
        acc_rgcn_non_attacked = test(adj, defense = "RGCN")
        acc_rgcn_attacked = test(modified_adj, defense = "RGCN")

        l_acc_rgcn_non.append(acc_rgcn_non_attacked)
        l_acc_rgcn_att.append(acc_rgcn_attacked)

        print('=== testing GCNSVD ===')
        attention = False
        acc_gcn_svd_non_attacked = test(adj, defense = "GCNSVD")
        acc_gcn_svd_attacked = test(modified_adj, defense = "GCNSVD")

        l_acc_gcn_svd_non.append(acc_gcn_svd_non_attacked)
        l_acc_gcn_svd_att.append(acc_gcn_svd_attacked)

        print('=== testing ParsevalGCN ===')
        attention=False
        acc_parseval_clean = test_parseval(adj)
        acc_parseval_attacked = test_parseval(modified_adj)

        l_acc_parseval_non.append(acc_parseval_clean)
        l_acc_parseval_att.append(acc_parseval_attacked)


        print('=== testing GCORN ===')
        attention=False
        acc_bjorck_clean = test_gcorn(adj)
        acc_bjorck_attacked = test_gcorn(modified_adj)

        l_acc_bjorck_non.append(acc_bjorck_clean)
        l_acc_bjorck_att.append(acc_bjorck_attacked)



    print('For Dataset : {} and budget : {}' .format(args.dataset, args.ptb_rate) )
    print('---------------')
    print("GCN Non Attacked Acc - {} - {} " .format(np.mean(l_acc_gcn_non), np.std(l_acc_gcn_non)))
    print("GNNGuard Non Attacked Acc - {} - {}  " .format(np.mean(l_acc_gnnguard_non), np.std(l_acc_gnnguard_non)))
    print("GCNJaccard Non Attacked Acc - {} - {} " .format(np.mean(l_acc_jaccard_non), np.std(l_acc_jaccard_non)))
    print("RGCN Non Attacked Acc - {} - {} " .format(np.mean(l_acc_rgcn_non), np.std(l_acc_rgcn_non)))
    print("GCNSVD Non Attacked Acc - {} - {} " .format(np.mean(l_acc_gcn_svd_non), np.std(l_acc_gcn_svd_non)))
    print("GCORN Non Attacked Acc - {} - {} " .format(np.mean(l_acc_bjorck_non), np.std(l_acc_bjorck_non)))
    print("Parseval Non Attacked Acc - {} - {} " .format(np.mean(l_acc_parseval_non), np.std(l_acc_parseval_non)))


    print('---------------')

    print("GCN Attacked Acc - {} - {} " .format(np.mean(l_acc_gcn_att), np.std(l_acc_gcn_att)))
    print("GNNGuard Attacked Acc - {} - {} " .format(np.mean(l_acc_gnnguard_att), np.std(l_acc_gnnguard_att)))
    print("GCNJaccard Attacked Acc - {} - {} " .format(np.mean(l_acc_jaccard_att), np.std(l_acc_jaccard_att)))
    print("RGCN Attacked Acc - {} - {} " .format(np.mean(l_acc_rgcn_att), np.std(l_acc_rgcn_att)))
    print("GCNSVD Attacked Acc - {} - {} " .format(np.mean(l_acc_gcn_svd_att), np.std(l_acc_gcn_svd_att)))
    print("GCORN Attacked Acc - {} - {} " .format(np.mean(l_acc_bjorck_att), np.std(l_acc_bjorck_att)))
    print("Parseval Attacked Acc - {} - {} " .format(np.mean(l_acc_parseval_att), np.std(l_acc_parseval_att)))

    print('---------------')
