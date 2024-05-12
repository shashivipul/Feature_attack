"""
Contains the complete implementation to reproduce the results of the "Mettack"
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
from deeprobust.graph.defense.gcorn import GCORN

from deeprobust.graph.defense.gcn_parseval import Parseval_GCN

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np
import os

from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float,     default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)

adj, features, labels = data.adj, data.features, data.labels

# features = normalize(features, axis=0)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)


perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)


#1. to CSR sparse
adj, features = csr_matrix(adj), csr_matrix(features)


"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1


# Setup GCN as the Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=False, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)


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
        dropout=0.5, device=device, bjorck_iter=iter_val, bjorck_order=25)

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

    best_acc_val = 0
    for alpha_val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:

        classifier = Parseval_GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1,
            dropout=0.5, scale_param = alpha_val, num_passes=2, device=device)


        classifier = classifier.to(device)

        classifier.fit(features, adj, labels, idx_train, train_iters=200,
                       idx_val=idx_val,
                       idx_test=idx_test,
                       verbose=False, attention=False) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
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

    if defense == "GCN":
        classifier = globals()[args.modelname](nfeat=features.shape[1], with_bias=False, nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "Guard":
        classifier = globals()[args.modelname](nfeat=features.shape[1], with_bias=False, nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    else:
        classifier = globals()[defense](nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
                                                  nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=attention)
    classifier.eval()

    acc_test, _ = classifier.test(idx_test)
    return acc_test.item()



if __name__ == '__main__':

    # Attack the model
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    modified_adj_sparse = csr_matrix(modified_adj.cpu().numpy())


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
        # Test on clean and attacked accuracy
        print('=== testing Normal GNN ===')

        acc_gcn_non_attacked = test(adj)
        acc_gcn_attacked = test(modified_adj_sparse)

        l_acc_gcn_non.append(acc_gcn_non_attacked)
        l_acc_gcn_att.append(acc_gcn_attacked)


        print('=== testing GNNGuard ===')
        attention = True
        acc_non_attacked = test(adj, defense="Guard")
        acc_attacked = test(modified_adj_sparse, defense="Guard")

        l_acc_gnnguard_non.append(acc_non_attacked)
        l_acc_gnnguard_att.append(acc_attacked)


        print('=== testing GCNJaccard ===')
        attention = False
        acc_jaccard_non_attacked = test(adj, defense = "GCNJaccard")
        acc_jaccard_attacked = test(modified_adj_sparse, defense = "GCNJaccard")

        l_acc_jaccard_non.append(acc_jaccard_non_attacked)
        l_acc_jaccard_att.append(acc_jaccard_attacked)
        #
        print('=== testing RGCN ===')
        attention = False
        acc_rgcn_non_attacked = test(adj, defense = "RGCN")
        acc_rgcn_attacked = test(modified_adj_sparse, defense = "RGCN")

        l_acc_rgcn_non.append(acc_rgcn_non_attacked)
        l_acc_rgcn_att.append(acc_rgcn_attacked)

        print('=== testing GCNSVD ===')
        attention = False
        acc_gcn_svd_non_attacked = test(adj, defense = "GCNSVD")
        acc_gcn_svd_attacked = test(modified_adj_sparse, defense = "GCNSVD")

        l_acc_gcn_svd_non.append(acc_gcn_svd_non_attacked)
        l_acc_gcn_svd_att.append(acc_gcn_svd_attacked)

        print('=== testing ParsevalGCN ===')
        attention=False
        acc_parseval_clean = test_parseval(adj)
        acc_parseval_attacked = test_parseval(modified_adj_sparse)

        l_acc_parseval_non.append(acc_parseval_clean)
        l_acc_parseval_att.append(acc_parseval_attacked)


        print('=== testing GCORN ===')
        attention=False
        acc_bjorck_clean = test_gcorn(adj)
        acc_bjorck_attacked = test_gcorn(modified_adj_sparse)

        l_acc_bjorck_non.append(acc_bjorck_clean)
        l_acc_bjorck_att.append(acc_bjorck_attacked)


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
