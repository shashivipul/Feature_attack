'''
    This script contains the implementation of our proposed GCORN.
    Please refer to the details of the method in the main paper.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from torch.nn import Parameter
import numpy as np
from matrix_ortho import *

class ConvClass(nn.Module):
    def __init__(self, input_dim , output_dim, activation):
        super(ConvClass, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))

        self.activation = activation

        self.reset_parameters()

        self.safe_scaling = True
        self.bjorck_beta = 0.5
        self.bjorck_iter = 25
        self.bjorck_order = 2

    def forward(self, x, adj):

        if self.safe_scaling:
            scaling = scale_values(self.weight.data).to(x.device)

        else:
            scaling = 1.0

        ortho_w = orthonormalize_weights(self.weight.t() / scaling,
                                        beta = self.bjorck_beta,
                                        iters = self.bjorck_iter,
                                        order = self.bjorck_order).t()

        x = F.linear(x, ortho_w)

        return self.activation(torch.mm(adj,x))

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)


def normalize_tensor_adj(adj):
    n = adj.shape[0]
    A = adj + torch.eye(n).to(adj.device)
    D = torch.sum(A, 0)
    D_hat = torch.diag(((D) ** (-0.5)))
    adj_normalized = torch.mm(torch.mm(D_hat, A), D_hat)

    return adj_normalized

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.lin(x)
        return x


class GCORN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.activation = nn.ReLU()

        self.conv1 = ConvClass(in_channels, hidden_channels, activation = self.activation)
        self.conv2 = ConvClass(hidden_channels, hidden_channels, activation = self.activation)

        self.lin = MLPClassifier(hidden_channels, out_channels, self.activation)


    def forward(self, x, adj, edge_weight=None):
        x = self.conv1(x, adj)
        # x = self.activation(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)#x
