
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np
import math
import torch
import torch.nn as nn
from matrix_ortho import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bjorck_iter=10, bias=True):
        super(BGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.reset_parameters()

        self.safe_scaling = True
        self.bjorck_beta = 0.5
        self.bjorck_iter = bjorck_iter
        self.bjorck_order = 1


    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)

    def forward(self, input, adj):

        if self.safe_scaling:
            scaling = scale_values(self.weight.data).to(input.device)

        else:
            scaling = 1.0

        ortho_w = orthonormalize_weights(self.weight.t() / scaling,
                                        beta = self.bjorck_beta,
                                        iters = self.bjorck_iter,
                                        order = self.bjorck_order).t()


        support = torch.mm(input, ortho_w)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
