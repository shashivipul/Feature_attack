'''
    This script contains the implementation of our proposed GCORN.
    Please refer to the details of the method in the main paper.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BGraphConvolution


class GCORN(nn.Module):
    def __init__(self, nfeat, nhid, num_layers, nclass, dropout, bjorck_iter):
        super(GCORN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(BGraphConvolution(nfeat, nhid, bjorck_iter))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))

        for _ in range(num_layers - 2):
            self.convs.append(
                BGraphConvolution(nhid, nhid))
            self.bns.append(torch.nn.BatchNorm1d(nhid))

        self.convs.append(BGraphConvolution(nhid, nclass, bjorck_iter))
        self.dropout = dropout

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x.log_softmax(dim=-1)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, num_layers, nclass, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhid))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))

        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConvolution(nhid, nhid))
            self.bns.append(torch.nn.BatchNorm1d(nhid))

        self.convs.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x.log_softmax(dim=-1)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
