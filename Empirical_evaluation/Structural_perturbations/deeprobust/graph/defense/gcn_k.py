from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F



from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    # torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    # is_torch_sparse_tensor,
    scatter,
    # spmm,
    # to_edge_index,
)

from torch_sparse import spmm
from torch_geometric.utils import remove_self_loops, degree, add_self_loops, negative_sampling

from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.utils.sparse import set_sparse_value

from deeprobust.graph import utils

@torch.jit._overload
def gcn_norm(edge_index, edge_weight, num_nodes, improved, add_self_loops,
             flow, dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight, num_nodes, improved, add_self_loops,
             flow, dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.


    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GCNConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,eps:float,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        #super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, edge_attr_kernel: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        out_2 = self.propagate(edge_index, x=x, edge_weight=edge_attr_kernel,
                             size=None)
        out = self.eps * out_1 + (1-self.eps) *out_2
        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)




import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_add_pool

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

class GCNK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, eps = 0.5,
                lr=0.01, weight_decay=5e-4):
        super().__init__()

        self.eps = eps
        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        self.lr = lr
        self.weight_decay = weight_decay

        self.conv1 = GCNConv(in_channels = in_channels, out_channels = hidden_channels ,eps= self.eps)
        self.conv2 =  GCNConv(in_channels = hidden_channels,out_channels=  hidden_channels,eps= self.eps)
        self.lin = MLPClassifier(hidden_channels, out_channels, self.activation)


    def forward(self, data_temp):

        x, edge_index, edge_weight, batch = data_temp.x, data_temp.edge_index, data_temp.edge_weight, data_temp.batch
        edge_index_kernel, edge_attr_kernel  = data_temp.edge_index_kernel.detach(), data_temp.edge_attr_kernel.detach()


        # Normalization of the Kernel
        row, col = edge_index_kernel[0], edge_index_kernel[1]
        idx = col
        num_nodes = data_temp.x.size(0)
        deg = scatter(edge_attr_kernel, idx, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_attr_kernel = deg_inv_sqrt[row] * edge_attr_kernel * deg_inv_sqrt[col]

        x = self.conv1(x, edge_index,None,  edge_attr_kernel)
        x = self.activation(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, None, edge_attr_kernel)
        x = self.activation(x)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


    def fit(self, data_temp, verbose = False):

        ratio = 1.0
        edge_index, device = data_temp.edge_index, data_temp.edge_index.device
        num_edges, num_nodes = edge_index.size(1), data_temp.x.size(0)
        edge_prob = min(1.0, ratio*num_edges/(num_nodes*(num_nodes-1)))
        print("edge_probability is {}\n".format(edge_prob))
        edge_index_kernel = edge_index
        edge_index_kernel, _ = remove_self_loops(edge_index_kernel, edge_attr=None)
        edge_index_kernel, _ = add_self_loops(edge_index_kernel)
        row_kernel, col_kernel = edge_index_kernel
        Q = torch.bmm(data_temp.x[row_kernel].unsqueeze(-2), data_temp.x[col_kernel].unsqueeze(-1)).squeeze().view(-1, 1)
        data_temp.edge_index_kernel, data_temp.edge_attr_kernel = edge_index_kernel, Q

        self.labels = data_temp.y.squeeze(1)

        self.device = self.conv1.bias.device
        self._train_without_val(data_temp, verbose)


    def _train_without_val(self, data_temp, verbose = False):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_iters= 300
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(data_temp)
            loss_train = F.nll_loss(output[data_temp.train_mask], self.labels[data_temp.train_mask])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = output = self.forward(data_temp)
        self.output = output


    def test(self, idx_test):
        # output = self.forward()
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output
