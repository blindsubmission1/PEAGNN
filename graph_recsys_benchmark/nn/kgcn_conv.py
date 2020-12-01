import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import glorot, zeros


class KGCNConv(MessagePassing):
    def __init__(
            self, in_channels, out_channels,
            negative_slope=0.2, bias=True, **kwargs):
        super(KGCNConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, att_map, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x, att_map=att_map)

    def message(self, x_j, att_map):
        return x_j * att_map.view(-1, 1)

    def update(self, aggr_out, x):
        aggr_out = F.relu(torch.mm(aggr_out + x, self.weight) + self.bias)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
