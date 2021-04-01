import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import glorot, zeros


class SumAggregatorConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SumAggregatorConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return torch.tanh(aggr_out)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
