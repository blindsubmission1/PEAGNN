import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import glorot, zeros


class MultiGCCFConv(MessagePassing):
    def __init__(
            self, in_channels, out_channels, **kwargs):
        super(MultiGCCFConv, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = Parameter(torch.Tensor(in_channels + out_channels, out_channels))
        self.Q = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.Q)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return torch.mm(x_j, self.Q)

    def update(self, aggr_out, x):
        aggr_out = torch.cat([x, torch.tanh(aggr_out)], dim=-1)
        aggr_out = torch.tanh(torch.mm(aggr_out, self.W))
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
