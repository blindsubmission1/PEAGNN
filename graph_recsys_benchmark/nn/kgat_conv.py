import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import glorot, zeros


class KGATConv(MessagePassing):
    def __init__(
            self, in_channels, out_channels,
            negative_slope=0.2, bias=True, **kwargs):
        super(KGATConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        self.weight_add = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight_bi = Parameter(
            torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_add)
        glorot(self.weight_bi)
        zeros(self.bias)

    def forward(self, x, edge_index, att_map, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x, att_map=att_map)

    def message(self, x_j, att_map):
        return x_j * att_map.view(-1, 1)

    def update(self, aggr_out, x):
        add_aggr = x + aggr_out
        add_aggr = F.leaky_relu(torch.mm(add_aggr, self.weight_add), negative_slope=self.negative_slope)
        bi_aggr = x * aggr_out
        bi_aggr = F.leaky_relu(torch.mm(bi_aggr, self.weight_bi), negative_slope=self.negative_slope)
        aggr_out = add_aggr + bi_aggr
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
