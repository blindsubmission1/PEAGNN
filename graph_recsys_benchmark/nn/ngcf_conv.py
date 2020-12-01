import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import glorot, zeros


class NGCFConv(MessagePassing):
    def __init__(
            self, in_channels, out_channels,
            negative_slope=0.2, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        self.W_1 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.W_2 = Parameter(
            torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W_1)
        glorot(self.W_2)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

        if not hasattr(self, 'deg'):
            deg = []
            for i in range(x.shape[0]):
                deg.append(torch.sum(torch.sum(edge_index == i)).cpu().item())
            self.deg = torch.tensor(deg, dtype=torch.long, device=x.device).view(-1, 1) / 2
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, edge_index_i, edge_index_j):
        coff = 1 / torch.sqrt((self.deg[edge_index_i] * self.deg[edge_index_j]).float())
        return coff * (torch.mm(x_j, self.W_1) + torch.mm((x_i * x_j), self.W_2))

    def update(self, aggr_out, x):
        add_aggr = F.leaky_relu(torch.mm(x, self.W_1) + aggr_out, negative_slope=self.negative_slope)
        return add_aggr

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
