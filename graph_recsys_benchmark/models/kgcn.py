import torch
import torch.nn.functional as F
from torch.nn import Parameter
from graph_recsys_benchmark.nn import KGCNConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class KGCNRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(KGCNRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.dropout = kwargs['dropout']

        self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        self.r = Parameter(torch.Tensor(kwargs['dataset'].num_edge_types, kwargs['emb_dim']))
        self.proj_mat = Parameter(torch.Tensor(kwargs['emb_dim'], kwargs['emb_dim']))

        self.edge_index, self.edge_attr = self.update_graph_input(kwargs['dataset'])

        self.conv1 = KGCNConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
        )
        self.conv2 = KGCNConv(
            kwargs['hidden_size'],
            kwargs['hidden_size'] // 2,
        )
        self.conv3 = KGCNConv(
            kwargs['hidden_size'] // 2,
            kwargs['hidden_size'] // 4,
        )

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.r)
        glorot(self.proj_mat)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, att_map):
        x, edge_index, edge_attr = self.x, self.edge_index, self.edge_attr
        x_1 = F.dropout(self.conv1(x, edge_index, att_map), p=self.dropout, training=self.training)
        x_2 = F.dropout(self.conv2(x_1, edge_index, att_map), p=self.dropout, training=self.training)
        x_3 = F.dropout(self.conv3(x_2, edge_index, att_map), p=self.dropout, training=self.training)
        x = torch.cat([F.normalize(x_1, dim=-1), F.normalize(x_2, dim=-1), F.normalize(x_3, dim=-1)], dim=-1)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)

    def kg_eval(self):
        super(GraphRecsysModel, self).eval()

    def cf_eval(self, att_map):
        super(GraphRecsysModel, self).eval()
        with torch.no_grad():
            self.cached_repr = self.forward(att_map)
