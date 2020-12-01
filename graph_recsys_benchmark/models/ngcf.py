import torch
import torch.nn.functional as F
from torch.nn import Parameter
from graph_recsys_benchmark.nn import NGCFConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class NGCFRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(NGCFRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_index = self.update_graph_input(kwargs['dataset'])

        self.conv1 = NGCFConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = NGCFConv(kwargs['hidden_size'], kwargs['hidden_size'] // 2)
        self.conv3 = NGCFConv(kwargs['hidden_size'] // 2, kwargs['hidden_size'] // 4)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x_1 = F.dropout(self.conv1(x, edge_index), p=self.dropout, training=self.training)
        x_2 = F.dropout(self.conv2(x_1, edge_index), p=self.dropout, training=self.training)
        x_3 = F.dropout(self.conv3(x_2, edge_index), p=self.dropout, training=self.training)
        x = torch.cat([F.normalize(x_1, dim=-1), F.normalize(x_2, dim=-1), F.normalize(x_3, dim=-1)], dim=-1)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)
