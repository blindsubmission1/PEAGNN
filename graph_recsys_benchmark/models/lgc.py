import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class LGCRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(LGCRecsysModel, self).__init__(**kwargs)

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

        self.att = Parameter(torch.Tensor(3, 1))

        self.conv1 = GCNConv(
            kwargs['emb_dim'],
            kwargs['hidden_size'],
        )
        self.conv2 = GCNConv(
            kwargs['hidden_size'],
            kwargs['hidden_size'],
        )
        self.conv3 = GCNConv(
            kwargs['hidden_size'],
            kwargs['hidden_size'],
        )

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.att)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x_1 = F.dropout(self.conv1(x, edge_index), p=self.dropout, training=self.training)
        x_2 = F.dropout(self.conv2(x_1, edge_index), p=self.dropout, training=self.training)
        x_3 = F.dropout(self.conv3(x_2, edge_index), p=self.dropout, training=self.training)
        concated_x = torch.cat([x_1.unsqueeze(0), x_2.unsqueeze(0), x_3.unsqueeze(0)],dim=0)
        att = torch.ones_like(x_1).unsqueeze(0).repeat_interleave(3, dim=0)
        att *= F.softmax(self.att, dim=0).reshape(3, 1, 1)
        x = torch.sum(concated_x * att, dim=0)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)