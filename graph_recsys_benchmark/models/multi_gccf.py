import torch
from torch.nn import Parameter
from graph_recsys_benchmark.nn import MultiGCCFConv, SumAggregatorConv
from torch_geometric.nn.inits import glorot

from .base import GraphRecsysModel


class MultiGCCFRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(MultiGCCFRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_index, self.user2item_edge_index, self.item2user_edge_index = self.update_graph_input(kwargs['dataset'])

        self.skip_fc = Parameter(torch.Tensor(kwargs['emb_dim'], kwargs['repr_dim']))
        self.aggr_user = SumAggregatorConv(kwargs['hidden_size'], kwargs['repr_dim'])
        self.aggr_item = SumAggregatorConv(kwargs['hidden_size'], kwargs['repr_dim'])

        self.conv1 = MultiGCCFConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = MultiGCCFConv(kwargs['hidden_size'], kwargs['repr_dim'])

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.skip_fc)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.aggr_user.reset_parameters()
        self.aggr_item.reset_parameters()

    def forward(self):
        x, edge_index, user2item_edge_index, item2user_edge_index = self.x, self.edge_index, self.user2item_edge_index, self.item2user_edge_index
        x_1 = self.conv1(x, edge_index)
        x_repr = self.conv2(x_1, edge_index)

        x_skip = torch.mm(x, self.skip_fc)

        x_user = self.aggr_user(x_repr, item2user_edge_index)
        x_item = self.aggr_item(x_repr, user2item_edge_index)

        return x_repr + x_skip + x_user + x_item

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)