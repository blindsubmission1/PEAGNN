import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot

from .base import MFRecsysModel


class CFKGRecsysModel(MFRecsysModel):
    def __init__(self, **kwargs):
        super(CFKGRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        self.r = Parameter(torch.Tensor(kwargs['dataset'].num_edge_types, kwargs['emb_dim']))
        self.rating_r_idx = kwargs['dataset'].edge_type_dict['user2item']

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.r)

    def forward(self, unids, inids):
        u_repr = self.x[unids]
        i_repr = self.x[inids]
        r_repr = self.r[torch.ones((unids.shape[0]), dtype=torch.long) * self.rating_r_idx]
        pred = torch.sum((u_repr + r_repr) * i_repr, dim=-1)
        pred = torch.exp(pred)
        return pred
