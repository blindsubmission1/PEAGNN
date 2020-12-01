import torch
from torch_geometric.nn import GCNConv

from .base import PEABaseChannel, PEABaseRecsysModel


class PEAGCNChannel(PEABaseChannel):
    def __init__(self, **kwargs):
        super(PEAGCNChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        self.gnn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] == 1:
            self.gnn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['repr_dim']))
        else:
            self.gnn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['hidden_size']))
            for i in range(kwargs['num_steps'] - 2):
                self.gnn_layers.append(GCNConv(kwargs['hidden_size'], kwargs['hidden_size']))
            self.gnn_layers.append(GCNConv(kwargs['hidden_size'], kwargs['repr_dim']))

        self.reset_parameters()


class PEAGCNRecsysModel(PEABaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEAGCNChannel
        super(PEAGCNRecsysModel, self).__init__(**kwargs)
