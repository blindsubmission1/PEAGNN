from .base import GraphRecsysModel
import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class WalkBasedRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(WalkBasedRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.cached_repr = kwargs['embedding']

        self.fc1 = torch.nn.Linear(2 * kwargs['embedding_dim'], kwargs['embedding_dim'])
        self.fc2 = torch.nn.Linear(kwargs['embedding_dim'], 1)

    def eval(self):
        return self.train(False)

    def reset_parameters(self):
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)

    def forward(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, unids, inids):
        return self.forward(unids, inids)
