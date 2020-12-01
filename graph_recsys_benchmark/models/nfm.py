import torch
from .base import MFRecsysModel
from torchfm.model.nfm import NeuralFactorizationMachineModel


class NFMRecsysModel(MFRecsysModel):
    def __init__(self, **kwargs):
        super(NFMRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.emb = NeuralFactorizationMachineModel([kwargs['num_users'], kwargs['num_items']], kwargs['emb_dim'], (kwargs['hidden_size'],), [kwargs['dropout'], kwargs['dropout']])

    def reset_parameters(self):
        pass

    def forward(self, uid, iid):
        user_item_pair_t = torch.cat([uid.view(-1, 1), iid.view(-1, 1)], dim=1)
        rating = self.emb(user_item_pair_t)
        return rating

    def loss(self, pos_neg_pair_t):
        loss_func = torch.nn.MSELoss()
        if self.training:
            pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
            label = pos_neg_pair_t[:, -1].float()
        else:
            pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])[:1]
            neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).float()

        loss = loss_func(pred, label)
        return loss
