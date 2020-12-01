from .base import BaseRecsysModel
import torch
from torch_geometric.nn.inits import glorot


class HeRecRecsysModel(BaseRecsysModel):
    def __init__(self, **kwargs):
        super(HeRecRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.rk_embeddings = kwargs['embeddings']
        self.num_uids = kwargs['num_uids']
        self.num_iids = kwargs['num_iids']
        self.acc_uids = kwargs['acc_uids']
        self.acc_iids = kwargs['acc_iids']

        self.emb_trans = torch.nn.ModuleList([torch.nn.Linear(kwargs['embedding_dim'], kwargs['embedding_dim']) for _ in range(len(self.rk_embeddings))])

        self.user_emb = torch.nn.Embedding(self.num_uids, kwargs['embedding_dim'])
        self.item_emb = torch.nn.Embedding(self.num_iids, kwargs['embedding_dim'])

        self.user_rk_bias = torch.nn.Embedding(self.num_uids, kwargs['embedding_dim'])
        self.item_rk_bias = torch.nn.Embedding(self.num_iids, kwargs['embedding_dim'])

    def eval(self):
        return self.train(False)

    def reset_parameters(self):
        for _ in self.emb_trans:
            glorot(_.weight)
        glorot(self.user_emb.weight)
        glorot(self.item_emb.weight)
        glorot(self.user_rk_bias.weight)
        glorot(self.item_rk_bias.weight)

    def forward(self, unids, inids):
        pred = torch.sum(self.user_emb(unids - self.acc_uids) * self.item_emb(inids - self.acc_iids), dim=-1)
        rk_user_embs = [rk_embedding[unids] for rk_embedding in self.rk_embeddings]
        rk_user_embs = [torch.sigmoid(emb_tran(rk_user_emb)).unsqueeze(-1) for emb_tran, rk_user_emb in zip(self.emb_trans, rk_user_embs)]
        rk_user_embs = torch.sum(torch.cat(rk_user_embs, dim=-1), dim=-1)
        rk_item_embs = [rk_embedding[inids] for rk_embedding in self.rk_embeddings]
        rk_item_embs = [torch.sigmoid(emb_tran(rk_item_emb)).unsqueeze(-1) for emb_tran, rk_item_emb in zip(self.emb_trans, rk_item_embs)]
        rk_item_embs = torch.sum(torch.cat(rk_item_embs, dim=-1), dim=-1)
        pred += torch.sum(rk_user_embs * self.user_rk_bias(unids - self.acc_uids), dim=-1)
        pred += torch.sum(rk_item_embs * self.item_rk_bias(inids - self.acc_iids), dim=-1)
        return pred

    def predict(self, unids, inids):
        return self.forward(unids, inids)
