import argparse
import torch
import os
import numpy as np
import random as rd
import time
import tqdm
import sys

sys.path.append('..')
from torch.utils.data import DataLoader
from graph_recsys_benchmark.models import CFKGRecsysModel
from graph_recsys_benchmark.solvers import BaseSolver
from graph_recsys_benchmark.utils import *

MODEL_TYPE = 'Graph'
GRAPH_TYPE = 'hete'
KG_LOSS_TYPE = 'COS'
MODEL = 'CFKG'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')        # Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')          # 25m, latest-small
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='unseen', help='')     # unseen(for latest-small), random(for Yelp,25m)

# Model params
parser.add_argument('--emb_dim', type=int, default=64, help='')

# Train params
parser.add_argument('--init_eval', type=str, default='false', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=5, help='')             #5(for MovieLens), 3(for Yelp)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for MovieLens), 20(for Yelp)
parser.add_argument('--batch_size', type=int, default=1024, help='')        #1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='5,10,15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=26, help='')        #26(for MovieLens), 16(only for Yelp)


args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=KG_LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': False, 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'cf_loss_type': None, 'type': GRAPH_TYPE,
    'entity_aware': False, 'sampling_strategy': args.sampling_strategy
}
model_args = {'model_type': MODEL_TYPE, 'emb_dim': args.emb_dim}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': 0,
    'device': device,
    'lr': args.lr,
    'num_workers': args.num_workers,
    'weights_folder': os.path.join(weights_folder, str(model_args.copy())[:255]),
    'logger_folder': os.path.join(logger_folder, str(model_args.copy())[:255]),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


class MCFKGRecsysModel(CFKGRecsysModel):
    def kg_loss(self, batch):
        h = self.x[batch[:, 0]]
        pos_t = self.x[batch[:, 1]]
        neg_t = self.x[batch[:, 2]]

        r = self.r[batch[:, 3]]
        pos_sim = torch.sum((h + r) * pos_t, dim=-1)
        neg_sim = torch.sum((h + r) * neg_t, dim=-1)

        loss = - (pos_sim.sigmoid().log().sum() + (-neg_sim).sigmoid().log().sum())

        return loss


class MCFKGRecsysModelSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(MCFKGRecsysModelSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def metrics(
            self,
            run,
            epoch,
            model,
            dataset
    ):
        """
        Generate the positive and negative candidates for the recsys evaluation
        :param run:
        :param epoch:
        :param model:
        :param dataset:
        :return: a tuple (pos_i_nids, neg_i_nids), two entries should be both list
        """
        HRs, NDCGs, AUC = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1))

        test_pos_unid_inid_map, neg_unid_inid_map = \
            dataset.test_pos_unid_inid_map, dataset.neg_unid_inid_map

        u_nids = list(test_pos_unid_inid_map.keys())
        test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
        for u_idx, u_nid in enumerate(test_bar):
            pos_i_nids, neg_i_nids = self.generate_candidates(
                dataset, u_nid
            )
            if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
                raise ValueError("No pos or neg samples found in evaluation!")

            pos_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(pos_i_nids))])).to(
                self.train_args['device'])
            pos_i_nids_t = torch.from_numpy(np.array(pos_i_nids)).to(self.train_args['device'])
            neg_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(neg_i_nids))])).to(
                self.train_args['device'])
            neg_i_nids_t = torch.from_numpy(np.array(neg_i_nids)).to(self.train_args['device'])

            pos_pred = model.predict(pos_u_nids_t, pos_i_nids_t).reshape(-1)
            neg_pred = model.predict(neg_u_nids_t, neg_i_nids_t).reshape(-1)

            _, indices = torch.sort(torch.cat([pos_pred, neg_pred]), descending=True)
            hit_vec = (indices < len(pos_i_nids)).cpu().detach().numpy()
            pos_pred = pos_pred.cpu().detach().numpy()
            neg_pred = neg_pred.cpu().detach().numpy()

            HRs = np.vstack([HRs, hit(hit_vec)])
            NDCGs = np.vstack([NDCGs, ndcg(hit_vec)])
            AUC = np.vstack([AUC, auc(pos_pred, neg_pred)])
            test_bar.set_description(
                'Run {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}'.format(
                    run, epoch, HRs.mean(axis=0)[0], HRs.mean(axis=0)[5], HRs.mean(axis=0)[10], HRs.mean(axis=0)[15],
                    NDCGs.mean(axis=0)[0], NDCGs.mean(axis=0)[5], NDCGs.mean(axis=0)[10], NDCGs.mean(axis=0)[15],
                    AUC.mean(axis=0)[0])
            )
        return np.mean(HRs, axis=0), np.mean(NDCGs, axis=0), np.mean(AUC, axis=0)

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
        kg_train_loss_per_run_np, kg_eval_loss_per_run_np, last_run = \
            load_kg_global_logger(global_logger_file_path)

        # Create the dataset
        dataset = load_dataset(self.dataset_args)

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'a') as logger_file:
            start_run = last_run + 1
            if start_run <= self.train_args['runs']:
                for run in range(start_run, self.train_args['runs'] + 1):
                    # Fix the random seed
                    seed = 2019 + run
                    rd.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    # Create model and optimizer
                    self.model_args['num_nodes'] = dataset.num_nodes
                    self.model_args['dataset'] = dataset

                    model = self.model_class(**self.model_args).to(self.train_args['device'])

                    opt_class = get_opt_class(self.train_args['opt'])
                    optimizer = opt_class(
                        params=model.parameters(),
                        lr=self.train_args['lr'],
                        weight_decay=self.train_args['weight_decay']
                    )

                    # Load models
                    weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)

                    weights_file = os.path.join(weights_path, 'latest.pkl')
                    model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                           self.train_args['device'])

                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, \
                    kg_train_loss_per_epoch_np, kg_eval_loss_per_epoch_np = rec_metrics

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.train_args['device'])

                    t_start = time.perf_counter()
                    # Train KG embedding
                    for epoch in range(1, self.train_args['epochs'] + 1):
                        # Train KG part
                        model.train()
                        kg_loss_per_batch = []
                        dataset.kg_negative_sampling()
                        dataloader = DataLoader(
                            dataset,
                            shuffle=True,
                            batch_size=self.train_args['batch_size'],
                            num_workers=self.train_args['num_workers']
                        )
                        train_bar = tqdm.tqdm(dataloader, total=len(dataloader))
                        for _, batch in enumerate(train_bar):
                            batch = batch.to(self.train_args['device'])

                            optimizer.zero_grad()
                            loss = model.kg_loss(batch)
                            loss.backward()
                            optimizer.step()

                            kg_loss_per_batch.append(loss.detach().cpu().item())
                            kg_train_loss = np.mean(kg_loss_per_batch)
                            train_bar.set_description(
                                'Run: {}, epoch: {}, kg train loss: {:.4f}'.format(run, epoch, kg_train_loss)
                            )

                        # Eval CF part
                        model.eval()
                        with torch.no_grad():
                            HRs, NDCGs, AUC = self.metrics(run, epoch, model, dataset)

                        # Sumarize the epoch
                        HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                        NDCGs_per_epoch_np = np.vstack([NDCGs_per_epoch_np, NDCGs])
                        AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])

                        kg_train_loss_per_epoch_np = np.vstack([kg_train_loss_per_epoch_np, np.array([kg_train_loss])])
                        kg_eval_loss_per_epoch_np = np.vstack([kg_eval_loss_per_epoch_np, np.array([0])])

                        print(
                            'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}\n'.format(
                                run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                AUC[0]
                            )
                        )
                        logger_file.write(
                            'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}\n'.format(
                                run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                AUC[0]
                            )
                        )
                        instantwrite(logger_file)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.train_args['device'])
                    t_end = time.perf_counter()

                    HRs_per_run_np =  np.vstack([HRs_per_run_np, np.max(HRs_per_epoch_np, axis=0)])
                    NDCGs_per_run_np =  np.vstack([NDCGs_per_run_np, np.max(NDCGs_per_epoch_np, axis=0)])
                    AUC_per_run_np = np.vstack([AUC_per_run_np, np.max(AUC_per_epoch_np, axis=0)])
                    kg_train_loss_per_run_np = np.vstack([kg_train_loss_per_run_np, np.mean(kg_train_loss_per_epoch_np, axis=0)])
                    kg_eval_loss_per_run_np = np.vstack([kg_eval_loss_per_run_np, np.mean(kg_eval_loss_per_epoch_np, axis=0)])

                    weightpath = os.path.join(weights_path, 'latest.pkl')
                    save_kgat_model(
                        weightpath,
                        model, optimizer, epoch,
                        rec_metrics=(
                            HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np,
                            kg_train_loss_per_epoch_np, kg_eval_loss_per_epoch_np
                        )
                    )

                    save_kg_global_logger(
                        global_logger_file_path,
                        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                        kg_train_loss_per_run_np, kg_eval_loss_per_run_np
                    )
                    print(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f},'
                        'kg_train_loss: {:.4f}, kg_eval_loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0],
                            np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10],
                            np.max(NDCGs_per_epoch_np, axis=0)[15], np.max(AUC_per_epoch_np, axis=0)[0],
                            kg_train_loss_per_epoch_np[-1][0], kg_eval_loss_per_epoch_np[-1][0])
                    )
                    logger_file.write(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f},'
                        'kg_train_loss: {:.4f}, kg_eval_loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0],
                            np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10], np.max(NDCGs_per_epoch_np, axis=0)[15],
                            np.max(AUC_per_epoch_np, axis=0)[0], kg_train_loss_per_epoch_np[-1][0], kg_eval_loss_per_epoch_np[-1][0])
                    )
                    instantwrite(logger_file)

                    del model, optimizer, loss, kg_loss_per_batch

                print(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f},'
                    'kg_train_loss: {:.4f}, kg_eval_loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                        NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                        NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                        kg_train_loss_per_run_np.mean(axis=0)[0], kg_eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                logger_file.write(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f},'
                    'kg_train_loss: {:.4f}, kg_eval_loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                        NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                        NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                        kg_train_loss_per_run_np.mean(axis=0)[0], kg_eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                instantwrite(logger_file)


if __name__ == '__main__':
    solver = MCFKGRecsysModelSolver(MCFKGRecsysModel, dataset_args, model_args, train_args)
    solver.run()
