import argparse
import torch
import os
import numpy as np
import random as rd
import tqdm
import time
import inspect
import sys

sys.path.append('..')
from graph_recsys_benchmark.models import MetaPath2Vec
from torch.utils.data import DataLoader

from graph_recsys_benchmark.models import WalkBasedRecsysModel
from graph_recsys_benchmark.utils import *
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'Walk'
LOSS_TYPE = 'BPR'
MODEL = 'MetaPath2Vec'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')		#Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')	#25m, latest-small
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='unseen', help='')		#unseen(for latest-small), random(for Yelp,25m)
parser.add_argument('--entity_aware', type=str, default='false', help='')

# Model params
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--walks_per_node', type=int, default=1000, help='')
parser.add_argument('--walk_length', type=int, default=100, help='')
parser.add_argument('--context_size', type=int, default=7, help='')
parser.add_argument('--random_walk_num_negative_samples', type=int, default=5, help='')
parser.add_argument('--sparse', type=str, default='true', help='')

# Train params
parser.add_argument('--init_eval', type=str, default='true', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=5, help='')         #5(for MovieLens), 3(for Yelp)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for MovieLens), 20(only for Yelp)
parser.add_argument('--random_walk_batch_size', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=1024, help='')		#1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--random_walk_opt', type=str, default='SparseAdam', help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--random_walk_lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='5,10,15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=26, help='')        #26(for others), 16(only for Yelp)

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': args.if_use_features.lower() == 'true', 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'cf_loss_type': LOSS_TYPE, 'type': GRAPH_TYPE,
    'sampling_strategy': args.sampling_strategy, 'entity_aware': args.entity_aware.lower() == 'true'
}
model_args = {
    'embedding_dim': args.emb_dim, 'model_type': MODEL_TYPE,
    'walk_length': args.walk_length, 'context_size': args.context_size,
    'walks_per_node': args.walks_per_node, 'num_negative_samples': args.random_walk_num_negative_samples,
    'sparse': args.sparse.lower() == 'true'
}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'random_walk_opt': args.random_walk_opt, 'opt': args.opt,
    'runs': args.runs, 'epochs': args.epochs,
    'batch_size': args.batch_size, 'random_walk_batch_size': args.random_walk_batch_size,
    'num_workers': args.num_workers,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device, 'random_walk_lr': args.random_walk_lr,
    'weights_folder': os.path.join(weights_folder, str(model_args)[:255]),
    'logger_folder': os.path.join(logger_folder, str(model_args)[:255]),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


class MetaPath2VecRecsysModel(WalkBasedRecsysModel):
    def loss(self, batch):
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return cf_loss


class MetaPath2VecSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(MetaPath2VecSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
        random_walk_train_loss_per_run_np, train_loss_per_run_np, \
        eval_loss_per_run_np, last_run = \
            load_random_walk_global_logger(global_logger_file_path)

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

                    # Create random walk model
                    if self.dataset_args['dataset'] == "Movielens":
                        edge_index_dict = {
                            ('genre', 'as the genre of', 'iid'): torch.from_numpy(dataset.edge_index_nps['genre2item']).long().to(self.train_args['device']),
                            ('iid', 'has been watched by', 'uid'): torch.from_numpy(np.flip(dataset.edge_index_nps['user2item'], 0).copy()).long().to(self.train_args['device']),
                            ('uid', 'has watched', 'iid'): torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(self.train_args['device']),
                            ('iid', 'has the genre', 'genre'): torch.from_numpy(np.flip(dataset.edge_index_nps['genre2item'], 0).copy()).long().to(self.train_args['device']),
                        }
                        metapath = [
                            ('uid', 'has watched', 'iid'),
                            ('iid', 'has the genre', 'genre'),
                            ('genre', 'as the genre of', 'iid'),
                            ('iid', 'has been watched by', 'uid'),
                        ]

                    elif self.dataset_args['dataset'] == "Yelp":
                        edge_index_dict = {
                            ('item_reviewcount', 'as the number of reviews of', 'iid'): torch.from_numpy(dataset.edge_index_nps['reviewcount2item']).long().to(self.train_args['device']),
                            ('iid', 'has been visited by', 'uid'): torch.from_numpy(np.flip(dataset.edge_index_nps['user2item'], 0).copy()).long().to(self.train_args['device']),
                            ('uid', 'has the number of friends', 'user_friendcount'): torch.from_numpy(np.flip(dataset.edge_index_nps['friendcount2user'], 0).copy()).long().to(self.train_args['device']),
                            ('user_friendcount', 'as the number of friends of', 'uid'): torch.from_numpy(dataset.edge_index_nps['friendcount2user']).long().to(self.train_args['device']),
                            ('uid', 'has visited', 'iid'): torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(self.train_args['device']),
                            ('iid', 'has the number of reviews', 'item_reviewcount'): torch.from_numpy(np.flip(dataset.edge_index_nps['reviewcount2item'], 0).copy()).long().to(self.train_args['device']),

                        }
                        metapath = [
                            ('item_reviewcount', 'as the number of reviews of', 'iid'),
                            ('iid', 'has been visited by', 'uid'),
                            ('uid', 'has the number of friends', 'user_friendcount'),
                            ('user_friendcount', 'as the number of friends of', 'uid'),
                            ('uid', 'has visited', 'iid'),
                            ('iid', 'has the number of reviews', 'item_reviewcount')
                        ]
                    self.model_args['metapath'] = metapath
                    self.model_args['edge_index_dict'] = edge_index_dict

                    random_walk_model_args = {k: v for k, v in self.model_args.items() if k in inspect.signature(MetaPath2Vec.__init__).parameters}
                    random_walk_model_args['types'] = dataset.types
                    random_walk_model_args['num_nodes_dict'] = dataset.num_nodes_dict
                    random_walk_model_args['type_accs'] = dataset.type_accs

                    random_walk_model = MetaPath2Vec(**random_walk_model_args).to(self.train_args['device'])
                    opt_class = get_opt_class(self.train_args['random_walk_opt'])
                    random_walk_optimizer = opt_class(
                        params=random_walk_model.parameters(),
                        lr=self.train_args['random_walk_lr'],
                    )
                    loader = random_walk_model.loader(batch_size=self.train_args['random_walk_batch_size'], shuffle=True, num_workers=0)

                    # Load the random walk model
                    weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)
                    weights_file = os.path.join(weights_path, 'random_walk_{}.pkl'.format(self.model_args['walks_per_node']))
                    if os.path.isfile(weights_file):
                        # Load random walk model
                        random_walk_model, random_walk_optimizer, random_walk_train_loss_per_run = \
                            load_random_walk_model(weights_file, random_walk_model, random_walk_optimizer, self.train_args['device'])
                        print("Loaded random walk model checkpoint_backup '{}'".format(weights_file))
                    else:
                        print("Train new random walk model, since no random walk model checkpoint_backup found at '{}'".format(weights_file))
                        # Train random walk model
                        random_walk_model.train()
                        pbar = tqdm.tqdm(loader, total=len(loader))
                        random_walk_loss = 0
                        for random_walk_batch_idx, (pos_rw, neg_rw) in enumerate(pbar):
                            random_walk_optimizer.zero_grad()
                            loss = random_walk_model.loss(pos_rw.to(self.train_args['device']), neg_rw.to(self.train_args['device']))
                            random_walk_loss += loss.detach().cpu().item()
                            loss.backward()
                            random_walk_optimizer.step()
                            pbar.set_description('Random walk loss {:.4f}'.format(random_walk_loss / (random_walk_batch_idx + 1)))
                        print('walk loss: {:.4f}'.format(random_walk_loss / len(loader)))
                        random_walk_train_loss_per_run = random_walk_loss / len(loader)

                        weightpath = os.path.join(weights_path, 'random_walk_{}.pkl'.format(self.model_args['walks_per_node']))
                        save_random_walk_model(weightpath, random_walk_model, random_walk_optimizer, random_walk_train_loss_per_run)

                    # Init the RecSys model
                    with torch.no_grad():
                        random_walk_model.eval()
                        self.model_args['embedding'] = torch.tensor(random_walk_model.embedding.weight, requires_grad=False)
                    # Do cleaning
                    del self.model_args['metapath']
                    del self.model_args['edge_index_dict']
                    del random_walk_model
                    # torch.cuda.empty_cache()
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
                    model, optimizer, last_epoch, rec_metrics = \
                        load_model(weights_file, model, optimizer, self.train_args['device'])
                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, \
                    train_loss_per_epoch_np, eval_loss_per_epoch_np = rec_metrics

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start_epoch = last_epoch + 1
                    if start_epoch == 1 and self.train_args['init_eval']:
                        model.eval()
                        HRs_before_np, NDCGs_before_np, AUC_before_np, eval_loss_before_np = \
                            self.metrics(run, 0, model, dataset)
                        print(
                            'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                                NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                                AUC_before_np[0], eval_loss_before_np[0]
                            )
                        )
                        logger_file.write(
                            'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '
                            'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                                NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                                AUC_before_np[0], eval_loss_before_np[0]
                            )
                        )
                        instantwrite(logger_file)
                        # clearcache()

                    t_start = time.perf_counter()
                    if start_epoch <= self.train_args['epochs']:
                        # Start training model
                        for epoch in range(start_epoch, self.train_args['epochs'] + 1):
                            loss_per_batch = []
                            model.train()
                            dataset.cf_negative_sampling()
                            train_dataloader = DataLoader(
                                dataset,
                                shuffle=True,
                                batch_size=self.train_args['batch_size'],
                                num_workers=self.train_args['num_workers']
                            )
                            train_bar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))

                            for _, batch in enumerate(train_bar):
                                if self.model_args['model_type'] == 'MF':
                                    if self.model_args['loss_type'] == 'BCE':
                                        batch[:, 0] -= dataset.e2nid_dict['uid'][0]
                                        batch[:, 1] -= dataset.e2nid_dict['iid'][0]
                                    elif self.model_args['loss_type'] == 'BPR':
                                        batch[:, 0] -= dataset.e2nid_dict['uid'][0]
                                        batch[:, 1:] -= dataset.e2nid_dict['iid'][0]
                                batch = batch.to(self.train_args['device'])

                                optimizer.zero_grad()
                                loss = model.loss(batch)
                                loss.backward()
                                optimizer.step()

                                loss_per_batch.append(loss.detach().cpu().item())
                                train_loss = np.mean(loss_per_batch)
                                train_bar.set_description(
                                    'Run: {}, epoch: {}, train loss: {:.4f}'.format(run, epoch, train_loss)
                                )

                            model.eval()
                            HRs, NDCGs, AUC, eval_loss = self.metrics(run, epoch, model, dataset)
                            HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                            NDCGs_per_epoch_np = np.vstack([NDCGs_per_epoch_np, NDCGs])
                            AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])
                            train_loss_per_epoch_np = np.vstack([train_loss_per_epoch_np, np.array([train_loss])])
                            eval_loss_per_epoch_np = np.vstack([eval_loss_per_epoch_np, np.array([eval_loss])])

                            if epoch in self.train_args['save_epochs']:
                                weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                        HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np,
                                        eval_loss_per_epoch_np)
                                )
                            if epoch > self.train_args['save_every_epoch']:
                                weightpath = os.path.join(weights_path, 'latest.pkl')
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                        HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np,
                                        eval_loss_per_epoch_np)
                                )
                            print(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                    AUC[0], train_loss, eval_loss[0]
                                )
                            )
                            logger_file.write(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                    AUC[0], train_loss, eval_loss[0]
                                )
                            )
                            instantwrite(logger_file)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_end = time.perf_counter()

                        HRs_per_run_np = np.vstack([HRs_per_run_np, np.max(HRs_per_epoch_np, axis=0)])
                        NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, np.max(NDCGs_per_epoch_np, axis=0)])
                        AUC_per_run_np = np.vstack([AUC_per_run_np, np.max(AUC_per_epoch_np, axis=0)])
                        random_walk_train_loss_per_run_np = np.vstack([random_walk_train_loss_per_run_np, random_walk_train_loss_per_run])
                        train_loss_per_run_np = np.vstack(
                            [train_loss_per_run_np, np.mean(train_loss_per_epoch_np, axis=0)])
                        eval_loss_per_run_np = np.vstack(
                            [eval_loss_per_run_np, np.mean(eval_loss_per_epoch_np, axis=0)])

                        save_random_walk_logger(
                            global_logger_file_path,
                            HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                            random_walk_train_loss_per_run_np, train_loss_per_run_np, eval_loss_per_run_np
                        )
                        print(
                            'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            'walk loss: {:.4f}, train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0], np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5], np.max(NDCGs_per_epoch_np, axis=0)[10],
                            np.max(NDCGs_per_epoch_np, axis=0)[15], np.max(AUC_per_epoch_np, axis=0)[0], random_walk_train_loss_per_run_np[-1][0],
                            train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                        )
                        logger_file.write(
                            'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            'walk loss: {:.4f}, train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                                run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0], np.max(HRs_per_epoch_np, axis=0)[5],
                                np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                                np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                                np.max(NDCGs_per_epoch_np, axis=0)[10], np.max(NDCGs_per_epoch_np, axis=0)[15],
                                np.max(AUC_per_epoch_np, axis=0)[0], random_walk_train_loss_per_run_np[-1][0],
                                train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                        )
                        instantwrite(logger_file)

                        del model, optimizer, loss, loss_per_batch, rec_metrics, train_dataloader

                print(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, walk loss: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0], NDCGs_per_run_np.mean(axis=0)[5],
                        NDCGs_per_run_np.mean(axis=0)[10], NDCGs_per_run_np.mean(axis=0)[15],
                        AUC_per_run_np.mean(axis=0)[0], random_walk_train_loss_per_run_np.mean(axis=0)[0],
                        train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                logger_file.write(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, walk loss: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0], NDCGs_per_run_np.mean(axis=0)[5],
                        NDCGs_per_run_np.mean(axis=0)[10], NDCGs_per_run_np.mean(axis=0)[15],
                        AUC_per_run_np.mean(axis=0)[0], random_walk_train_loss_per_run_np.mean(axis=0)[0],
                        train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                instantwrite(logger_file)


if __name__ == '__main__':
    solver = MetaPath2VecSolver(MetaPath2VecRecsysModel, dataset_args, model_args, train_args)
    solver.run()
