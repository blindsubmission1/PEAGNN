import argparse
import torch
import os
import sys

sys.path.append('..')
from graph_recsys_benchmark.models import NFMRecsysModel
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'MF'
LOSS_TYPE = 'BCE'
MODEL = 'NFM'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')  # Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')  # 25m, latest-small
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='random', help='')  # unseen(for latest-small), random(for Yelp,25m)
parser.add_argument('--entity_aware', type=str, default='false', help='')

# Model params
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='')

# Train params
parser.add_argument('--init_eval', type=str, default='true', help='')
parser.add_argument('--num_negative_samples', type=int, default=2, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=5, help='')              #5(for MovieLens), 3(for Yelp)
parser.add_argument('--epochs', type=int, default=50, help='')          #30(for MovieLens), 20(only for Yelp)
parser.add_argument('--batch_size', type=int, default=1024, help='')    #1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='5,10,15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=26, help='')        #26(for MovieLens), 16(only for Yelp)

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
    'sampling_strategy': args.sampling_strategy, 'entity_aware': args.entity_aware.lower() == 'true',
    'model': MODEL
}
model_args = {
    'model_type': MODEL_TYPE, 'dropout': args.dropout, 'hidden_size': args.hidden_size,
    'emb_dim': args.emb_dim, 'if_use_features': args.if_use_features.lower() == 'true',
    'loss_type': LOSS_TYPE
}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'pretrain': False,
    'opt': args.opt,
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,  'device': device,
    'lr': args.lr,
    'num_workers': args.num_workers,
    'weights_folder': os.path.join(weights_folder, str(model_args)[:255]),
    'logger_folder': os.path.join(logger_folder, str(model_args)[:255]),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


if __name__ == '__main__':
    solver = BaseSolver(NFMRecsysModel, dataset_args, model_args, train_args)
    solver.run()
