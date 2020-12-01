import os.path as osp
import torch
import os
import pickle
import numpy as np
import gc
import pandas as pd
import tqdm

from ..datasets import MovieLens, Yelp


def get_folder_path(model, dataset, loss_type):
    if dataset[:4] == "Yelp":
        dataset = "Yelp"
    data_folder = osp.join(
        'checkpoint', 'data', dataset)
    weights_folder = osp.join(
        'checkpoint', 'weights', dataset, model, loss_type)
    logger_folder = osp.join(
        'checkpoint', 'loggers', dataset, model, loss_type)
    data_folder = osp.expanduser(osp.normpath(data_folder))
    weights_folder = osp.expanduser(osp.normpath(weights_folder))
    logger_folder = osp.expanduser(osp.normpath(logger_folder))

    return data_folder, weights_folder, logger_folder


def get_opt_class(opt):
    if opt.lower() == 'adam':
        return torch.optim.Adam
    elif opt.lower() == 'sgd':
        return torch.optim.SGD
    elif opt.lower() == 'sparseadam':
        return torch.optim.SparseAdam
    else:
        raise NotImplementedError('No such optims!')


def save_model(file_path, model, optim, epoch, rec_metrics, silent=False):
    model_states = {'model': model.state_dict()}
    optim_states = {'optim': optim.state_dict()}
    states = {
        'epoch': epoch,
        'model_states': model_states,
        'optim_states': optim_states,
        'rec_metrics': rec_metrics
    }

    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("Saved checkpoint_backup '{}'".format(file_path))


def save_kgat_model(file_path, model, optim, epoch, rec_metrics, silent=False):
    model_states = {'model': model.state_dict()}
    optim_states = {'optim': optim.state_dict()}
    states = {
        'epoch': epoch,
        'model_states': model_states,
        'optim_states': optim_states,
        'rec_metrics': rec_metrics
    }

    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("Saved checkpoint_backup '{}'".format(file_path))


def save_random_walk_model(file_path, model, optim, train_loss, silent=False):
    model_states = {'model': model.state_dict()}
    optim_states = {'optim': optim.state_dict()}
    states = {
        'model_states': model_states,
        'optim_states': optim_states,
        'random_walk_train_loss_per_run': train_loss,
    }

    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("Saved checkpoint_backup '{}'".format(file_path))


def load_model(file_path, model, optim, device):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_states']['model'])
        optim.load_state_dict(checkpoint['optim_states']['optim'])
        rec_metrics = checkpoint['rec_metrics']
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Loaded checkpoint_backup '{}'".format(file_path))
    else:
        print("No checkpoint_backup found at '{}'".format(file_path))
        epoch = 0
        rec_metrics = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

    return model, optim, epoch, rec_metrics


def load_kgat_model(file_path, model, optim, device):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_states']['model'])
        optim.load_state_dict(checkpoint['optim_states']['optim'])
        rec_metrics = checkpoint['rec_metrics']
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Loaded checkpoint_backup '{}'".format(file_path))
    else:
        print("No checkpoint_backup found at '{}'".format(file_path))
        epoch = 0
        rec_metrics = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

    return model, optim, epoch, rec_metrics


def save_global_logger(
        global_logger_filepath,
        HR_per_run, NDCG_per_run, AUC_per_run,
        train_loss_per_run, eval_loss_per_run
):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump(
            [HR_per_run, NDCG_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run],
            f
        )


def save_kg_global_logger(
        global_logger_filepath,
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
        kg_train_loss_per_run_np, kg_eval_loss_per_run_np
):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump(
            [
                HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                kg_train_loss_per_run_np, kg_eval_loss_per_run_np
            ],
            f
        )


def save_kgat_global_logger(
        global_logger_filepath,
        HR_per_run, NDCG_per_run, AUC_per_run,
        kg_train_loss_per_run, cf_train_loss_per_run,
        cf_eval_loss_per_run
):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump(
            [
                HR_per_run, NDCG_per_run, AUC_per_run,
                kg_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run
            ],
            f
        )


def save_random_walk_logger(
        global_logger_filepath,
        HR_per_run, NDCG_per_run, AUC_per_run,
        random_walk_train_loss_per_run, train_loss_per_run, eval_loss_per_run
):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump(
            [HR_per_run, NDCG_per_run, AUC_per_run, random_walk_train_loss_per_run, train_loss_per_run, eval_loss_per_run],
            f
        )


def load_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath, 'rb') as f:
            HRs_per_run, NDCGs_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run = pickle.load(f)
    else:
        print("No loggers found at '{}'".format(global_logger_filepath))
        HRs_per_run, NDCGs_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run = \
            np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

    return HRs_per_run, NDCGs_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run, HRs_per_run.shape[0]


def load_random_walk_model(file_path, model, optim, device):
    if os.path.isfile(file_path):
        print('Random walk model loaded!')
        checkpoint = torch.load(file_path, map_location=device)
        model.load_state_dict(checkpoint['model_states']['model'])
        optim.load_state_dict(checkpoint['optim_states']['optim'])
        train_loss = checkpoint['random_walk_train_loss_per_run']
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print('No random walk model found! Create new!')
        train_loss = 0
    return model, optim, train_loss


def load_kg_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath, 'rb') as f:
            HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
            kg_train_loss_per_run_np, kg_eval_loss_per_run_np = pickle.load(f)
    else:
        print("No loggers found at '{}'".format(global_logger_filepath))
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
        kg_train_loss_per_run_np, kg_eval_loss_per_run_np = \
            np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), \
            np.zeros((0, 1))

    return HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
            kg_train_loss_per_run_np, kg_eval_loss_per_run_np, \
           HRs_per_run_np.shape[0]


def load_kgat_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath, 'rb') as f:
            HRs_per_run, NDCGs_per_run, AUC_per_run, \
            kg_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run = pickle.load(f)
    else:
        print("No loggers found at '{}'".format(global_logger_filepath))
        HRs_per_run, NDCGs_per_run, AUC_per_run, \
        kg_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run = \
            np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), \
            np.zeros((0, 1)), np.zeros((0, 1))

    return HRs_per_run, NDCGs_per_run, AUC_per_run, \
           kg_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run, \
           HRs_per_run.shape[0]


def load_random_walk_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath, 'rb') as f:
            HRs_per_run, NDCGs_per_run, AUC_per_run, \
            random_walk_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run = pickle.load(f)
    else:
        print("No loggers found at '{}'".format(global_logger_filepath))
        HRs_per_run, NDCGs_per_run, AUC_per_run, \
        random_walk_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run = \
            np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), \
            np.zeros((0, 1)), np.zeros((0, 1))

    return HRs_per_run, NDCGs_per_run, AUC_per_run, \
           random_walk_train_loss_per_run, cf_train_loss_per_run, cf_eval_loss_per_run, HRs_per_run.shape[0]


def load_dataset(dataset_args):
    if dataset_args['dataset'] == 'Movielens':
        return MovieLens(**dataset_args)
    elif dataset_args['dataset'] == 'Yelp':
        return Yelp(**dataset_args)
    else:
        raise NotImplemented('Dataset not implemented!')


def instantwrite(filename):
    filename.flush()
    os.fsync(filename.fileno())


def clearcache():
    gc.collect()
    torch.cuda.empty_cache()


def update_pea_graph_input(dataset_args, train_args, dataset):
    if dataset_args['dataset'] == "Movielens":
        if dataset_args['name'] == "latest-small":
            user2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(train_args['device'])
            year2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['year2item']).long().to(train_args['device'])
            actor2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['actor2item']).long().to(train_args['device'])
            director2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['director2item']).long().to(train_args['device'])
            writer2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['writer2item']).long().to(train_args['device'])
            genre2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['genre2item']).long().to(train_args['device'])
            tag2item_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['tag2item']).long().to(train_args['device'])
            tag2user_edge_index = \
                torch.from_numpy(dataset.edge_index_nps['tag2user']).long().to(train_args['device'])

            meta_path_edge_indicis_1 = [user2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_2 = [torch.flip(user2item_edge_index, dims=[0]), user2item_edge_index]
            meta_path_edge_indicis_3 = [year2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_4 = [actor2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_5 = [writer2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_6 = [director2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_7 = [genre2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_8 = [tag2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_9 = [tag2user_edge_index, user2item_edge_index]
            meta_path_edge_index_list = [
                meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3,
                meta_path_edge_indicis_4, meta_path_edge_indicis_5, meta_path_edge_indicis_6,
                meta_path_edge_indicis_7, meta_path_edge_indicis_8, meta_path_edge_indicis_9
            ]

        if dataset_args['name'] == "25m":
            user2item_edge_index = torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(
                train_args['device'])
            year2item_edge_index = torch.from_numpy(dataset.edge_index_nps['year2item']).long().to(
                train_args['device'])
            actor2item_edge_index = torch.from_numpy(dataset.edge_index_nps['actor2item']).long().to(
                train_args['device'])
            director2item_edge_index = torch.from_numpy(dataset.edge_index_nps['director2item']).long().to(
                train_args['device'])
            writer2item_edge_index = torch.from_numpy(dataset.edge_index_nps['writer2item']).long().to(
                train_args['device'])
            genre2item_edge_index = torch.from_numpy(dataset.edge_index_nps['genre2item']).long().to(
                train_args['device'])
            genome_tag2item_edge_index = torch.from_numpy(dataset.edge_index_nps['genome_tag2item']).long().to(
                train_args['device'])
            tag2item_edge_index = torch.from_numpy(dataset.edge_index_nps['tag2item']).long().to(
                train_args['device'])
            tag2user_edge_index = torch.from_numpy(dataset.edge_index_nps['tag2user']).long().to(
                train_args['device'])

            meta_path_edge_indicis_1 = [user2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_2 = [year2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_3 = [actor2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_4 = [writer2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_5 = [director2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_6 = [genre2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_7 = [genome_tag2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
            meta_path_edge_indicis_8 = [torch.flip(tag2user_edge_index, dims=[0]), tag2user_edge_index]
            meta_path_edge_indicis_9 = [torch.flip(tag2item_edge_index, dims=[0]), tag2user_edge_index]

            meta_path_edge_indicis_10 = [torch.flip(user2item_edge_index, dims=[0]), user2item_edge_index]
            meta_path_edge_indicis_11 = [tag2user_edge_index, user2item_edge_index]
            meta_path_edge_indicis_12 = [torch.flip(tag2item_edge_index, dims=[0]), tag2item_edge_index]
            meta_path_edge_indicis_13 = [torch.flip(tag2user_edge_index, dims=[0]), tag2item_edge_index]

            meta_path_edge_index_list = [
                meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3,
                meta_path_edge_indicis_4, meta_path_edge_indicis_5, meta_path_edge_indicis_6,
                meta_path_edge_indicis_7, meta_path_edge_indicis_8, meta_path_edge_indicis_9,
                meta_path_edge_indicis_10, meta_path_edge_indicis_11, meta_path_edge_indicis_12,
                meta_path_edge_indicis_13
            ]
    elif dataset_args['dataset'] == "Yelp":
        user2item_edge_index = torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(train_args['device'])
        stars2item_edge_index = torch.from_numpy(dataset.edge_index_nps['stars2item']).long().to(
            train_args['device'])
        reviewcount2item_edge_index = torch.from_numpy(dataset.edge_index_nps['reviewcount2item']).long().to(
            train_args['device'])
        attributes2item_edge_index = torch.from_numpy(dataset.edge_index_nps['attributes2item']).long().to(
            train_args['device'])
        categories2item_edge_index = torch.from_numpy(dataset.edge_index_nps['categories2item']).long().to(
            train_args['device'])
        checkincount2item_edge_index = torch.from_numpy(dataset.edge_index_nps['checkincount2item']).long().to(
            train_args['device'])
        reviewcount2user_edge_index = torch.from_numpy(dataset.edge_index_nps['reviewcount2user']).long().to(
            train_args['device'])
        friendcount2user_edge_index = torch.from_numpy(dataset.edge_index_nps['friendcount2user']).long().to(
            train_args['device'])
        fans2user_edge_index = torch.from_numpy(dataset.edge_index_nps['fans2user']).long().to(train_args['device'])
        stars2user_edge_index = torch.from_numpy(dataset.edge_index_nps['stars2user']).long().to(
            train_args['device'])

        meta_path_edge_indicis_1 = [user2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_2 = [torch.flip(user2item_edge_index, dims=[0]), user2item_edge_index]
        meta_path_edge_indicis_3 = [stars2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_4 = [reviewcount2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_5 = [attributes2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_6 = [categories2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_7 = [checkincount2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_8 = [reviewcount2user_edge_index, user2item_edge_index]
        meta_path_edge_indicis_9 = [friendcount2user_edge_index, user2item_edge_index]
        meta_path_edge_indicis_10 = [fans2user_edge_index, user2item_edge_index]
        meta_path_edge_indicis_11 = [stars2user_edge_index, user2item_edge_index]
        meta_path_edge_index_list = [
            meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3, meta_path_edge_indicis_4,
            meta_path_edge_indicis_5, meta_path_edge_indicis_6, meta_path_edge_indicis_7, meta_path_edge_indicis_8,
            meta_path_edge_indicis_9, meta_path_edge_indicis_10, meta_path_edge_indicis_11
        ]
    else:
        raise NotImplementedError
    return meta_path_edge_index_list


def compute_item_similarity_mat(dataset, metapath):
    print('Compute similiary for metapath {}'.format(metapath))
    assert len(metapath) > 0
    if metapath[0][0] == '-':
        path = np.flip(dataset.edge_index_nps[metapath[0][1:]], axis=0).astype(np.int)
    else:
        path = dataset.edge_index_nps[metapath[0]].astype(np.int)
    path_df = pd.DataFrame(path.T, columns=['0', '1'])
    for step_idx in range(1, len(metapath)):
        if metapath[step_idx][0] == '-':
            path = np.flip(dataset.edge_index_nps[metapath[step_idx][1:]], axis=0).astype(np.int)
        else:
            path = dataset.edge_index_nps[metapath[step_idx]].astype(np.int)
        step_df = pd.DataFrame(path.T, columns=[str(step_idx), str(step_idx + 1)])
        path_df = path_df.merge(step_df, how='inner', on=str(step_idx))
    path_np = path_df.to_numpy()

    num_iids = dataset['num_iids']
    iid_accs = dataset.type_accs['iid']
    S = np.eye(num_iids)
    pbar = tqdm.tqdm(range(num_iids))
    for i in pbar:
        S[i, i] = np.sum(path_np[path_np[:, 0] == (i + iid_accs)][:, -1] == (i + iid_accs))
    pbar = tqdm.tqdm(range(num_iids))
    for i in pbar:
        head = path_np[path_np[:, 0] == (i + iid_accs)]
        for j in range(num_iids):
            if i == j:
                continue
            if S[i, i] + S[j, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = 2 * np.sum(head[:, -1] == (j + iid_accs)) / (S[i, i] + S[j, j])
    return S


def compute_diffused_score_mat(dataset, S):
    print('Compute diffused score mat')
    diffused_score_mat = np.zeros((dataset.num_uids, dataset.num_iids))
    edge_index = dataset.edge_index_nps['user2item'].astype(np.int)
    for edge, rating in zip(edge_index.T, dataset.rating_np):
        diffused_score_mat[edge[0] - dataset.type_accs['uid'], edge[1] - dataset.type_accs['iid']] = rating

    filter = np.zeros((dataset.num_uids, dataset.num_iids))
    filter[np.where(diffused_score_mat != 0)] = 1
    norm = np.copy(S)
    np.fill_diagonal(norm, 0)
    norm = np.matmul(filter, norm)

    norm[np.where(norm == 0)] = 10e-6

    diffused_score_mat[np.where(diffused_score_mat == 0)] = \
        np.matmul(diffused_score_mat, S)[np.where(diffused_score_mat == 0)] / norm[np.where(diffused_score_mat == 0)]
    return diffused_score_mat
