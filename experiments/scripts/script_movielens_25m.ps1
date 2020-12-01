# Movielens25m

# NFM
python3 nfm_solver_bce.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --emb_dim=64 --hidden_size=64 --dropout=0.3 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# CFKG
python3 cfkg_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --emb_dim=64 --init_eval=false --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# HeRec
python3 herec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --emb_dim=64 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# Metapath2Vec
python3 metapath2vec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --emb_dim=64 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


#NGCF
python3 ngcf_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.2 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# KGCN
python3 kgcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.1 --emb_dim=64 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# KGAT
python3 kgat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.1 --emb_dim=64 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEAGCN
# --entity_aware=false
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=false

# --entity_aware=true
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=true


# PEAGAT
# --entity_aware=false
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=false

# --entity_aware=true
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=true


# PEASage
# --entity_aware=false
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=false

# --entity_aware=true
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=true --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26 --metapath_test=true




