import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
from networks.gol_adaptive_sigma import GOL_adaptiveSigma
from data.datasets import OL_basic_train, basic
import torch
import random
random_seed = 10
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

def uniform_sample_nodes(labels, keep_ratio):
    n_sample_total = len(labels)*keep_ratio
    n_rank = len(np.unique(labels))
    n_per_rank = int(n_sample_total/n_rank)
    uniq_ranks = np.unique(labels)
    idxs = []
    print(f'sampling node with {keep_ratio} for each rank. (#Per_rank :{n_per_rank})')

    for r in uniq_ranks:
        r_idxs = np.argwhere(labels == r).flatten()
        if len(r_idxs) > n_per_rank:
            selected = np.random.choice(r_idxs, n_per_rank, replace=False)
        else:
            selected = r_idxs
            # selected = np.random.choice(r_idxs, n_per_rank, replace=True)
        idxs.append(selected)
    idxs = np.concatenate(idxs)
    return idxs

def get_datasets(cfg):
    tr_std = None
    te_std = None
    if cfg.dataset =='morph':
        if cfg.noised:
            img_root = cfg.img_root
            
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['filename']]
            tr_ages = tr_list['age'].to_numpy()
            tr_noised_ages = tr_list['age_noised'].to_numpy()

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['filename']]
            te_ages = te_list['age'].to_numpy()
            te_noised_ages = te_list['age_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages

    elif cfg.dataset =='clap':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        
            tr_noised_ages = np.array(tr_list['age_noised'])
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            ## tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/clap/setting_tau3/noise20%/WARMUP_V2_CentroidFixed_noisedTrue20%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau3_Feval_on_test_GOL_vgg16v2norm_2024-09-30 12:34:26/updated.csv').iloc[:,-1])
            tr_ages = np.array(tr_list['age'])
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_noised_ages = np.array(te_list['age_noised'])
            te_ages = np.array(te_list['age'])
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in range(len(te_list))]

            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]
        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = None
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in
                       range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in
                       range(len(te_list))]
            te_ages = te_list[:, cfg.lb_idx]
            te_nosied_ages = None
            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]

    elif cfg.dataset =='aadb':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file)
            # tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, names=['img_dir', 'score'])
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['img_dir']]
            tr_ages = tr_list['score'].to_numpy()
            tr_noised_ages = tr_list['score_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/aadb/setting_tau5/noise40%/V2_CentroidFixed_noisedTrue40%_SOL_DiscrTrueL1_pairwise_lr5e-05_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau5_F0_GOL_vgg16v2norm_2024-10-01 19:42:17/updated.csv').iloc[:,-1])


            te_list = pd.read_csv(cfg.test_file)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['img_dir']]
            te_ages = te_list['score'].to_numpy()
            te_noised_ages = te_list['score_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, header=None)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter, header=None)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages
    elif cfg.dataset == 'rsna' or cfg.dataset =='rsna_toy':
        if cfg.noised:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_list['boneage_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/rsna_revised/setting_tau3/noise15%/CentroidFixed_SOL_DiscrTrueL1_StochasticOrderTrue_pairwise_lr5e-05_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2025-01-18 18:37:10/updated.csv').iloc[:,-1])
            
            # tr_noised_ages = np.array(pd.read_csv('/home/cwlee/ICLR2026/SOL/rsna_Skewed5/setting_tau3/noise15%/WARMUP_V2_CentroidFixed_noisedTrue15%_SOL_DiscrTrueL1_pairwise_lr5e-05_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F_GOL_vgg16v2norm_2025-11-13 17:41:06/updated.csv').iloc[:,-1])

            # te_img_root = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_img_root2 = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_list1 = pd.read_csv(cfg.test_file)
            # te_imgs =
            # te_ages = te_list1['Bone Age (months)'].to_numpy()
            # te_noised_ges = None

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel('/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages = te_list['Ground truth bone age (months)'].to_numpy()
        else:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_ages

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel(
                '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages =te_ages

    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

            if cfg.noised:
                tr_noised_ages = data['age_noised']
            else:
                # tr_noised_ages = None
                tr_noised_ages = tr_ages

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']

            if cfg.noised:
                te_noised_ages = data['age_noised']
            else:
                # te_noised_ages = None
                te_noised_ages = te_ages


    # sampled_idxs = uniform_sample_nodes(tr_noised_ages, keep_ratio =cfg.sampling_ratio)
    # cfg.sampled_idxs = sampled_idxs
    if cfg.noised:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        # loader_dict['train_for_val_sampled'] = DataLoader(
        #     basic.Basic_Noised_Stochastic(np.array(tr_imgs)[sampled_idxs], tr_noised_ages[sampled_idxs], tr_ages[sampled_idxs], transform=cfg.transform_te,
        #                                   is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
        #                                   n_ranks=n_ranks),
        #     batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        #     num_workers=cfg.num_workers)
        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)


    else:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)


        # loader_dict = dict()
        # n_ranks = len(np.unique(tr_noised_ages))
        # loader_dict['train_for_val'] = DataLoader(
        #     basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
        #                                   is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
        #                                   n_ranks=n_ranks),
        #     batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        #     num_workers=cfg.num_workers)
        
        # loader_dict['trainval_vis'] = DataLoader(
        #     basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
        #                                   is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
        #     batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        #     num_workers=cfg.num_workers)


        # loader_dict['val'] = DataLoader(
        #     basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
        #                                   is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
        #                                   n_ranks=n_ranks),
        #     batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

        # # loader_dict['train'] = DataLoader(OL_basic_train.OLBasic_Train(tr_imgs, tr_ages, cfg.transform_tr, cfg.tau, logscale=cfg.logscale, is_filelist=cfg.is_filelist, prob_std=cfg.prob_std),
        # #                                   batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        # # loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_te, is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
        # #                                 batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        # #                                 num_workers=cfg.num_workers)

        # # loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False, prob_std=cfg.prob_std),
        # #                                  batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

    return loader_dict


def get_datasets_subsample(cfg):
    tr_std = None
    te_std = None
    if cfg.dataset =='morph':
        if cfg.noised:
            img_root = cfg.img_root
            
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['filename']]
            tr_ages = tr_list['age'].to_numpy()
            tr_noised_ages = tr_list['age_noised'].to_numpy()

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['filename']]
            te_ages = te_list['age'].to_numpy()
            te_noised_ages = te_list['age_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages

    elif cfg.dataset =='clap':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        
            tr_noised_ages = np.array(tr_list['age_noised'])
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            ## tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/clap/setting_tau3/noise20%/WARMUP_V2_CentroidFixed_noisedTrue20%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau3_Feval_on_test_GOL_vgg16v2norm_2024-09-30 12:34:26/updated.csv').iloc[:,-1])
            tr_ages = np.array(tr_list['age'])
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_noised_ages = np.array(te_list['age_noised'])
            te_ages = np.array(te_list['age'])
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in range(len(te_list))]

            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]
        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = None
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in
                       range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in
                       range(len(te_list))]
            te_ages = te_list[:, cfg.lb_idx]
            te_nosied_ages = None
            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]

    elif cfg.dataset =='aadb':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file)
            # tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, names=['img_dir', 'score'])
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['img_dir']]
            tr_ages = tr_list['score'].to_numpy()
            tr_noised_ages = tr_list['score_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/aadb/setting_tau5/noise40%/V2_CentroidFixed_noisedTrue40%_SOL_DiscrTrueL1_pairwise_lr5e-05_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau5_F0_GOL_vgg16v2norm_2024-10-01 19:42:17/updated.csv').iloc[:,-1])


            te_list = pd.read_csv(cfg.test_file)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['img_dir']]
            te_ages = te_list['score'].to_numpy()
            te_noised_ages = te_list['score_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, header=None)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter, header=None)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages
    elif cfg.dataset == 'rsna' or cfg.dataset =='rsna_toy':
        if cfg.noised:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_list['boneage_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/rsna_revised/setting_tau3/noise15%/CentroidFixed_SOL_DiscrTrueL1_StochasticOrderTrue_pairwise_lr5e-05_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2025-01-18 18:37:10/updated.csv').iloc[:,-1])

            # te_img_root = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_img_root2 = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_list1 = pd.read_csv(cfg.test_file)
            # te_imgs =
            # te_ages = te_list1['Bone Age (months)'].to_numpy()
            # te_noised_ges = None

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel('/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages = te_list['Ground truth bone age (months)'].to_numpy()
        else:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_ages

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel(
                '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages =te_ages

    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

            if cfg.noised:
                tr_noised_ages = data['age_noised']
            else:
                tr_noised_ages = None

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']

            if cfg.noised:
                te_noised_ages = data['age_noised']
            else:
                te_noised_ages = None


    sampled_idxs = uniform_sample_nodes(tr_noised_ages, keep_ratio =cfg.sampling_ratio)
    cfg.sampled_idxs = sampled_idxs
    if cfg.noised:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['train_for_val_sampled'] = DataLoader(
            basic.Basic_Noised_Stochastic(np.array(tr_imgs)[sampled_idxs], tr_noised_ages[sampled_idxs], tr_ages[sampled_idxs], transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)
        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)


    else:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)



    return loader_dict


def get_datasets_adaptiveSigma(cfg):
    tr_std = None
    te_std = None
    if cfg.dataset =='morph':
        if cfg.noised:
            img_root = cfg.img_root
            
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['filename']]
            tr_ages = tr_list['age'].to_numpy()
            tr_noised_ages = tr_list['age_noised'].to_numpy()

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['filename']]
            te_ages = te_list['age'].to_numpy()
            te_noised_ages = te_list['age_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages

    elif cfg.dataset =='clap':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        
            tr_noised_ages = np.array(tr_list['age_noised'])
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            ## tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/clap/setting_tau3/noise20%/WARMUP_V2_CentroidFixed_noisedTrue20%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau3_Feval_on_test_GOL_vgg16v2norm_2024-09-30 12:34:26/updated.csv').iloc[:,-1])
            tr_ages = np.array(tr_list['age'])
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_noised_ages = np.array(te_list['age_noised'])
            te_ages = np.array(te_list['age'])
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in range(len(te_list))]

            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]
        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_list = np.array(tr_list)
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = None
            tr_imgs = [f'{img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in
                       range(len(tr_list))]
            tr_std = tr_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(tr_ages < 60).flatten()
            # tr_ages = tr_ages[idx]
            # tr_imgs = np.array(tr_imgs)[idx]
            # tr_std = tr_std[idx]

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{te_list[i, cfg.data_type_idx]}/{te_list[i, cfg.img_idx]}' for i in
                       range(len(te_list))]
            te_ages = te_list[:, cfg.lb_idx]
            te_nosied_ages = None
            te_std = te_list[:, cfg.std_idx]
            #
            # # debug for n_ranks and margin relation
            # idx = np.argwhere(te_ages < 60).flatten()
            # te_ages = te_ages[idx]
            # te_imgs = np.array(te_imgs)[idx]
            # te_std = te_std[idx]

    elif cfg.dataset =='aadb':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file)
            # tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, names=['img_dir', 'score'])
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['img_dir']]
            tr_ages = tr_list['score'].to_numpy()
            tr_noised_ages = tr_list['score_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/aadb/setting_tau5/noise40%/V2_CentroidFixed_noisedTrue40%_SOL_DiscrTrueL1_pairwise_lr5e-05_refinement_noiseRate0.85_correct_constant_rate_mean/PREFIX_0.25_tau5_F0_GOL_vgg16v2norm_2024-10-01 19:42:17/updated.csv').iloc[:,-1])


            te_list = pd.read_csv(cfg.test_file)
            te_imgs = [f"{img_root}/{i_path}" for i_path in te_list['img_dir']]
            te_ages = te_list['score'].to_numpy()
            te_noised_ages = te_list['score_noised'].to_numpy()

        else:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter, header=None)
            tr_list = np.array(tr_list)
            tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
            tr_ages = tr_list[:, cfg.lb_idx]
            tr_noised_ages = tr_ages

            te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter, header=None)
            te_list = np.array(te_list)
            te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
            te_ages = te_list[:, cfg.lb_idx]
            te_noised_ages = te_ages
    elif cfg.dataset == 'rsna' or cfg.dataset =='rsna_toy':
        if cfg.noised:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_list['boneage_noised'].to_numpy()
            # tr_noised_ages = np.array(pd.read_csv(os.path.join(cfg.save_folder, 'updated.csv')).iloc[:,-1])
            # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/rsna_revised/setting_tau3/noise15%/CentroidFixed_SOL_DiscrTrueL1_StochasticOrderTrue_pairwise_lr5e-05_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2025-01-18 18:37:10/updated.csv').iloc[:,-1])

            # te_img_root = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_img_root2 = os.path.join(cfg.ds_root, 'Bone Age Validation Set/boneage-validation-dataset1')
            # te_list1 = pd.read_csv(cfg.test_file)
            # te_imgs =
            # te_ages = te_list1['Bone Age (months)'].to_numpy()
            # te_noised_ges = None

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel('/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages = te_list['Ground truth bone age (months)'].to_numpy()
        else:
            ds_root = cfg.ds_root
            tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
            tr_list = pd.read_csv(cfg.train_file)
            tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in tr_list['id']]
            tr_ages = tr_list['boneage'].to_numpy()
            tr_noised_ages = tr_ages

            te_img_root = os.path.join(cfg.ds_root, 'Bone Age Test Set/Test Set Images')
            te_list = pd.read_excel(
                '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx')
            te_imgs = [f'{te_img_root}/{i_path}.png' for i_path in te_list['Case ID']]
            te_ages = te_list['Ground truth bone age (months)'].to_numpy()
            te_noised_ages =te_ages

    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

            if cfg.noised:
                tr_noised_ages = data['age_noised']
            else:
                tr_noised_ages = None

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']

            if cfg.noised:
                te_noised_ages = data['age_noised']
            else:
                te_noised_ages = None

    if cfg.noised:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['train_for_val_sampled'] = DataLoader(
            basic.Basic_Noised_Stochastic(np.array(tr_imgs)[sampled_idxs], tr_noised_ages[sampled_idxs], tr_ages[sampled_idxs], transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)
        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)


    else:
        loader_dict = dict()
        n_ranks = len(np.unique(tr_noised_ages))
        if 'pairwise' in cfg.experiment_name:
            loader_dict['train'] = DataLoader(
                OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                               cfg.tau, logscale=cfg.logscale,
                                                               is_filelist=cfg.is_filelist, prob_std=cfg.prob_std,
                                                               n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        else:
            loader_dict['train'] = DataLoader(
                basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                              is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                              n_ranks=n_ranks),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        loader_dict['train_for_val'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['trainval_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['val'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std,
                                          n_ranks=n_ranks),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)



    return loader_dict



def get_clean_dataset(cfg, clean_indices):
    if cfg.dataset == 'morph':
        if cfg.noised:
            img_root = cfg.img_root
            tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
            tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['filename'].to_numpy()[clean_indices]]
            tr_ages = tr_list['age'].to_numpy()[clean_indices]
            tr_noised_ages = tr_list['age_noised'].to_numpy()[clean_indices]

    loader_dict = dict()
    loader_dict['train'] = DataLoader(
        OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                       cfg.tau, logscale=cfg.logscale,
                                                       is_filelist=cfg.is_filelist, prob_std=cfg.prob_std),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(
        basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                      is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers)


    return loader_dict['train'], loader_dict['train_for_val']