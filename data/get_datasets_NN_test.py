import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from data.datasets import basic


def get_datasets_NN_test(cfg):
    if cfg.is_filelist:
        if cfg.dataset == 'morph':
            if cfg.noised:
                img_root = cfg.img_root
                tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
                tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['filename']]
                tr_ages = tr_list['age'].to_numpy()
                # tr_noised_ages = tr_list['age_noised'].to_numpy()

                tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/ICLR2025_F/morph/settingA_tau3/noise20%/WARMUP_V2_CentroidFixed_noisedTrue20%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2024-09-29 19:10:22/updated.csv').iloc[:,-1])
                # tr_noised_ages = np.array(pd.read_csv('/ssd1/cwlee/datasets/ICLR2025_F/morph/settingA_tau3/noise40%/WARMUP_V2_CentroidFixed_noisedTrue40%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2024-09-29 20:33:27/updated.csv').iloc[:,-1])

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

        elif cfg.dataset == 'clap':
            if cfg.noised:
                img_root = cfg.img_root
                tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
                tr_list = np.array(tr_list)
                tr_noised_ages = tr_list[:, 5]
                tr_ages = tr_list[:, cfg.lb_idx]
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
                te_noised_ages = te_list[:, 5]
                te_ages = te_list[:, cfg.lb_idx]
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
                tr_imgs = [f"{img_root}/{i_path}" for i_path in tr_list['img_dir']]
                tr_ages = tr_list['score'].to_numpy()
                tr_noised_ages = tr_list['score_noised'].to_numpy()

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
                te_noised_ages = te_ages


    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']

    loader_dict = dict()
    if cfg.noised:
        loader_dict['train'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['train_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.num_workers)

        loader_dict['test'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
        loader_dict['test_vis'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_vis,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    else:
        loader_dict['train'] = DataLoader(
        basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                        is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers)
        loader_dict['test'] = DataLoader(
            basic.Basic_Noised_Stochastic(te_imgs, te_noised_ages, te_ages, transform=cfg.transform_te,
                                          is_filelist=cfg.is_filelist, norm_age=False, prob_std=cfg.prob_std),
            batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

        # loader_dict['train'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_te, cfg.tau, is_filelist=cfg.is_filelist),
        #                                   batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
        # loader_dict['test'] = DataLoader(basic.Basic(te_imgs, te_ages, cfg.transform_te, cfg.tau, is_filelist=cfg.is_filelist),
        #                                  batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict





