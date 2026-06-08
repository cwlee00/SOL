import os
import time
import sys
from copy import deepcopy

import math
import pandas as pd
import numpy as np
import random
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import defaultdict

from config.basic import ConfigBasic
from utils.util import write_log, get_current_time, to_np, make_dir, log_configs, save_ckpt, set_wandb
from utils.util import adjust_learning_rate, AverageMeter, ClassWiseAverageMeter, cls_accuracy, extract_embs, \
    print_eval_result_by_groups_and_k, evaluate_metric, update_centroid, evaluate_metric_aadb, evaluate_metric_rsna
from utils.loss_util import compute_stochastic_order_loss, compute_discriminative_loss
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_tr_OLbasic_val_NN import get_datasets

from utils.util import select_n_random, get_prob_db_tensor, update_centroid_allRanks
from torch.utils.tensorboard import SummaryWriter
import argparse

from torch.utils.data import DataLoader
from data.datasets import OL_basic_train, basic


def parse_args():
    ############ cmd process ##############
    parser = argparse.ArgumentParser(description='SOL')
    parser.add_argument('--result_root_path', type=str, default="/media/cwlee/HDD1/2024/ICLR2025_finalV2/",
                        help="path to store the results")
    parser.add_argument('--dataset', type=str, default='morph',
                        help='dataset')
    # parser.add_argument('--seed', type=int, default=0,
    #                     help="seed")
    parser.add_argument('--gpu', type=int, default=0,
                        help="train on which cuda device")
    ########## data settings #########
    parser.add_argument('--noised', type=bool, default=True,
                        help='whether data is noised')
    parser.add_argument('--noise_percentage', type=int, default=30,
                        help='percentage of gaussian noise')
    parser.add_argument('--noise_type', type=str, default='Gaussian',
                        help='MORPH dataset setting')
    parser.add_argument('--setting', type=str, default='A',
                        help='MORPH dataset setting')
    parser.add_argument('--fold', type=str, default='0',
                        help='dataset fold number')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='dataset fold number')
    parser.add_argument('--L', type=int, default=1,
                        help="train on which cuda device")
    parser.add_argument('--tau', type=int, default=3,
                    help="train on which cuda device")
    parser.add_argument('--noise_rate', type=float, default=0.9,
                        help='dataset fold number')
    parser.add_argument('--seed', type=int, default=0,
                    help="train on which cuda device")
    ########## cmd end ############

    args = parser.parse_args()

    return args


def set_local_config(cfg, args):
    # Dataset
    cfg.sampling_ratio = 1.0
    cfg.dataset = args.dataset
    cfg.setting = args.setting
    cfg.fold = args.fold
    #############################
    cfg.seed = args.seed
    cfg.noised = args.noised
    cfg.batch_size = 32
    cfg.prob_std = 1
    cfg.noise_percentage = args.noise_percentage
    cfg.noise_type = args.noise_type
    #############################
    cfg.logscale = False
    cfg.set_dataset()
    # if cfg.dataset == 'aadb':
    #     cfg.tau = 5
    # else:
    #     cfg.tau = 3  # original value:1 # 3 for others, 1 for adience
    cfg.tau = args.tau
    # Model
    cfg.model = 'GOL'
    cfg.backbone = 'vgg16v2norm'
    cfg.metric = 'L2_squared'
    # cfg.k = np.arange(2, 60, 2)
    cfg.k = 50
    cfg.epochs = 100
    cfg.scheduler = 'cosine'
    
    cfg.lr_decay_epochs = [100, 200, 300]
    cfg.period = 3

    cfg.margin = 0.25
    cfg.ref_mode = 'stochastic_OL'
    cfg.ref_point_num = 55  # 60 Fold1, 58 Fold0 setting D // 56 setting c // 58 setting B // 55 setting A
    cfg.fiducial_point_num = 55
    cfg.drct_wieght = 1
    cfg.start_norm = True
    ################################################33
    cfg.norm_centroids = True
    cfg.test_mode = 'nearest_E'
    cfg.metric_loss = False
    cfg.discriminative_loss = True
    cfg.order_loss = False
    cfg.center_loss = True
    cfg.stochastic_centroid = True

    cfg.L = args.L
    cfg.N = 2
    cfg.learning_rate = args.lr
    if cfg.dataset == 'aadb' or cfg.dataset == 'rsna':
        cfg.learning_rate = 0.00005
    cfg.change_std = False
    # if cfg.dataset == 'clap' or cfg.dataset == 'aadb':
    #     cfg.noise_rate = 0.85
    # elif cfg.dataset == 'morph' or cfg.dataset == 'rsna':
    #     cfg.noise_rate = 0.9
    cfg.noise_rate= args.noise_rate

    cfg.refine_version = 'constant_rate_mean'
    if cfg.dataset == 'aadb':
        cfg.start_epoch = 0
    elif cfg.dataset == 'rsna':
        cfg.start_epoch = 0
    elif cfg.dataset == 'clap':
        cfg.start_epoch = 0
        # cfg.start_epoch = 1
    else:
        cfg.start_epoch = 1
    ########################################################
    # Log
    cfg.wandb = False

    if cfg.discriminative_loss:
        alg_name = 'SOL'
    else:
        alg_name = 'GOL'

    cfg.experiment_name = f'WARMUP_V2_CentroidFixed_noised{cfg.noised}{cfg.noise_percentage}%_{alg_name}_Discr{cfg.center_loss}L{cfg.L}_pairwise_lr{cfg.learning_rate}_refinement_noiseRate{cfg.noise_rate}_correct_{cfg.refine_version}'
    cfg.save_folder = f'../{cfg.dataset}_{cfg.noise_type}/SEED{cfg.seed}_setting{cfg.setting}_tau{cfg.tau}/noise{cfg.noise_percentage}%/{cfg.experiment_name}/PREFIX_{cfg.margin}_tau{cfg.tau}_F{cfg.fold}_{cfg.model}_{cfg.backbone}_{get_current_time()}'
    make_dir(cfg.save_folder)

    cfg.n_gpu = 1
    cfg.num_workers = 1
    if cfg.dataset == 'aadb' or cfg.dataset == 'rsna':
        cfg.num_workers = 8
    cfg.gpu_ids = [args.gpu]
    cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    return cfg


def detect_outliers(embs_train, ranks_train, pred_deviation, noise_level, train_labels, real_train_labels, cfg):
    indices = np.arange(len(embs_train))

    title = '-' * 150
    title += '\nRank num\t\\  '
    title += '\t# of train labels\t|\tAvg noise level\t\t|\tAvg abs pred dev\t|\tPred dev threshold\t|\tnoise(clean)/noise(noisy)\t|'
    title += '\n' + '-' * 150
    write_log(cfg.noiseLogfile, title)
    # write_log(cfg.ALLnoiseLogfile, title)
    clean_indices = []
    noisy_indices = []

    # if pred_deviation_j.max() > abs(pred_deviation).mean():
    # noisy_pr = np.zeros_like(pred_deviation_j)
    for j in range(cfg.n_ranks):
        rank_log = f'\t{j:^5}\t|'

        rank_mask = ranks_train == j
        pred_deviation_j = abs(pred_deviation)[rank_mask]
        noise_level_j = noise_level[rank_mask].astype(int)
        rank_log += f'\t{rank_mask.sum():^18}\t|'
        rank_log += f'\t{np.round(noise_level_j.mean(), 2):^18}\t|'
        rank_log += f'\t{np.round(pred_deviation_j.mean(), 2):^18}\t|'
        rank_log += f'\t{np.round(pred_deviation_j.max() * cfg.noise_rate, 2):^18}\t|'

        # if pred_deviation_j.max() > abs(pred_deviation).mean():
        if len(pred_deviation_j) > 0:
            if pred_deviation_j.max() == 0:
                noisy_pr = np.ones_like(pred_deviation_j)
                # noisy_pr = np.zeros_like(pred_deviation_j)
            else:
                noisy_pr = pred_deviation_j / pred_deviation_j.max()

            if (noisy_pr < cfg.noise_rate).any():
                clean_idx = np.where(noisy_pr < (cfg.noise_rate))[0]
                clean_indices += indices[rank_mask][clean_idx].tolist()
                rank_log += f'\t{np.round(noise_level_j[clean_idx].mean(), 2):^9}'
            else:
                rank_log += f'\tNo clean\t'
            if (noisy_pr >= (cfg.noise_rate)).any():
                noisy_idx = np.where(noisy_pr >= (cfg.noise_rate))[0]
                noisy_indices += indices[rank_mask][noisy_idx].tolist()
                rank_log += f'/{np.round(noise_level_j[noisy_idx].mean(), 2):^9}\t|'
            else:
                rank_log += f'\tNo noisy\t'
        else:
            clean_indices += indices[rank_mask].tolist()
            rank_log += f'\t{np.round(noise_level_j.mean(), 2)}/No noisy\t|'
        # write_log(cfg.ALLnoiseLogfile, rank_log)

    # rank_log += '\n'+'-' * 150
    # rank_log += '\nRank num\t\\  '
    # rank_log += '\t# of train labels\t|\tAvg noise level\t\t|\tAvg abs pred dev\t|\tPred dev threshold\t|\tnoise(clean)/noise(noisy)\t|'
    # rank_log += '\n'+'-' * 150
    if len(clean_indices) == 0:
        rank_log = f'\n\tTotal\t|{len(ranks_train):^22}|{(np.round(noise_level.mean(), 2)):^24}|{np.round(abs(pred_deviation).mean(), 2):^23}|{cfg.noise_rate:^23}|\t{np.round(noise_level[np.asarray(noisy_indices)].mean(), 2):^12}'
    else:
        rank_log = f'\n\tTotal\t|{len(ranks_train):^22}|{(np.round(noise_level.mean(), 2)):^24}|{np.round(abs(pred_deviation).mean(), 2):^23}|{cfg.noise_rate:^23}|\t{np.round(noise_level[np.asarray(clean_indices)].mean(), 2):^12}/{np.round(noise_level[np.asarray(noisy_indices)].mean(), 2):^12}'
    write_log(cfg.noiseLogfile, rank_log)

    final_noisy_indices = np.asarray(noisy_indices)
    final_clean_indices = np.asarray(clean_indices)

    return final_noisy_indices, final_clean_indices


def correct_noisy_instances(pred_deviation, noisy_indices, train_labels, correction_rate, cfg):
    to_correct_mask = np.zeros_like(pred_deviation).astype(bool)
    to_correct_mask[noisy_indices] = True

    correct_dir = np.zeros_like(pred_deviation)
    correct_dir[np.logical_and(to_correct_mask, pred_deviation > 0)] = - 1
    correct_dir[np.logical_and(to_correct_mask, pred_deviation < 0)] = + 1

    if cfg.dataset == 'aadb':
        new_train_labels = np.round(train_labels + correct_dir * correction_rate, 2)
        new_train_labels[new_train_labels < 0] = 0
        new_train_labels[new_train_labels > 1] = 1
    else:
        new_train_labels = np.round(train_labels + correct_dir * correction_rate)
        new_train_labels[new_train_labels < 0] = 0

    return new_train_labels



def get_new_dataloader(loader_dict, new_train_labels, cfg):

    tr_noised_ages = new_train_labels

    if cfg.dataset == 'morph':
        tr_imgs = [f"{cfg.img_root}/{i_path}" for i_path in cfg.tr_list['filename']]
        tr_ages = cfg.tr_list['age'].to_numpy()

        if cfg.epoch == cfg.start_epoch+1:
            cfg.new_tr_list = cfg.tr_list.copy().drop('noise', axis=1)
        cfg.new_tr_list[f'age_noised_ep{cfg.epoch}'] = new_train_labels.astype(int)
        cfg.new_tr_list.to_csv(os.path.join(cfg.save_folder, f'updated.csv'))

    elif cfg.dataset == 'clap' or cfg.dataset == 'clap_v2':
        if cfg.dataset == 'clap':
            tr_list = np.asarray(cfg.tr_list)
        tr_imgs = [f'{cfg.img_root}/{tr_list[i, cfg.data_type_idx]}/{tr_list[i, cfg.img_idx]}' for i in
                   range(len(tr_list))]
        tr_ages = tr_list[:, cfg.lb_idx]

        if cfg.epoch == cfg.start_epoch+1:
            cfg.new_tr_list = cfg.tr_list.copy().drop('noise', axis=1)
        cfg.new_tr_list[f'age_noised_ep{cfg.epoch}'] = new_train_labels.astype(int)
        cfg.new_tr_list.to_csv(os.path.join(cfg.save_folder, f'updated.csv'))

    elif cfg.dataset == 'aadb':
        tr_imgs = [f"{cfg.img_root}/{i_path}" for i_path in cfg.tr_list['img_dir']]
        tr_ages = cfg.tr_list['score'].to_numpy()

        if cfg.epoch == cfg.start_epoch:
            cfg.new_tr_list = cfg.tr_list.copy().drop('noise', axis=1)
        cfg.new_tr_list[f'age_noised_ep{cfg.epoch}'] = new_train_labels.astype(int)
        cfg.new_tr_list.to_csv(os.path.join(cfg.save_folder, f'updated.csv'))


    elif cfg.dataset == 'rsna':
        tr_img_root = os.path.join(cfg.ds_root, 'Bone Age Training Set/boneage-training-dataset')
        tr_imgs = [f'{tr_img_root}/{i_path}.png' for i_path in cfg.tr_list['id']]
        tr_ages = cfg.tr_list['boneage'].to_numpy()

        if cfg.epoch == cfg.start_epoch:
            cfg.new_tr_list = cfg.tr_list.copy().drop('noise', axis=1)
        cfg.new_tr_list[f'age_noised_ep{cfg.epoch}'] = new_train_labels.astype(int)
        cfg.new_tr_list.to_csv(os.path.join(cfg.save_folder, f'updated.csv'))
        

    if 'pairwise' in cfg.experiment_name:
        loader_dict['train'] = DataLoader(
            OL_basic_train.OLBasic_Train_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, cfg.transform_tr,
                                                           cfg.tau, logscale=cfg.logscale,
                                                           is_filelist=cfg.is_filelist),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    else:
        loader_dict['train'] = DataLoader(
            basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_tr,
                                          is_filelist=cfg.is_filelist, norm_age=False),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

    loader_dict['train_for_val'] = DataLoader(
        basic.Basic_Noised_Stochastic(tr_imgs, tr_noised_ages, tr_ages, transform=cfg.transform_te,
                                      is_filelist=cfg.is_filelist, norm_age=False),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers)

    cfg.n_ranks = len(np.unique(loader_dict['train'].dataset.ranks))
    return loader_dict



def main():
    # np.random.seed(999)
 

    cfg = ConfigBasic()
    args = parse_args()

    # random_seed = 10
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    cfg = set_local_config(cfg, args)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')
    cfg.noiseLogfile = log_configs(cfg, log_file='train_noise_track_log.txt')
    # dataloader
    loader_dict = get_datasets(cfg)
    # cfg.n_ranks = len(np.unique(loader_dict['train'].dataset.ranks))
    cfg.n_ranks = loader_dict['train'].dataset.ranks.max() + 1
    print(f'[*] {cfg.n_ranks} ranks exist. ')
    #######################################
    cfg.ref_point_num = cfg.n_ranks
    cfg.fiducial_point_num = cfg.n_ranks
    writer = SummaryWriter(f'runs/{cfg.experiment_name}')

    cfg.tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)

    model = prepare_model(cfg)

    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    if cfg.adam:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    if cfg.scheduler == 'cosine':
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate * 0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate * 0.001)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)
    elif cfg.scheduler == 'sequential':
        scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs - 10, eta_min=cfg.learning_rate * 0.001)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])

    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.to(cfg.device)
    if cfg.dataset == 'rsna':
        val_mae_best = 11.0
    else:
        val_mae_best = 4.5
    log_dict = dict()
    # init loss matrix
    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]

    for epoch in range(cfg.epochs):
        cfg.epoch = epoch

        if epoch == 0:
            with torch.no_grad():
                model.eval()
                embs_train, prob_dbs_train = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
                embs_train = embs_train.to(cfg.device)
                centroids = update_centroid(embs_train, prob_dbs_train, cfg)
        else:
            centroids = cfg.centroids

        print("==> training...")
        time1 = time.time()
        train_loss, loss_record = train(epoch, loader_dict['train'], centroids, model, optimizer, cfg,
                                        prev_loss_record=loss_record)

        if cfg.scheduler:
            scheduler.step()

        if epoch % cfg.val_freq == 0:
            print('==> validation...')
            if cfg.test_mode == 'kNN':
                val_mae, val_cs, acc = validate_kNN(loader_dict, model, writer, cfg)
            if cfg.test_mode == 'nearest_E':
                val_mae, val_cs, acc, loader_dict = validate_nearestExpectation(loader_dict, model, writer, cfg)

                time2 = time.time()
                print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))
            elif cfg.test_mode == 'all':
                val_mae, val_cs, acc = validate_all(loader_dict, model, writer, cfg)

            if val_mae < val_mae_best:
                val_mae_best = val_mae
                # save_ckpt(cfg, model, f'ep_{epoch}_val_best_{val_mae:.3f}_k{best_k}.pth')
                save_ckpt(cfg, model, f'ep_{epoch}_val_best_{val_mae:.3f}.pth')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['Train Loss'] = train_loss
            log_dict['Val Mae'] = val_mae
            log_dict['LR'] = scheduler.get_lr()[0] if scheduler else cfg.learning_rate
            wandb.log(log_dict)

    print('[*] Training ends')


def update_loss_matrix(A, loss, base_ranks, ref_ranks=None):
    batch_size = len(base_ranks)
    if ref_ranks is not None:
        for i in range(batch_size):
            A[0][base_ranks[i], ref_ranks[i]] += loss[i]
            A[1][base_ranks[i], ref_ranks[i]] += 1
    else:
        for i in range(batch_size):
            A[0][base_ranks[i]] += loss[i]
            A[1][base_ranks[i]] += 1
    return A


def get_pairs_equally(ranks, tau, m=32):
    orders = []
    base_idx = []
    ref_idx = []
    N = len(ranks)

    base_prob_db = []
    ref_prob_db = []

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand(1) > 0.5:
                base_idx.append(i)
                ref_idx.append(j)
                order_ij = get_order_labels(ranks[i], ranks[j], tau)
                orders.append(order_ij)

                # base_prob_db.append(prob_dbs[i])
                # ref_prob_db.append(prob_dbs[j])

            else:
                base_idx.append(j)
                ref_idx.append(i)
                order_ji = get_order_labels(ranks[j], ranks[i], tau)
                orders.append(order_ji)

                # base_prob_db.append(prob_dbs[j])
                # ref_prob_db.append(prob_dbs[i])

    refine = []
    orders = np.array(orders)

    for o in range(3):
        o_idxs = np.argwhere(orders == o).flatten()
        if len(o_idxs) > m:
            sel = np.random.choice(o_idxs, m, replace=False)
            refine.append(sel)
        else:
            refine.append(o_idxs)

    refine = np.concatenate(refine)
    base_idx = np.array(base_idx)[refine]
    ref_idx = np.array(ref_idx)[refine]
    orders = orders[refine]

    # base_prob_db = np.array(base_prob_db)[refine]
    # ref_prob_db = np.array(ref_prob_db)[refine]
    # real_orders = np.array([get_order_labels(ranks_real[base_idx[i]], ranks_real[ref_idx[i]], tau) for i in range(len(base_idx))])

    # return base_idx, ref_idx, orders, base_prob_db, ref_prob_db
    return base_idx, ref_idx, orders


def get_order_labels(rank_base, rank_ref, tau):
    if rank_base > rank_ref + tau:
        order = 0
    elif rank_base < rank_ref - tau:
        order = 1
    else:
        order = 2
    return order


def hellinger_dot(p, q):
    """Hellinger distance between two discrete distributions.
       Using numpy.
       For Python >= 3.5 only"""
    z = torch.sqrt(p) - torch.sqrt(q)
    return torch.sqrt(z @ z / 2)


def train(epoch, train_loader, centroids, model, optimizer, cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    center_losses = AverageMeter()
    angle_losses = AverageMeter()
    discr_losses = AverageMeter()
    dist_losses = AverageMeter()

    angle_acc_meter = ClassWiseAverageMeter(2)
    # dist_acc_meter = ClassWiseAverageMeter(2)

    loss_record = deepcopy(prev_loss_record)
    end = time.time()

    for idx, batch_data in enumerate(train_loader):
        x_base = batch_data['base_img']
        x_ref = batch_data['ref_img']
        # ranks, prob_dbs = batch_data['ranks'], batch_data['prob_db']
        ranks = batch_data['ranks']
        labels = batch_data['noised_base_age']
        prob_dbs_std = batch_data['prob_db_std']
        ################################################################################################################
        # Get pairs
        labels_np = torch.cat(ranks).detach().numpy()
        # prob_dbs_np = torch.cat(prob_dbs)
        prob_dbs_np = torch.cat(
            [get_prob_db_tensor(len(centroids), m, cfg.prob_std).unsqueeze(0) for m in labels_np])
        prob_dbs_std = torch.cat(prob_dbs_std).detach().cpu().numpy()
        base_idx, ref_idx, order_labels = get_pairs_equally(labels_np, cfg.tau)
        # base_idx, ref_idx, order_labels, base_prob_db, ref_prob_db = get_pairs_equally(labels_np, prob_dbs_np, cfg.tau)
        ################################################################################################################

        if torch.cuda.is_available():
            x_base = x_base.to(cfg.device)
            x_ref = x_ref.to(cfg.device)

            prob_dbs_np = prob_dbs_np.to(cfg.device)

        data_time.update(time.time() - end)

        # ===================forward=====================
        embs = model.encoder(torch.cat([x_base, x_ref], dim=0))

        # =====================loss======================
        tic = time.time()

        total_loss = 0.
        if cfg.discriminative_loss:
            discriminative_loss = compute_stochastic_order_loss(embs, base_idx, ref_idx, labels_np, prob_dbs_np,
                                                                centroids, cfg.margin, cfg, record=False)
            discr_loss_time = time.time() - tic
            tic = time.time()

            total_loss += discriminative_loss
            discr_losses.update(discriminative_loss.item(), x_base.size(0))

        if cfg.center_loss:
            center_loss = compute_discriminative_loss(embs, centroids, prob_dbs_np,  cfg, record=False)
            # center_loss = compute_fine_discriminative_loss_v3(embs, centroids, prob_dbs_np, prob_dbs_std, cfg, record=False)
            center_loss_time = time.time() - tic
            tic = time.time()

            total_loss += center_loss
            center_losses.update(center_loss.item(), x_base.size(0))

        # total_loss = (cfg.drct_wieght * angle_loss) + center_loss + discriminative_loss + dist_loss
        losses.update(total_loss.item(), x_base.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'

                      f'Discriminative-Loss {discr_losses.val:.4f}\t'
                      f'Center-Loss {center_losses.val:.4f}\t'
                      f'Angle-Acc [{angle_acc_meter.val[0]:.3f}  {angle_acc_meter.val[1]:.3f}]  [{angle_acc_meter.total_avg:.3f}]\t'
                      )
            sys.stdout.flush()
    #####################################

    return losses.avg, loss_record


def validate_kNN(loader_dict, model, writer, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train, _ = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    embs_test, _ = extract_embs(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))

    if cfg.noised:
        noised_test_labels = loader_dict['val'].dataset.labels
        test_labels = loader_dict['val'].dataset.real_labels
        train_labels = loader_dict['train_for_val'].dataset.labels
        real_train_labels = loader_dict['train_for_val'].dataset.real_labels
    else:
        test_labels = loader_dict['val'].dataset.labels
        train_labels = loader_dict['train_for_val'].dataset.labels

    preds_all = defaultdict(list)

    k = cfg.k
    with torch.no_grad():
        end = time.time()
        for idx in range(n_batch):
            data_time.update(time.time() - end)
            i_st = idx * cfg.batch_size
            i_end = min(i_st + cfg.batch_size, n_test)

            # ===================meters=====================
            vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=k,
                                  metric=cfg.metric)
            inds = np.squeeze(to_np(inds), 0)

            nn_labels = train_labels[inds[:, :k]]
            pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
            preds_all[k].append(pred_mean)

    preds_all[k] = np.concatenate(preds_all[k])
    pred_age = preds_all[k]

    mae, cs, acc = evaluate_metric(pred_age, test_labels)

    ############### Test################
    # acc = np.sum(test_labels == pred_age) / len(test_labels)
    write_log(cfg.logfile, f'MAE: {mae:.3f}, CS:{cs:.4f}, Acc : {acc * 100:.2f}')
    sys.stdout.flush()
    return mae, cs, acc


def validate_nearestExpectation(loader_dict, model, writer, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train, prob_dbs_train = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)
    if cfg.stochastic_centroid:
        centroids = update_centroid(embs_train, prob_dbs_train, cfg)
        cfg.centroids = centroids
    else:
        centroids = model.ref_points

    embs_test, _= extract_embs(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)

    if cfg.noised:
        test_labels = loader_dict['val'].dataset.real_labels.astype(float)
        train_labels = loader_dict['train_for_val'].dataset.labels.astype(float)
        real_train_labels = loader_dict['train_for_val'].dataset.real_labels
        ranks_train = loader_dict['train_for_val'].dataset.ranks

        mappings = loader_dict['train_for_val'].dataset.mapping
    else:
        test_labels = loader_dict['val'].dataset.labels
        mappings = loader_dict['train_for_val'].dataset.mapping

    ##########################################################################################################

    # Correct
    if cfg.noise_rate != 0:
        # Detect outliers
        pred_age_train = predict_nearestExpectation(embs_train, centroids, mappings, cfg)
        pred_deviation = train_labels - pred_age_train
        noise_level = abs(real_train_labels - train_labels)

        write_log(cfg.noiseLogfile,
                  f'P_noisy - min{abs(pred_deviation).min()}, mean{abs(pred_deviation).astype(float).mean()}, max{abs(pred_deviation).max()}')
        ###############################################################################################################################################
        # title = '-' * 100
        # title += '\nLabel range\t\\  '
        # title += '\t# of train labels\t|\tAvg noise level\t\t|\tPred dev/Abs pred dev\t|'
        # title += '\n' + '-' * 100
        # write_log(cfg.labelLogfile, title)

        # label_log = ''
        # for k in range(np.ceil((train_labels.max() - train_labels.min()) / 10).astype(int)):
        #     label_log = f'\t{10 * k}-{10 * (k + 1)}\t|'

        #     tr_label_range = np.logical_and(train_labels >= 10 * k, train_labels < 10 * (k + 1))
        #     tr_label_num = tr_label_range.sum()
        #     label_log += f'\t{tr_label_num:^18}\t|'
        #     noise_level_mean = noise_level[tr_label_range].mean()
        #     label_log += f'\t{np.round(noise_level_mean, 2):^18}\t|'

        #     pred_dev_mean = pred_deviation[tr_label_range].mean()
        #     abs_pred_dev_mean = abs(pred_deviation)[tr_label_range].mean()
        #     label_log += f'\t{np.round(pred_dev_mean, 2):^9}/{np.round(abs_pred_dev_mean, 2):^9}\t\t|'
        #     write_log(cfg.labelLogfile, label_log)

        noisy_indices, clean_indices = detect_outliers(embs_train, ranks_train, pred_deviation,
                                                       noise_level, train_labels, real_train_labels,
                                                       cfg)

        if cfg.refine_version == 'individual_rate':
            correction_rate = np.floor(0.5 * abs(pred_deviation))
            correction_rate[correction_rate > cfg.tau] = 2

        elif cfg.refine_version == 'constant_rate_mean':
            if cfg.dataset == 'aadb':
                correction_rate = (0.5 * abs(pred_deviation).mean())
            else:
                correction_rate = np.floor(0.5*abs(pred_deviation).mean())
   
        elif cfg.refine_version == 'constant_rate_mean_tau':
            if cfg.dataset == 'aadb':
                correction_rate = (1/cfg.tau * abs(pred_deviation).mean())
            else:
                correction_rate = np.floor(1/cfg.tau * abs(pred_deviation).mean())         
        elif cfg.refine_version == 'constant_rate_fixed':
            correction_rate = 1

        if cfg.dataset == 'aadb' or cfg.dataset == 'rsna':
            new_train_labels = correct_noisy_instances(pred_deviation, noisy_indices, train_labels,
                                                    correction_rate, cfg)
            new_noise_level = abs(real_train_labels - new_train_labels)
            #################################################################################################################################################3
            title = '-' * 125
            title += '\nNoise level\t\\  '
            title += '\tTotal # of instances (before/after)\t|\t# of detected noisy instances\t|\t#of corrected noisy instances'
            title += '\n' + '-' * 125
            write_log(cfg.noiseLogfile, title)

            total_num_corrected = 0
            max_noise_level = max(noise_level.max(), new_noise_level.max())
            if cfg.dataset == 'aadb':
                for k in range(101):
                    noise_level_log = f'\t{np.round(k*0.01, 2)}\t\t|'
                    noise_level_k = noise_level ==np.round(k*0.01, 2)
                    noise_level_k_num = noise_level_k.sum()

                    new_noise_level_k = new_noise_level == np.round(k*0.01, 2)
                    new_noise_level_k_num = new_noise_level_k.sum()

                    noise_level_k_noisy_num = noise_level_k[noisy_indices].sum()
                    new_noise_level_noisy = new_noise_level[noisy_indices][noise_level_k[noisy_indices]]
                    num_corrected_k = ((new_noise_level[noisy_indices][noise_level_k[noisy_indices]] -
                                    noise_level[noisy_indices][noise_level_k[noisy_indices]]) < 0).sum()

                    noise_level_log += f'{noise_level_k_num:>15} \ {new_noise_level_k_num:<15}' + f'({new_noise_level_k_num - noise_level_k_num:^4})|'
                    noise_level_log += f'{noise_level_k_noisy_num:^35}|'
                    noise_level_log += f'{num_corrected_k:^35}|'
                    if noise_level_k_num != 0 or new_noise_level_k_num != 0:
                        write_log(cfg.noiseLogfile, noise_level_log)
                
            else:
                
                for k in range(int(max_noise_level) + 1):
                    noise_level_log = f'\t{k}\t\t|'
                    noise_level_k = noise_level == k
                    noise_level_k_num = noise_level_k.sum()

                    new_noise_level_k = new_noise_level == k
                    new_noise_level_k_num = new_noise_level_k.sum()
   
                    noise_level_k_noisy_num = noise_level_k[noisy_indices].sum()
                    new_noise_level_noisy = new_noise_level[noisy_indices][noise_level_k[noisy_indices]]
                    num_corrected_k = ((new_noise_level[noisy_indices][noise_level_k[noisy_indices]] -
                                    noise_level[noisy_indices][noise_level_k[noisy_indices]]) < 0).sum()

                    noise_level_log += f'{noise_level_k_num:>15} \ {new_noise_level_k_num:<15}' + f'({new_noise_level_k_num - noise_level_k_num:^4})|'
                    noise_level_log += f'{noise_level_k_noisy_num:^35}|'
                    noise_level_log += f'{num_corrected_k:^35}|'
                    write_log(cfg.noiseLogfile, noise_level_log)

            noise_level_log = f'\tTotal\t|{len(noise_level):^39}|{len(noisy_indices):^35}|{((new_noise_level[noisy_indices] - noise_level[noisy_indices]) < 0).sum():^36}|'
            write_log(cfg.noiseLogfile, noise_level_log)
            ######################################################################################################################
            to_print = f'Avg noise level (Before / After) {noise_level.mean():.4f} / {new_noise_level.mean():.4f}'
            to_print += f'\nNoisy indices only (Before / After) {noise_level[noisy_indices].mean():.4f} / {new_noise_level[noisy_indices].mean():.4f}'
            to_print += f'\n # of correct direction {(((train_labels - real_train_labels) < 0)[noisy_indices] == (pred_deviation < 0)[noisy_indices]).sum()}/{len(noisy_indices)}'
            write_log(cfg.logfile, to_print)
            write_log(cfg.noiseLogfile, to_print)

            loader_dict = get_new_dataloader(loader_dict, new_train_labels, cfg)
            new_mappings = loader_dict['train_for_val'].dataset.mapping
            new_embs_train, new_prob_dbs_train = extract_embs(model.encoder, loader_dict['train_for_val'],
                                                                cfg)
            new_embs_train = new_embs_train.to(cfg.device)
            centroids = update_centroid(new_embs_train, new_prob_dbs_train, cfg)
            pred_age_nE = predict_nearestExpectation(embs_test, centroids, new_mappings, cfg)
            if cfg.dataset == 'aadb':
                mae_nE, cs_nE, acc_nE = evaluate_metric_aadb(pred_age_nE, test_labels)
            elif cfg.dataset == 'rsna':
                mae_nE, cs_nE, acc_nE = evaluate_metric_rsna(pred_age_nE, test_labels)
            else:
                mae_nE, cs_nE, acc_nE = evaluate_metric(pred_age_nE, test_labels)
            write_log(cfg.logfile, f'nearest_Expectation - MAE: {mae_nE:.3f}, CS:{cs_nE:.4f}, Acc:{acc_nE}')
            cfg.centroids = centroids

            # cfg.noise_rate += 0.01
        else:
            if cfg.epoch > cfg.start_epoch:
                new_train_labels = correct_noisy_instances(pred_deviation, noisy_indices, train_labels,
                                                        correction_rate, cfg)
                new_noise_level = abs(real_train_labels - new_train_labels)
                #################################################################################################################################################3
                title = '-' * 125
                title += '\nNoise level\t\\  '
                title += '\tTotal # of instances (before/after)\t|\t# of detected noisy instances\t|\t#of corrected noisy instances'
                title += '\n' + '-' * 125
                write_log(cfg.noiseLogfile, title)

                total_num_corrected = 0
                max_noise_level = max(noise_level.max(), new_noise_level.max())
                if cfg.dataset == 'aadb':
                    for k in range(101):
                        noise_level_log = f'\t{np.round(k*0.01, 2)}\t\t|'
                        noise_level_k = noise_level ==np.round(k*0.01, 2)
                        noise_level_k_num = noise_level_k.sum()

                        new_noise_level_k = new_noise_level == np.round(k*0.01, 2)
                        new_noise_level_k_num = new_noise_level_k.sum()

                        noise_level_k_noisy_num = noise_level_k[noisy_indices].sum()
                        new_noise_level_noisy = new_noise_level[noisy_indices][noise_level_k[noisy_indices]]
                        num_corrected_k = ((new_noise_level[noisy_indices][noise_level_k[noisy_indices]] -
                                        noise_level[noisy_indices][noise_level_k[noisy_indices]]) < 0).sum()

                        noise_level_log += f'{noise_level_k_num:>15} \ {new_noise_level_k_num:<15}' + f'({new_noise_level_k_num - noise_level_k_num:^4})|'
                        noise_level_log += f'{noise_level_k_noisy_num:^35}|'
                        noise_level_log += f'{num_corrected_k:^35}|'
                        if noise_level_k_num != 0 or new_noise_level_k_num != 0:
                            write_log(cfg.noiseLogfile, noise_level_log)
                    
                else:
                    
                    for k in range(int(max_noise_level) + 1):
                        noise_level_log = f'\t{k}\t\t|'
                        noise_level_k = noise_level == k
                        noise_level_k_num = noise_level_k.sum()

                        new_noise_level_k = new_noise_level == k
                        new_noise_level_k_num = new_noise_level_k.sum()
    
                        noise_level_k_noisy_num = noise_level_k[noisy_indices].sum()
                        new_noise_level_noisy = new_noise_level[noisy_indices][noise_level_k[noisy_indices]]
                        num_corrected_k = ((new_noise_level[noisy_indices][noise_level_k[noisy_indices]] -
                                        noise_level[noisy_indices][noise_level_k[noisy_indices]]) < 0).sum()

                        noise_level_log += f'{noise_level_k_num:>15} \ {new_noise_level_k_num:<15}' + f'({new_noise_level_k_num - noise_level_k_num:^4})|'
                        noise_level_log += f'{noise_level_k_noisy_num:^35}|'
                        noise_level_log += f'{num_corrected_k:^35}|'
                        write_log(cfg.noiseLogfile, noise_level_log)

                noise_level_log = f'\tTotal\t|{len(noise_level):^39}|{len(noisy_indices):^35}|{((new_noise_level[noisy_indices] - noise_level[noisy_indices]) < 0).sum():^36}|'
                write_log(cfg.noiseLogfile, noise_level_log)
                ######################################################################################################################
                to_print = f'Avg noise level (Before / After) {noise_level.mean():.4f} / {new_noise_level.mean():.4f}'
                to_print += f'\nNoisy indices only (Before / After) {noise_level[noisy_indices].mean():.4f} / {new_noise_level[noisy_indices].mean():.4f}'
                to_print += f'\n # of correct direction {(((train_labels - real_train_labels) < 0)[noisy_indices] == (pred_deviation < 0)[noisy_indices]).sum()}/{len(noisy_indices)}'
                write_log(cfg.logfile, to_print)
                write_log(cfg.noiseLogfile, to_print)

                loader_dict = get_new_dataloader(loader_dict, new_train_labels, cfg)
                new_mappings = loader_dict['train_for_val'].dataset.mapping
                new_embs_train, new_prob_dbs_train = extract_embs(model.encoder, loader_dict['train_for_val'],
                                                                    cfg)
                new_embs_train = new_embs_train.to(cfg.device)
                centroids = update_centroid(new_embs_train, new_prob_dbs_train, cfg)
                pred_age_nE = predict_nearestExpectation(embs_test, centroids, new_mappings, cfg)
                if cfg.dataset == 'aadb':
                    mae_nE, cs_nE, acc_nE = evaluate_metric_aadb(pred_age_nE, test_labels)
                elif cfg.dataset == 'rsna':
                    mae_nE, cs_nE, acc_nE = evaluate_metric_rsna(pred_age_nE, test_labels)
                else:
                    mae_nE, cs_nE, acc_nE = evaluate_metric(pred_age_nE, test_labels)
                write_log(cfg.logfile, f'nearest_Expectation - MAE: {mae_nE:.3f}, CS:{cs_nE:.4f}, Acc:{acc_nE}')
                cfg.centroids = centroids

                # cfg.noise_rate += 0.01
            else:
                pred_age_nE = predict_nearestExpectation(embs_test, centroids, mappings, cfg)
                if cfg.dataset == 'aadb':
                    mae_nE, cs_nE, acc_nE = evaluate_metric_aadb(pred_age_nE, test_labels)
                elif cfg.dataset == 'rsna':
                    mae_nE, cs_nE, acc_nE = evaluate_metric_rsna(pred_age_nE, test_labels)
                else:
                    mae_nE, cs_nE, acc_nE = evaluate_metric(pred_age_nE, test_labels)
                write_log(cfg.logfile, f'nearest_Expectation - MAE: {mae_nE:.3f}, CS:{cs_nE:.4f}, Acc:{acc_nE}')

    else:
        ########################################################################################
        pred_age_nE = predict_nearestExpectation(embs_test, centroids, mappings, cfg)
        if cfg.dataset == 'aadb':
            mae_nE, cs_nE, acc_nE = evaluate_metric_aadb(pred_age_nE, test_labels)
        elif cfg.dataset == 'rsna':
            mae_nE, cs_nE, acc_nE = evaluate_metric_rsna(pred_age_nE, test_labels)
        else:
            mae_nE, cs_nE, acc_nE = evaluate_metric(pred_age_nE, test_labels)
        write_log(cfg.logfile, f'nearest_Expectation - MAE: {mae_nE:.3f}, CS:{cs_nE:.4f}, Acc:{acc_nE}')

    ############### Test################
    sys.stdout.flush()

    return mae_nE, cs_nE, acc_nE, loader_dict


def validate_all(loader_dict, model, writer, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train, prob_dbs_train, labels_train = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    embs_test, _, _ = extract_embs(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)

    if cfg.noised:
        test_labels = loader_dict['val'].dataset.real_labels
        train_labels = loader_dict['train_for_val'].dataset.labels
        real_train_labels = loader_dict['train_for_val'].dataset.real_labels

        mappings = loader_dict['train_for_val'].dataset.mapping
        ranks = loader_dict['train_for_val'].dataset.ranks
        noise_level = abs(real_train_labels - train_labels)
    else:
        test_labels = loader_dict['val'].dataset.labels
        train_labels = loader_dict['train_for_val'].dataset.labels

        mappings = loader_dict['train_for_val'].dataset.mapping
        ranks = loader_dict['train_for_val'].dataset.ranks

    pred_age_kNN = predict_kNN(embs_train, embs_test, train_labels, test_labels, data_time, cfg)
    # pred_age_nCC = predict_nearestCentroid(embs_test, centroids, mappings, cfg)

    if cfg.stochastic_centroid:
        centroids = update_centroid(embs_train, prob_dbs_train, cfg)
    else:
        centroids = model.ref_points
    cfg.centroids = centroids
    pred_age_nE = predict_nearestExpectation(embs_test, centroids, mappings, cfg)

    if cfg.dataset == 'aadb':
        mae_kNN, cs_kNN, acc_kNN = evaluate_metric_aadb(pred_age_kNN, test_labels)
        # mae_nCC, cs_nCC, acc_nCC = evaluate_metric(pred_age_nCC, test_labels)
        mae_nE, cs_nE, acc_nE = evaluate_metric_aadb(pred_age_nE, test_labels)
    else:
        mae_kNN, cs_kNN, acc_kNN = evaluate_metric(pred_age_kNN, test_labels)
        mae_nE, cs_nE, acc_nE = evaluate_metric(pred_age_nE, test_labels)

    write_log(cfg.logfile, f'kNN - MAE: {mae_kNN:.3f}, CS:{cs_kNN:.4f}, Acc:{acc_kNN}')
    # write_log(cfg.logfile, f'nearest_Centroid - MAE: {mae_nCC:.3f}, CS:{cs_nCC:.4f}, Acc:{acc_nCC}')
    write_log(cfg.logfile, f'nearest_Expectation - MAE: {mae_nE:.3f}, CS:{cs_nE:.4f}, Acc:{acc_nE}')

    best_mae, best_CS, best_pred_age = mae_kNN, cs_kNN, pred_age_kNN
    # if mae_nCC < best_mae:
    #     best_mae = mae_nCC
    #     best_CS = cs_nCC
    #     best_pred_age = pred_age_nCC
    if mae_nE < best_mae:
        best_mae = mae_nE
        best_CS = cs_nE
        best_pred_age = pred_age_nE

    ############### Test################
    acc = np.sum(test_labels == best_pred_age) / len(test_labels)
    write_log(cfg.logfile, f'MAE: {best_mae:.3f}, CS:{best_CS:.4f}, Acc : {acc * 100:.2f}')
    sys.stdout.flush()

    return best_mae, best_CS, acc


def predict_kNN(embs_train, embs_test, train_labels, test_labels, data_time, cfg):
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))

    if isinstance(cfg.k, int):
        preds_all = defaultdict(list)
        with torch.no_grad():
            end = time.time()
            for idx in range(n_batch):
                i_st = idx * cfg.batch_size
                i_end = min(i_st + cfg.batch_size, n_test)

                # ===================meters=====================
                vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=cfg.k,
                                      metric=cfg.metric)
                inds = np.squeeze(to_np(inds), 0)
                nn_labels = train_labels[inds[:, :cfg.k]]
                if cfg.dataset == 'aadb':
                    pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32), 2)
                else:
                    pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                preds_all[cfg.k].append(pred_mean)

        for key in preds_all.keys():
            preds_all[key] = np.concatenate(preds_all[key])
        best_pred_age = preds_all[key]

    else:
        preds_all = defaultdict(list)
        with torch.no_grad():
            end = time.time()
            for idx in range(n_batch):
                data_time.update(time.time() - end)
                i_st = idx * cfg.batch_size
                i_end = min(i_st + cfg.batch_size, n_test)

                # ===================meters=====================
                vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=max(cfg.k),
                                      metric=cfg.metric)
                inds = np.squeeze(to_np(inds), 0)
                for k in cfg.k:
                    nn_labels = train_labels[inds[:, :k]]
                    if cfg.dataset == 'aadb':
                        pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32), 2)
                    else:
                        pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                    preds_all[k].append(pred_mean)

        for key in preds_all.keys():
            preds_all[key] = np.concatenate(preds_all[key])

        best_mae, best_k, best_pred_age = print_eval_result_by_groups_and_k(test_labels, train_labels, preds_all,
                                                                            cfg.logfile, interval=3)
    return best_pred_age


def predict_nearestCentroid(embs_test, centroids, mappings, cfg):
    with torch.no_grad():
        if cfg.norm_centroids:
            centroids = nn.functional.normalize(centroids, dim=-1)

        nearest_centroids = torch.cdist(embs_test, centroids).argmin(dim=1)
        nearest_labels = np.asarray(
            [list(mappings.keys())[list(mappings.values()).index(v)] for v in nearest_centroids])

        pred_age = nearest_labels

    return pred_age


def predict_nearestExpectation(embs_test, centroids, mappings, cfg):
    with torch.no_grad():
        if cfg.norm_centroids:
            centroids = nn.functional.normalize(centroids, dim=-1)

        if cfg.metric == 'L2':
            dists = torch.cdist(embs_test, centroids)
        elif cfg.metric == 'L2_squared':
            dists = torch.square(torch.cdist(embs_test, centroids))
        elif cfg.metric == 'cosine':
            dists = 1 - torch.matmul(embs_test, centroids.transpose(1, 0))

        sums = []
        for l in range(len(centroids)):
            S_l = (get_prob_db_tensor(len(centroids), l, cfg.prob_std).to(cfg.device) * dists).sum(dim=-1).unsqueeze(0)
            sums.append(S_l)
        nearest_expectations = torch.cat(sums).argmin(0)

        nearest_labels = np.asarray(
            [list(mappings.keys())[list(mappings.values()).index(v)] for v in nearest_expectations])
        pred_age = nearest_labels

    return pred_age


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
