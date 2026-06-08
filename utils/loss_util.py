import torch
import torch.nn as nn
import numpy as np
import itertools
import torch.nn.functional as F
from sympy.polys.subresultants_qq_zz import backward_eye
from utils.util import to_np, get_prob_db_tensor, get_prob_db_tensor_V2, get_S_sum


def compute_stochastic_center_loss_pairwise(embs, centroids, rank_labels, rank_prob_dbs, cfg, record=False):
    centroids = nn.functional.normalize(centroids, dim=-1)
    centroid_ranks = np.array([((cfg.n_ranks - 1) / (len(centroids) - 1)) * i for i in range(len(centroids))])

    def get_pos_neg_idxs(ranks, fdc_ranks, cfg):
        adaptive_margin = cfg.n_ranks != len(fdc_ranks)
        if adaptive_margin:
            nn_idxs = []
            margins = []
            emb_idxs = []
            emb_idx = 0
            for r in ranks:
                abs_diff = np.abs(fdc_ranks-r)
                min_val = abs_diff.min()
                nn = np.argwhere(abs_diff==min_val).flatten()
                nn_idxs.append(nn)

                margin_val = min_val*cfg.margin/(max(cfg.tau, 1))
                margins.append([margin_val]*len(nn))
                emb_idxs.append([emb_idx]*len(nn))
                emb_idx += 1
            nn_idxs = np.concatenate(nn_idxs)
            margins = np.concatenate(margins)
            emb_idxs = np.concatenate(emb_idxs)
        else:
            nn_idxs = ranks
            margins = np.array([0.5 * cfg.margin / (max(cfg.tau, 1))] * len(nn_idxs))
            emb_idxs = np.arange(len(nn_idxs))

        return nn_idxs, emb_idxs, margins

    nn_idxs, emb_idxs, margins = get_pos_neg_idxs(rank_labels, centroid_ranks, cfg)

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    # loss = (dists[nn_idxs, emb_idxs].unsqueeze(-1) * torch.tensor(rank_prob_dbs).to(cfg.device)).sum(dim=1)
    loss = (dists[nn_idxs, emb_idxs].unsqueeze(-1) * rank_prob_dbs).sum(dim=1)
    # loss = nn.functional.relu(violation - cfg.margin)
    # loss = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
    if record:
        return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7), to_np(loss)
    return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7)

def compute_order_loss(embs, base_idx, ref_idx, rank_labels, centroids, cfg, record=False):
    def get_forward_and_backward_idxs(base_idx, ref_idx, ranks, fdc_ranks, cfg):
        batch_size = len(base_idx)
        base_ranks = ranks[base_idx]
        ref_ranks = ranks[ref_idx]
        forward_idxs = []
        backward_idxs = []
        mask = []
        gt = []
        for i in range(batch_size):
            if base_ranks[i] > ref_ranks[i]:
                fdc_1_idx = len(centroids) - np.sum(fdc_ranks - ref_ranks[i] > 0) - 1
                fdc_2_idx = len(centroids) - np.sum(fdc_ranks - base_ranks[i] >= 0)
                fdc_3_idx = fdc_2_idx + 1

                backward_idxs.append([fdc_1_idx, fdc_2_idx])
                if fdc_3_idx >= len(centroids):
                    forward_idxs.append([fdc_2_idx, fdc_2_idx-1])
                else:
                    forward_idxs.append([fdc_3_idx, fdc_2_idx])

                mask.append(True)
                gt.append(0)
            elif base_ranks[i] < ref_ranks[i]:
                fdc_1_idx = len(centroids) - np.sum(fdc_ranks - base_ranks[i] > 0) - 1
                fdc_2_idx = len(centroids) - np.sum(fdc_ranks - ref_ranks[i] >= 0)
                fdc_3_idx = fdc_1_idx - 1
                forward_idxs.append([fdc_2_idx, fdc_1_idx])
                if fdc_3_idx < 0:
                    backward_idxs.append([fdc_1_idx, fdc_1_idx+1])
                else:
                    backward_idxs.append([fdc_3_idx, fdc_1_idx])
                mask.append(True)
                gt.append(1)
            else:
                mask.append(False)

        return np.array(forward_idxs), np.array(backward_idxs), torch.tensor(gt).to(cfg.device), base_idx[mask], ref_idx[mask]

    centroids = nn.functional.normalize(centroids, dim=-1)
    hdim = centroids.shape[-1]
    # fdc_point_ranks = np.array([((cfg.n_ranks-1) / (cfg.fiducial_point_num-1)) * i for i in range(cfg.fiducial_point_num)])
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (len(centroids) - 1)) * i for i in range(len(centroids))])

    direction_matrix = centroids.view(len(centroids), 1, hdim).expand(len(centroids), len(centroids), hdim) - centroids.view(1, len(centroids), hdim).expand(len(centroids), len(centroids), hdim)
    direction_matrix = nn.functional.normalize(direction_matrix, dim=-1)

    forward_idxs, backward_idxs, gt, base_idx, ref_idx = get_forward_and_backward_idxs(base_idx, ref_idx, rank_labels, fdc_point_ranks, cfg)
    batch_size = base_idx.shape[0]

    v_xy = nn.functional.normalize(embs[ref_idx] - embs[base_idx], dim=-1)
    v_forward = direction_matrix[forward_idxs[:,0], forward_idxs[:,1]]
    v_backward = direction_matrix[backward_idxs[:,0], backward_idxs[:,1]]

    v_fb = torch.stack([v_backward, v_forward], dim=-1)
    logits = 20*torch.matmul(v_xy.view(batch_size, 1, hdim), v_fb).squeeze(1)


    if record:
        loss_per_pair = nn.CrossEntropyLoss(reduction='none')(logits, gt)
        loss = torch.mean(loss_per_pair)
        return loss, logits, gt, to_np(loss_per_pair)
    else:
        loss = nn.CrossEntropyLoss()(logits, gt)
        return loss, logits, gt

def compute_metric_loss(embs, base_idx, ref_idx, rank_labels, centroids, margin, cfg, record=False):
    centroids = nn.functional.normalize(centroids, dim=-1)
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (len(centroids) - 1)) * i for i in range(len(centroids))])

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))
    def get_pos_neg_idxs(base_idx, ref_idx, ranks, fdc_ranks, cfg):
        batch_size = len(base_idx)
        base_ranks = ranks[base_idx]
        ref_ranks = ranks[ref_idx]
        row_idxs = []
        pos_idxs = []
        neg_idxs = []
        split_idxs = []

        sim_row_idxs = []
        sim_pos_idxs = []
        sim_neg_idxs = []

        for i in range(batch_size):
            if base_ranks[i] > (ref_ranks[i] + cfg.tau):
                fdc_1_idx = len(centroids) - np.sum(fdc_ranks - ref_ranks[i] > 0) - 1
                fdc_2_idx = len(centroids) - np.sum(fdc_ranks - base_ranks[i] >= 0)

                row_idxs.append(np.arange(fdc_1_idx+1))
                pos_idxs.append([ref_idx[i]]*(fdc_1_idx+1))
                neg_idxs.append([base_idx[i]]*(fdc_1_idx+1))
                row_idxs.append(np.arange(fdc_2_idx, len(centroids)))
                pos_idxs.append([base_idx[i]]*(len(centroids)-fdc_2_idx))
                neg_idxs.append([ref_idx[i]]*(len(centroids)-fdc_2_idx))
                split_idxs.append(fdc_1_idx + 1 + len(centroids) - fdc_2_idx)

            elif base_ranks[i] < (ref_ranks[i] - cfg.tau):
                fdc_1_idx = len(centroids) - np.sum(fdc_point_ranks - rank_labels[base_idx[i]] > 0) - 1
                fdc_2_idx = len(centroids) - np.sum(fdc_point_ranks - rank_labels[ref_idx[i]] >= 0)

                row_idxs.append(np.arange(fdc_1_idx + 1))
                pos_idxs.append([base_idx[i]] * (fdc_1_idx + 1))
                neg_idxs.append([ref_idx[i]] * (fdc_1_idx + 1))
                row_idxs.append(np.arange(fdc_2_idx, len(centroids)))
                pos_idxs.append([ref_idx[i]] * (len(centroids)-fdc_2_idx))
                neg_idxs.append([base_idx[i]] * (len(centroids)-fdc_2_idx))
                split_idxs.append(fdc_1_idx + 1 + len(centroids)- fdc_2_idx)
            else:
                sim_row_idxs.append(np.arange(len(centroids)))
                sim_pos_idxs.append([base_idx[i]]*len(centroids))
                sim_neg_idxs.append([ref_idx[i]]*len(centroids))
                split_idxs.append(len(centroids))
        row_idxs = np.concatenate(row_idxs)
        pos_idxs = np.concatenate(pos_idxs)
        neg_idxs = np.concatenate(neg_idxs)
        sim_row_idxs = np.concatenate(sim_row_idxs)
        sim_pos_idxs = np.concatenate(sim_pos_idxs)
        sim_neg_idxs = np.concatenate(sim_neg_idxs)
        return row_idxs, pos_idxs, neg_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs, split_idxs

    row_idxs, pos_idxs, neg_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs, split_idxs = get_pos_neg_idxs(base_idx, ref_idx, rank_labels, fdc_point_ranks, cfg)

    violation = dists[row_idxs, pos_idxs] - dists[row_idxs,neg_idxs]
    violation = violation + margin

    if len(sim_row_idxs) > 0:
        if cfg.tau == 0:
            sim_violation = torch.abs(dists[sim_row_idxs, sim_pos_idxs] - dists[sim_row_idxs, sim_neg_idxs])
        else:
            sim_violation = torch.abs(dists[sim_row_idxs,sim_pos_idxs] - dists[sim_row_idxs,sim_neg_idxs]) - margin
        loss = torch.cat([nn.functional.relu(violation), nn.functional.relu(sim_violation)])

    else:
        loss = nn.functional.relu(violation)
    if record:
        loss_per_pairs = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
        return torch.sum(loss) / len(base_idx), to_np(loss_per_pairs)
    return torch.sum(loss) / len(base_idx)

def compute_center_loss(embs, rank_labels, centroids, cfg, record=False):
    centroids = nn.functional.normalize(centroids, dim=-1)
    fdc_point_ranks = np.array([((cfg.n_ranks - 1) / (cfg.fiducial_point_num - 1)) * i for i in range(cfg.fiducial_point_num)])

    def get_pos_neg_idxs(ranks, fdc_ranks, cfg):
        adaptive_margin = cfg.n_ranks != cfg.fiducial_point_num
        if adaptive_margin:
            nn_idxs = []
            margins = []
            emb_idxs = []
            emb_idx = 0
            for r in ranks:
                abs_diff = np.abs(fdc_ranks-r)
                min_val = abs_diff.min()
                nn = np.argwhere(abs_diff==min_val).flatten()
                nn_idxs.append(nn)

                margin_val = min_val*cfg.margin/(max(cfg.tau, 1))
                margins.append([margin_val]*len(nn))
                emb_idxs.append([emb_idx]*len(nn))
                emb_idx += 1
            nn_idxs = np.concatenate(nn_idxs)
            margins = np.concatenate(margins)
            emb_idxs = np.concatenate(emb_idxs)
        else:
            nn_idxs = ranks
            margins = np.array([0.5 * cfg.margin / (max(cfg.tau, 1))] * len(nn_idxs))
            emb_idxs = np.arange(len(nn_idxs))

        return nn_idxs, emb_idxs, margins

    nn_idxs, emb_idxs, margins = get_pos_neg_idxs(rank_labels, fdc_point_ranks, cfg)

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    loss = dists[nn_idxs, emb_idxs]

    # loss = nn.functional.relu(violation)
    # loss = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
    # if record:
    #     return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7), to_np(loss)
    # return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7)

    return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7)


def compute_stochastic_order_loss(embs, base_idx, ref_idx, rank_labels, prob_dbs, 
                             centroids, margin, cfg, record=False):
    if cfg.norm_centroids:
        centroids = nn.functional.normalize(centroids, dim=-1)
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (len(centroids) - 1)) * i for i in range(len(centroids))])

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric =='L2_squared':
        dists = torch.square(torch.cdist(centroids, embs))
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    batch_size = len(base_idx)
    base_ranks = rank_labels[base_idx]
    ref_ranks = rank_labels[ref_idx]

    all_prob_dbs = torch.cat(
        [get_prob_db_tensor(len(centroids), m, cfg.prob_std).unsqueeze(0) for m in range(len(centroids))]).to(
        torch.float32).to(cfg.device)  # i: mean
    S_matrix = torch.matmul(all_prob_dbs, dists)  
    
    #########################################################################################################################
    prob_dbs = prob_dbs.to(torch.float32)
    base_prob_dbs = prob_dbs[base_idx]
    ref_prob_dbs = prob_dbs[ref_idx]

    # Compute probability
    prob_same = torch.zeros(base_idx.shape).to(torch.float32).to(cfg.device)
    prob_less, prob_greater = torch.zeros_like(prob_same), torch.zeros_like(prob_same)
    for j in range(len(fdc_point_ranks)):
        min_sim, max_sim = max(0, j - cfg.tau), min(len(centroids) - 1, j + cfg.tau)
        prob_same += (base_prob_dbs[:, j].unsqueeze(-1) * ref_prob_dbs[:, min_sim:max_sim + 1]).sum(axis=1)

        min_less = min(len(centroids) - 1, j + cfg.tau + 1)
        prob_less += (base_prob_dbs[:, j].unsqueeze(-1) * ref_prob_dbs[:, min_less:]).sum(axis=1)
        prob_greater += (base_prob_dbs[:, min_less:] * ref_prob_dbs[:, j].unsqueeze(-1)).sum(axis=1)
    ##########################################################################################################################

    loss = 0.
    for i in range(batch_size):
        ##############################################################################################################
        # if base_ranks[i] > (ref_ranks[i] + cfg.tau):
        S_im_greater = torch.cat([-S_matrix[:ref_ranks[i], base_idx[i]], S_matrix[base_ranks[i]:, base_idx[i]]])
        S_in_greater = torch.cat([S_matrix[:ref_ranks[i], ref_idx[i]], -S_matrix[base_ranks[i]:, ref_idx[i]]])
        greater_violation = nn.functional.relu(S_im_greater + S_in_greater + cfg.margin)

        loss += prob_greater[i] * torch.sum(greater_violation)
        ##############################################################################################################
        # elif base_ranks[i] < (ref_ranks[i] - cfg.tau):
        S_im_less = torch.cat([S_matrix[:base_ranks[i], base_idx[i]], -S_matrix[ref_ranks[i]:, base_idx[i]]])
        S_in_less = torch.cat([-S_matrix[:base_ranks[i], ref_idx[i]], S_matrix[ref_ranks[i]:, ref_idx[i]]])

        less_violation = nn.functional.relu(S_im_less + S_in_less + cfg.margin)
        loss += prob_less[i] * torch.sum(less_violation)
        ##############################################################################################################
        # if similar ranks
        # else:
        S_im_sim = S_matrix[:, base_idx[i]]
        S_in_sim = S_matrix[:, ref_idx[i]]

        sim_violation = nn.functional.relu(abs(S_im_sim - S_in_sim) - cfg.margin)
        loss += prob_same[i] * torch.sum(sim_violation)
        ##############################################################################################################

    return loss / batch_size


def compute_discriminative_loss(embs, centroids, prob_dbs, cfg, record=False):
    if cfg.norm_centroids:
        centroids = nn.functional.normalize(centroids, dim=-1)

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric == 'L2_squared':
        dists = torch.square(torch.cdist(centroids, embs))
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    expectation = prob_dbs.argmax(axis=1)
    L = cfg.L
    loss = 0
    total_rank_num = centroids.shape[0]
    max_rank = total_rank_num - 1

    if cfg.change_std:
        all_prob_dbs = torch.cat(
            [get_prob_db_tensor(len(centroids), m, cfg.prob_std[idx]).unsqueeze(0) for idx, m in
             enumerate(range(len(centroids)))]).to(torch.float32).to(cfg.device)
    else:
        all_prob_dbs = torch.cat(
            [get_prob_db_tensor(len(centroids), m, cfg.prob_std).unsqueeze(0) for m in range(len(centroids))]).to(
            torch.float32).to(cfg.device)  # i: mean
    S_matrix = torch.matmul(all_prob_dbs, dists)
    S_0 = S_matrix[expectation, torch.arange(len(embs))]

    indices = torch.arange(len(embs)).to(cfg.device)
    for l in range(1, L + 1):
        plus_side_idx = (expectation > max_rank - l)
        minus_side_idx = (expectation < l)
        valid_idxs = torch.logical_and(expectation >= l, expectation <= max_rank - l)

        if (plus_side_idx).sum() != 0:
            S_minus_l = S_matrix[expectation[plus_side_idx] - l, indices[plus_side_idx]]
            loss += 2 * (S_0[plus_side_idx] - S_minus_l).sum()

        if (minus_side_idx).sum() != 0:
            S_plus_l = S_matrix[expectation[minus_side_idx] + l, indices[minus_side_idx]]
            loss += 2 * (S_0[minus_side_idx] - S_plus_l).sum()

        S_plus_valid_l = S_matrix[expectation[valid_idxs] + l, indices[valid_idxs]]
        S_minus_valid_l = S_matrix[expectation[valid_idxs] - l, indices[valid_idxs]]
        loss += ((S_0[valid_idxs] - S_plus_valid_l) + (S_0[valid_idxs] - S_minus_valid_l)).sum()

    return loss / len(embs)



def compute_stochastic_center_loss(embs, centroids, prob_dbs, cfg, record=False):
    centroids = nn.functional.normalize(centroids, dim=-1)

    if cfg.metric == 'L2':
        dists = torch.cdist(embs, centroids)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(embs.transpose(1, 0), centroids)

    loss = (dists * prob_dbs).sum(dim=1)

    if record:
        return torch.sum(loss), to_np(loss)
    else:
        return torch.sum(loss)
    
def gaussian_nll_loss(mu, sigma, target):
    # sigma: [B], mu: [B]
    var = sigma**2 + 1e-12

    return torch.mean(
        0.5 * ( (target - mu)**2 / var +  torch.log(var) )
    )

def compute_stochastic_order_loss_adaptiveSigma(embs, base_idx, ref_idx, rank_labels, prob_dbs, sigmas,
                             centroids, margin, cfg, record=False):
    if cfg.norm_centroids:
        centroids = nn.functional.normalize(centroids, dim=-1)
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (len(centroids) - 1)) * i for i in range(len(centroids))])

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric =='L2_squared':
        dists = torch.square(torch.cdist(centroids, embs))
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    batch_size = len(base_idx)
    base_ranks = rank_labels[base_idx]
    ref_ranks = rank_labels[ref_idx]

    # all_prob_dbs = torch.cat(
    #     [get_prob_db_tensor(len(centroids), m, cfg.prob_std).unsqueeze(0) for m in range(len(centroids))]).to(
    #     torch.float32).to(cfg.device)  # i: mean
    all_prob_dbs = torch.stack([
    torch.stack([
        get_prob_db_tensor(len(centroids), mean=m, prob_std=float(sigmas[b]))
        for m in range(len(centroids))
    ])
    for b in range(dists.shape[-1])]).to(torch.float32).to(cfg.device)    # shape: [B, C, C]
    S_matrix = torch.matmul(all_prob_dbs, dists)  
    
    #########################################################################################################################
    prob_dbs = prob_dbs.to(torch.float32)
    base_prob_dbs = prob_dbs[base_idx]
    ref_prob_dbs = prob_dbs[ref_idx]

    # Compute probability
    prob_same = torch.zeros(base_idx.shape).to(torch.float32).to(cfg.device)
    prob_less, prob_greater = torch.zeros_like(prob_same), torch.zeros_like(prob_same)
    for j in range(len(fdc_point_ranks)):
        min_sim, max_sim = max(0, j - cfg.tau), min(len(centroids) - 1, j + cfg.tau)
        prob_same += (base_prob_dbs[:, j].unsqueeze(-1) * ref_prob_dbs[:, min_sim:max_sim + 1]).sum(axis=1)

        min_less = min(len(centroids) - 1, j + cfg.tau + 1)
        prob_less += (base_prob_dbs[:, j].unsqueeze(-1) * ref_prob_dbs[:, min_less:]).sum(axis=1)
        prob_greater += (base_prob_dbs[:, min_less:] * ref_prob_dbs[:, j].unsqueeze(-1)).sum(axis=1)
    ##########################################################################################################################

    loss = 0.
    for i in range(batch_size):
        ##############################################################################################################
        # if base_ranks[i] > (ref_ranks[i] + cfg.tau):
        S_im_greater = torch.cat([-S_matrix[base_idx[i], :ref_ranks[i], base_idx[i]], S_matrix[base_idx[i], base_ranks[i]:, base_idx[i]]])
        S_in_greater = torch.cat([S_matrix[ref_idx[i], :ref_ranks[i], ref_idx[i]], -S_matrix[ref_idx[i], base_ranks[i]:, ref_idx[i]]])
        greater_violation = nn.functional.relu(S_im_greater + S_in_greater + cfg.margin)

        loss += prob_greater[i] * torch.sum(greater_violation)
        ##############################################################################################################
        # elif base_ranks[i] < (ref_ranks[i] - cfg.tau):
        S_im_less = torch.cat([S_matrix[base_idx[i],:base_ranks[i], base_idx[i]], -S_matrix[base_idx[i],ref_ranks[i]:, base_idx[i]]])
        S_in_less = torch.cat([-S_matrix[ref_idx[i], :base_ranks[i], ref_idx[i]], S_matrix[ref_idx[i], ref_ranks[i]:, ref_idx[i]]])

        less_violation = nn.functional.relu(S_im_less + S_in_less + cfg.margin)
        loss += prob_less[i] * torch.sum(less_violation)
        ##############################################################################################################
        # if similar ranks
        # else:
        S_im_sim = S_matrix[base_idx[i],:, base_idx[i]]
        S_in_sim = S_matrix[ref_idx[i],:, ref_idx[i]]

        sim_violation = nn.functional.relu(abs(S_im_sim - S_in_sim) - cfg.margin)
        loss += prob_same[i] * torch.sum(sim_violation)
        ##############################################################################################################

    return loss / batch_size


def compute_discriminative_loss_adaptiveSigma(embs, centroids, prob_dbs, sigmas, cfg, record=False):
    if cfg.norm_centroids:
        centroids = nn.functional.normalize(centroids, dim=-1)

    if cfg.metric == 'L2':
        dists = torch.cdist(centroids, embs)
    elif cfg.metric == 'L2_squared':
        dists = torch.square(torch.cdist(centroids, embs))
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(centroids, embs.transpose(1, 0))

    expectation = prob_dbs.argmax(axis=1)
    L = cfg.L
    loss = 0
    total_rank_num = centroids.shape[0]
    max_rank = total_rank_num - 1

    # if cfg.change_std:
    #     all_prob_dbs = torch.cat(
    #         [get_prob_db_tensor(len(centroids), m, cfg.prob_std[idx]).unsqueeze(0) for idx, m in
    #          enumerate(range(len(centroids)))]).to(torch.float32).to(cfg.device)
    # else:
    #     all_prob_dbs = torch.cat(
    #         [get_prob_db_tensor(len(centroids), m, cfg.prob_std).unsqueeze(0) for m in range(len(centroids))]).to(
    #         torch.float32).to(cfg.device)  # i: mean

    all_prob_dbs = torch.stack([
    torch.stack([
        get_prob_db_tensor(len(centroids), mean=m, prob_std=float(sigmas[b]))
        for m in range(len(centroids))
    ])
    for b in range(dists.shape[-1])]).to(torch.float32).to(cfg.device)    # shape: [B, C, C]

    S_matrix = torch.matmul(all_prob_dbs, dists)
    # S_0 = S_matrix[expectation, torch.arange(len(embs))]
    S_0 = S_matrix[torch.arange(len(embs)), expectation, torch.arange(len(embs))]
    indices = torch.arange(len(embs)).to(cfg.device)
    for l in range(1, L + 1):
        plus_side_idx = (expectation > max_rank - l)
        minus_side_idx = (expectation < l)
        valid_idxs = torch.logical_and(expectation >= l, expectation <= max_rank - l)

        if (plus_side_idx).sum() != 0:
            S_minus_l = S_matrix[indices[plus_side_idx], expectation[plus_side_idx] - l, indices[plus_side_idx]]
            loss += 2 * (S_0[plus_side_idx] - S_minus_l).sum()

        if (minus_side_idx).sum() != 0:
            S_plus_l = S_matrix[indices[minus_side_idx], expectation[minus_side_idx] + l, indices[minus_side_idx]]
            loss += 2 * (S_0[minus_side_idx] - S_plus_l).sum()

        S_plus_valid_l = S_matrix[indices[valid_idxs], expectation[valid_idxs] + l, indices[valid_idxs]]
        S_minus_valid_l = S_matrix[indices[valid_idxs], expectation[valid_idxs] - l, indices[valid_idxs]]
        loss += ((S_0[valid_idxs] - S_plus_valid_l) + (S_0[valid_idxs] - S_minus_valid_l)).sum()

    return loss / len(embs)
# def compute_stochastic_order_loss_adaptiveSigma(embs, base_idx, ref_idx, rank_labels, prob_dbs,
#                                   centroids, margin, cfg, record=False):
#     """
#     prob_dbs : [B, C, C]
#         prob_dbs[b,l,c] = probability distribution for instance b
#                           over centroid 'c' given latent rank 'l'.

#     centroids : [C, D]
#     embs      : [B, D]
#     """

#     # ============================
#     # Normalize centroids
#     # ============================
#     if cfg.norm_centroids:
#         centroids = nn.functional.normalize(centroids, dim=-1)

#     # ============================
#     # Compute distances (B, C)
#     # ============================
#     if cfg.metric == 'L2':
#         dists = torch.cdist(embs, centroids)          # [B,C]
#     elif cfg.metric == 'L2_squared':
#         dists = torch.square(torch.cdist(embs, centroids))
#     elif cfg.metric == 'cosine':
#         dists = 1 - embs @ centroids.T

#     B, C = dists.shape

#     # ============================
#     # Instance-wise S_matrix
#     # S[b,l] = Σ_c prob_dbs[b,l,c] * dist[b,c]
#     # ============================
#     S_matrix = (prob_dbs * dists.unsqueeze(1)).sum(dim=-1)    # [B,C]

#     # ============================
#     # Select pairs
#     # ============================
#     base_ranks = rank_labels[base_idx]   # numpy int
#     ref_ranks  = rank_labels[ref_idx]    # numpy int

#     # ============================
#     # Extract prob_dbs for pairs
#     # prob_dbs: [B, C, C]
#     # ============================
#     base_prob = prob_dbs[base_idx]       # [Mb, C, C]
#     ref_prob  = prob_dbs[ref_idx]        # [Mb, C, C]

#     # ============================
#     # Precompute probability for each case
#     # ============================
#     prob_same    = torch.zeros(len(base_idx), device=embs.device)
#     prob_less    = torch.zeros_like(prob_same)
#     prob_greater = torch.zeros_like(prob_same)

#     for l in range(C):
#         # j = l
#         min_sim = max(0, l - cfg.tau)
#         max_sim = min(C - 1, l + cfg.tau)

#         # SAME
#         prob_same += (base_prob[:,l,min_sim:max_sim+1] *
#                       torch.ones_like(ref_prob[:,l,min_sim:max_sim+1])).sum(dim=1)

#         # LESS: base > ref
#         min_less = min(C - 1, l + cfg.tau + 1)
#         if min_less < C:
#             prob_less += (base_prob[:,l,min_less:] * torch.ones_like(ref_prob[:,l,min_less:])).sum(dim=1)

#         # GREATER: base < ref
#         prob_greater += (base_prob[:,min_less:,l].sum(dim=1)
#                          if min_less < C else 0.)

#     # ============================
#     # Compute Loss
#     # ============================
#     loss = 0.0
#     Mb = len(base_idx)

#     for i in range(Mb):
#         br = base_ranks[i]
#         rr = ref_ranks[i]

#         # ------------------ Case 1: base > ref + tau
#         if br > rr + cfg.tau:
#             S_im_greater = torch.cat([-S_matrix[base_idx[i], :rr],
#                                        S_matrix[base_idx[i], br:]])
#             S_in_greater = torch.cat([ S_matrix[ref_idx[i], :rr],
#                                       -S_matrix[ref_idx[i], br:]])
#             violation = nn.functional.relu(S_im_greater + S_in_greater + margin)
#             loss += prob_greater[i] * violation.sum()

#         # ------------------ Case 2: base < ref − tau
#         elif br < rr - cfg.tau:
#             S_im_less = torch.cat([ S_matrix[base_idx[i], :br],
#                                    -S_matrix[base_idx[i], rr:]])
#             S_in_less = torch.cat([-S_matrix[ref_idx[i], :br],
#                                     S_matrix[ref_idx[i], rr:]])
#             violation = nn.functional.relu(S_im_less + S_in_less + margin)
#             loss += prob_less[i] * violation.sum()

#         # ------------------ Case 3: similar ranks
#         else:
#             S_im_sim = S_matrix[base_idx[i]]
#             S_in_sim = S_matrix[ref_idx[i]]
#             violation = nn.functional.relu(torch.abs(S_im_sim - S_in_sim) - margin)
#             loss += prob_same[i] * violation.sum()

#     return loss / Mb


# def compute_discriminative_loss_adaptiveSigma(embs, centroids, prob_dbs, cfg, record=False):
#     """
#     embs: [B,D]
#     centroids: [C,D]
#     prob_dbs: [B,C,C]
#     """
#     if cfg.norm_centroids:
#         centroids = nn.functional.normalize(centroids, dim=-1)

#     # distances: [B,C]
#     if cfg.metric == 'L2':
#         dists = torch.cdist(embs, centroids)
#     elif cfg.metric == 'L2_squared':
#         dists = torch.square(torch.cdist(embs, centroids))
#     elif cfg.metric == 'cosine':
#         dists = 1 - embs @ centroids.T

#     B, C = dists.shape
#     L = cfg.L
#     device = embs.device

#     # Compute S matrix: [B,C]
#     S_matrix = (prob_dbs * dists.unsqueeze(1)).sum(dim=-1)   # [B,C]

#     # expectation (latent best rank)
#     expectation = torch.argmin(S_matrix, dim=1)  # [B]

#     indices = torch.arange(B, device=device)

#     S0 = S_matrix[indices, expectation]   # [B]
#     loss = 0.0
#     max_rank = C - 1

#     for l in range(1, L + 1):
#         # Left boundary (cannot go below 0)
#         minus_side = (expectation >= l)
#         plus_side  = (expectation <= max_rank - l)

#         # right side: expectation + l
#         if plus_side.any():
#             S_plus = S_matrix[indices[plus_side], expectation[plus_side] + l]
#             loss += (S0[plus_side] - S_plus).sum()

#         # left side: expectation - l
#         if minus_side.any():
#             S_minus = S_matrix[indices[minus_side], expectation[minus_side] - l]
#             loss += (S0[minus_side] - S_minus).sum()

#     return loss / B
