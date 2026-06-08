import scipy
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.util import load_one_image, to_np, get_prob_db_tensor, get_prob_db_tensor_V2
import random




class OLBasic_Train(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, logscale=False, is_filelist=False, std=None, prob_std=None):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
        if logscale:
            self.labels = np.log(labels.astype(np.float32))
        else:
            if norm_age:
                self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        # mapping age to rank : because there are omitted ages
        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

        ###########################################################################################################
        self.prob_std = prob_std
        self.prob_dbs = []
        for mean in self.ranks:
            pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
            self.prob_dbs.append(pdf)
        self.prob_dbs = np.array(self.prob_dbs)
        ###########################################################################################################

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        ##################################################
        # probability distributions
        base_prob_db = self.prob_dbs[item]
        ref_prob_db = self.prob_dbs[ref_idx]
        #################################################3

        sample = {'base_img': base_img, 'ref_img': ref_img, 'order_label': order_label, 'ranks':[base_rank, ref_rank], 'prob_db':[base_prob_db, ref_prob_db], 'index': item}

        return sample
        # return base_img, ref_img, order_label, [base_rank, ref_rank], item

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        rng = np.random.default_rng(seed=10)
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx


class OLBasic_Train_V2(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, logscale=False, is_filelist=False, std=None, prob_std=None):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
        if logscale:
            self.labels = np.log(labels.astype(np.float32))
        else:
            if norm_age:
                self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        # mapping age to rank : because there are omitted ages
        rank = 0
        self.mapping = dict()
        # for cls in np.unique(self.labels):
        #     self.mapping[cls] = rank
        #     rank += 1
        for cls in range(self.labels.min(), self.labels.max()+1):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

        ###########################################################################################################
        self.prob_std = prob_std
        self.prob_dbs = []
        for mean in self.ranks:
            pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
            self.prob_dbs.append(pdf)
        self.prob_dbs = np.array(self.prob_dbs)
        ###########################################################################################################

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        ##################################################
        # probability distributions
        base_prob_db = self.prob_dbs[item]
        ref_prob_db = self.prob_dbs[ref_idx]
        #################################################3

        sample = {'base_img': base_img, 'ref_img': ref_img, 'order_label': order_label, 'ranks':[base_rank, ref_rank], 'prob_db':[base_prob_db, ref_prob_db], 'index': item}

        return sample
        # return base_img, ref_img, order_label, [base_rank, ref_rank], item

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        rng = np.random.default_rng(seed=10)
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx


class OLBasic_Train_Noised_Stochastic(Dataset):
    def __init__(self, imgs, noised_labels, real_labels, transform, tau, norm_age=True, logscale=False, is_filelist=False, std=None, prob_std=1, n_ranks=None):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = noised_labels
        self.real_labels = real_labels

        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
        self.min_age_bf_norm_real = self.real_labels.min()

        if logscale:
            self.labels = np.log(noised_labels.astype(np.float32))
            self.real_labels = np.log(real_labels.astype(np.float32))
        else:
            if norm_age:
                self.labels = self.labels - min(self.labels)
                self.real_labels = self.real_labels - min(self.real_labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        # mapping age to rank : because there are omitted ages
        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

        self.max_age_real = self.labels.max()
        self.min_age_real = self.labels.min()

        # rank_real = 0
        # self.mapping_real = dict()
        # for cls in np.unique(self.real_labels):
        #     self.mapping_real[cls] = rank_real
        #     rank_real += 1
        # self.ranks_real = np.array([self.mapping_real[l] for l in self.real_labels])

        #######################################################################################################################
        self.prob_std = prob_std
        self.prob_dbs = []

        if isinstance(prob_std, int):
            for mean in self.ranks:
                pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
                # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std))
                self.prob_dbs.append(pdf)

        elif isinstance(prob_std, float):
            for mean in self.ranks:
                pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
                # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std))
                self.prob_dbs.append(pdf)
        else:
            for i, mean in enumerate(self.ranks):
                pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std[i]))
                # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std[i].item()))
                self.prob_dbs.append(pdf)
        self.prob_dbs = np.array(self.prob_dbs)


    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')

        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]

        real_base_age = self.real_labels[item]
        real_ref_age = self.real_labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        # # probability distributions
        base_prob_db = self.prob_dbs[item]
        ref_prob_db = self.prob_dbs[ref_idx]
        if isinstance(self.prob_std, int):
            base_prob_db_std = self.prob_std
            ref_prob_db_std = self.prob_std
        elif isinstance(self.prob_std, float):
            base_prob_db_std = self.prob_std
            ref_prob_db_std = self.prob_std
        else:
            base_prob_db_std = self.prob_std[item]
            ref_prob_db_std = self.prob_std[ref_idx]


        sample = {'base_img': base_img, 'ref_img': ref_img, 'order_label': order_label, 'ranks':[base_rank, ref_rank], 'labels':[base_age, ref_age], 'index': item,
                  'noised_base_age': base_age, 'noised_ref_age': ref_age, 'real_base_age': real_base_age, 'real_ref_age': real_ref_age, 'prob_db':[base_prob_db, ref_prob_db], 'prob_db_std':[base_prob_db_std, ref_prob_db_std]}

        return sample

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        # rng = np.random.default_rng(seed=10)
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx

# class OLBasic_Train_Noised_Stochastic(Dataset):
#     def __init__(self, imgs, noised_labels, real_labels, transform, tau, norm_age=True, logscale=False, is_filelist=False, std=None, prob_std=1, n_ranks=None):
#         super(Dataset, self).__init__()
#         self.imgs = imgs
#         self.labels = noised_labels
#         self.real_labels = real_labels

#         self.transform = transform
#         self.n_imgs = len(self.imgs)
#         self.min_age_bf_norm = self.labels.min()
#         self.min_age_bf_norm_real = self.real_labels.min()

#         if logscale:
#             self.labels = np.log(noised_labels.astype(np.float32))
#             self.real_labels = np.log(real_labels.astype(np.float32))
#         else:
#             if norm_age:
#                 self.labels = self.labels - min(self.labels)
#                 self.real_labels = self.real_labels - min(self.real_labels)

#         self.max_age = self.labels.max()
#         self.min_age = self.labels.min()
#         self.tau = tau
#         self.is_filelist = is_filelist

#         # mapping age to rank : because there are omitted ages
#         rank = 0
#         self.mapping = dict()
#         for cls in np.unique(self.labels):
#             self.mapping[cls] = rank
#             rank += 1
#         self.ranks = np.array([self.mapping[l] for l in self.labels])

#         self.max_age_real = self.labels.max()
#         self.min_age_real = self.labels.min()

#         # rank_real = 0
#         # self.mapping_real = dict()
#         # for cls in np.unique(self.real_labels):z
#         #     self.mapping_real[cls] = rank_real
#         #     rank_real += 1
#         # self.ranks_real = np.array([self.mapping_real[l] for l in self.real_labels])

#         #######################################################################################################################
#         self.prob_std = prob_std
#         self.prob_dbs = []

#         if isinstance(prob_std, int):
#             for mean in self.ranks:
#                 pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
#                 # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std))
#                 self.prob_dbs.append(pdf)
#         elif isinstance(prob_std, float):
#             for mean in self.ranks:
#                 pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
#                 # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std))
#                 self.prob_dbs.append(pdf)
#         else:
#             for i, mean in enumerate(self.ranks):
#                 pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std[i]))
#                 # pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std[i].item()))
#                 self.prob_dbs.append(pdf)
#         self.prob_dbs = np.array(self.prob_dbs)


#     def __getitem__(self, item):
#         order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
#                                                    max_rank=self.max_age)
#         if self.is_filelist:
#             base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
#             ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
#         else:
#             base_img = np.asarray(self.imgs[item]).astype('uint8')
#             ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')

#         base_img = self.transform(base_img)
#         ref_img = self.transform(ref_img)

#         base_age = self.labels[item]
#         ref_age = self.labels[ref_idx]

#         real_base_age = self.real_labels[item]
#         real_ref_age = self.real_labels[ref_idx]

#         # gt ranks
#         base_rank = self.ranks[item]
#         ref_rank = self.ranks[ref_idx]

#         # # probability distributions
#         base_prob_db = self.prob_dbs[item]
#         ref_prob_db = self.prob_dbs[ref_idx]
#         if isinstance(self.prob_std, int):
#             base_prob_db_std = self.prob_std
#             ref_prob_db_std = self.prob_std
#         elif isinstance(self.prob_std, float):
#             base_prob_db_std = self.prob_std
#             ref_prob_db_std = self.prob_std
#         else:
#             base_prob_db_std = self.prob_std[item]
#             ref_prob_db_std = self.prob_std[ref_idx]


#         sample = {'base_img': base_img, 'ref_img': ref_img, 'order_label': order_label, 'ranks':[base_rank, ref_rank], 'labels':[base_age, ref_age], 'index': item,
#                   'noised_base_age': base_age, 'noised_ref_age': ref_age, 'real_base_age': real_base_age, 'real_ref_age': real_ref_age, 'prob_db':[base_prob_db, ref_prob_db], 'prob_db_std':[base_prob_db_std, ref_prob_db_std]}

#         return sample

#     def __len__(self):
#         return self.n_imgs

#     def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

#         def get_indices_in_range(search_range, ages):
#             """find indices of values within range[0] <= x <= range[1]"""
#             return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

#         # rng = np.random.default_rng(seed=10)
#         order = np.random.randint(0, 3)
#         ref_idx = -1
#         debug_flag = 0
#         while ref_idx == -1:
#             if debug_flag == 3:
#                 raise ValueError(f'Failed to find reference... base_score: {base_rank}')
#             if order == 0:  # base_rank > ref_rank + tau
#                 ref_range_min = min_rank
#                 ref_range_max = base_rank - self.tau - epsilon
#                 candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
#                 if len(candidates) > 0:
#                     # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
#                     ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
#                 else:
#                     order = (order + 1) % 3
#                     debug_flag += 1
#                     continue
#             elif order == 1:  # base_rank < ref_rank - tau
#                 ref_range_min = base_rank + self.tau + epsilon
#                 ref_range_max = max_rank
#                 candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
#                 if len(candidates) > 0:
#                     # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
#                     ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
#                 else:
#                     order = (order + 1) % 3
#                     debug_flag += 1
#                     continue

#             else:  # |base_rank - ref_rank| <= tau
#                 ref_range_min = base_rank - self.tau - epsilon
#                 ref_range_max = base_rank + self.tau + epsilon
#                 candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
#                 if len(candidates) > 0:
#                     # ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
#                     ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
#                 else:
#                     order = (order + 1) % 3
#                     debug_flag += 1
#         return order, ref_idx
