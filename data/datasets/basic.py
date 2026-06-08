import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from utils.util import load_one_image, to_np, get_prob_db_tensor, get_prob_db_tensor_V2

class Basic(Dataset):
    def __init__(self, imgs, labels, transform, noised=False, noised_labels=None, norm_age=True, is_filelist=False, return_ranks=False, std=None, prob_dbs=None, prob_std=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.labels = labels
        self.noised = noised
        self.noised_labels = noised_labels

        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if norm_age:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks
        self.std = std

        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

        ###############################################
        # self.prob_std = prob_std
        # if prob_dbs is None:
        #     self.prob_dbs = []
        #     for mean in self.ranks:
        #         pdf = to_np(get_prob_db_tensor(len(np.unique(self.ranks)), mean, self.prob_std))
        #         self.prob_dbs.append(pdf)

        #     self.prob_dbs = np.array(self.prob_dbs)
        # else:
        #     self.prob_dbs = prob_dbs
        ###########################################################3


    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)

        ##################################3
        # prob_db = self.prob_dbs[item]
        #####################################
        sample = {'base_img': img, 'real_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item}
        # sample = {'base_img': img, 'real_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'prob_db':prob_db}
        return sample
        # if self.return_ranks:
        #     return img, self.labels[item], self.ranks[item], item
        # else:
        #     return img, self.labels[item], item

    def __len__(self):
        return len(self.imgs)

class Basic_Noised_Stochastic(Dataset):
    def __init__(self, imgs, noised_labels, real_labels, gt_ranks=None, real_gt_ranks=None, transform=None, norm_age=True, is_filelist=False,
                 return_ranks=False, std=None, prob_dbs=None, prob_std=1, n_ranks=None, mappings=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.labels = noised_labels
        self.real_labels = real_labels

        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if norm_age:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks

        if mappings == None:
            rank = 0
            self.mapping = dict()
            for cls in np.unique(self.labels):
                self.mapping[cls] = rank
                rank += 1
        else:
            self.mapping = mappings
        self.ranks = np.array([self.mapping[l] for l in self.labels])
        ##################################################################3
        # if mappings == None:
        #     rank = 0
        #     self.mapping = dict()
        #     for cls in range(self.labels.min(), self.labels.max()+1, 1):
        #         self.mapping[cls] = rank
        #         rank += 1
        # else:
        #     self.mapping = mappings
        # self.ranks = np.array([self.mapping[l] for l in self.labels])
        ##################################################################3

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

        if gt_ranks is None:
            self.gt_ranks = self.ranks
        else:
            self.gt_ranks = gt_ranks


    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')

        img = self.transform(img)

        #################
        prob_db = self.prob_dbs[item]
        gt_rank = self.gt_ranks[item]

        if isinstance(self.prob_std, int):
            prob_db_std = self.prob_std

        elif isinstance(self.prob_std, float):
            prob_db_std = self.prob_std
        else:
            prob_db_std = self.prob_std[item]

        # real_gt_rank = self.real_gt_ranks[item]

        # sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'prob_db':prob_db, 'gt_rank':gt_rank, 'real_gt_rank': real_gt_rank}

        sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'gt_rank':gt_rank, 'prob_db':prob_db, 'prob_db_std':prob_db_std}
        return sample

    def __len__(self):
        return len(self.imgs)


# class Basic_Noised_Stochastic(Dataset):
#     def __init__(self, imgs, noised_labels, real_labels, gt_ranks=None, real_gt_ranks=None, transform=None, norm_age=True, is_filelist=False,
#                  return_ranks=False, std=None, prob_dbs=None, prob_std=1, n_ranks=None, mappings=None):
#         super(Dataset, self).__init__()
#         self.transform = transform
#         self.imgs = imgs
#         self.labels = noised_labels
#         self.real_labels = real_labels

#         self.n_imgs = len(self.imgs)
#         self.is_filelist = is_filelist
#         if norm_age:
#             self.labels = self.labels - min(self.labels)
#         self.return_ranks = return_ranks

#         if mappings == None:
#             rank = 0
#             self.mapping = dict()
#             for cls in np.unique(self.labels):
#                 self.mapping[cls] = rank
#                 rank += 1
#         else:
#             self.mapping = mappings
#         self.ranks = np.array([self.mapping[l] for l in self.labels])
#         ##################################################################3
#         if mappings == None:
#             rank = 0
#             self.mapping = dict()
#             for cls in range(self.labels.min(), self.labels.max()+1, 1):
#                 self.mapping[cls] = rank
#                 rank += 1
#         else:
#             self.mapping = mappings
#         self.ranks = np.array([self.mapping[l] for l in self.labels])
#         ##################################################################3

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

#         if gt_ranks is None:
#             self.gt_ranks = self.ranks
#         else:
#             self.gt_ranks = gt_ranks


#     def __getitem__(self, item):
#         if self.is_filelist:
#             img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
#         else:
#             img = np.asarray(self.imgs[item]).astype('uint8')

#         img = self.transform(img)

#         #################
#         prob_db = self.prob_dbs[item]
#         gt_rank = self.gt_ranks[item]

#         if isinstance(self.prob_std, int):
#             prob_db_std = self.prob_std
#         elif isinstance(self.prob_std, float):
#             prob_db_std = self.prob_std
#         else:
#             prob_db_std = self.prob_std[item]

#         # real_gt_rank = self.real_gt_ranks[item]

#         # sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'prob_db':prob_db, 'gt_rank':gt_rank, 'real_gt_rank': real_gt_rank}

#         sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'gt_rank':gt_rank, 'prob_db':prob_db, 'prob_db_std':prob_db_std}
#         return sample

#     def __len__(self):
#         return len(self.imgs)


class Basic_Noised_Stochastic_allRanks(Dataset):
    def __init__(self, imgs, noised_labels, real_labels, gt_ranks=None, real_gt_ranks=None, transform=None, norm_age=True, is_filelist=False,
                 return_ranks=False, std=None, prob_dbs=None, prob_std=None, n_ranks=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.labels = noised_labels
        self.real_labels = real_labels

        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if norm_age:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks

        rank = 0
        self.mapping = dict()

        for cls in range(self.labels.min(), self.labels.max()+1):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])

        if gt_ranks is None:
            self.gt_ranks = self.ranks
        else:
            self.gt_ranks = gt_ranks

        self.prob_std = prob_std

        self.prob_dbs = []
        if isinstance(prob_std, int):
            for mean in self.ranks:
                pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std))
                self.prob_dbs.append(pdf)
        else:
            for i, mean in enumerate(self.ranks):
                pdf = to_np(get_prob_db_tensor(n_ranks, mean, self.prob_std[i].item()))
                self.prob_dbs.append(pdf)
        self.prob_dbs = np.array(self.prob_dbs)

    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')

        img = self.transform(img)

        #################
        prob_db = self.prob_dbs[item]
        gt_rank = self.gt_ranks[item]
        # real_gt_rank = self.real_gt_ranks[item]

        # sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'prob_db':prob_db, 'gt_rank':gt_rank, 'real_gt_rank': real_gt_rank}

        sample = {'base_img': img, 'real_base_age': self.real_labels[item], 'noised_base_age': self.labels[item], 'ranks': self.ranks[item], 'index': item, 'prob_db':prob_db, 'gt_rank':gt_rank}
        return sample

    def __len__(self):
        return len(self.imgs)
