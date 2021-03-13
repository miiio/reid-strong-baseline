# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        print(self.index_dic)

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

# New add by Lao
class RandomIdentitySamplerIntraCamera(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_ids, num_camera):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = num_ids #self.batch_size // self.num_instances
        self.camids = list(range(num_camera))
        self.index_dic = {}
        self.pids = {}
        for camid in self.camids:
            self.index_dic[camid] = defaultdict(list)   #self.index_dic = defaultdict(list)
            self.pids[camid] = defaultdict(list)    #self.pids = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[camid][pid].append(index)
        for camid in self.camids:
            self.pids[camid] = list(self.index_dic[camid].keys())

        # estimate number of examples in an epoch
        self.length = batch_size * num_instances * num_ids * num_camera
        # self.length = 0
        # for pid in self.pids:
        #     idxs = self.index_dic[pid]
        #     num = len(idxs)
        #     if num < self.num_instances:
        #         num = self.num_instances
        #     self.length += num - num % self.num_instances
    def getPidsNum(self):
        return [len(self.pids[camid]) for camid in self.camids]


    def __iter__(self):
        batch_idxs_dict = {}
        for camid in self.camids:
            batch_idxs_dict[camid] = defaultdict(list)
        for camid in self.camids:
            for pid in self.pids[camid]:
                idxs = copy.deepcopy(self.index_dic[camid][pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[camid][pid].append(batch_idxs)
                        batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while False not in [len(avai_pids[camid])>=self.num_pids_per_batch for camid in self.camids]: #while len(avai_pids) >= self.num_pids_per_batch:
            for camid in self.camids:
                selected_pids = random.sample(avai_pids[camid], self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[camid][pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[camid][pid]) == 0:
                        avai_pids[camid].remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length