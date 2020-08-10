# encoding: utf-8
"""
@author:  eugene ang
@contact: phuaywee001@e.ntu.edu.sg
"""

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedCameraTripletSampler(Sampler):
    """
    Randomly sample C camera ids, then for each camera id,
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_cameras):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_cameras = num_cameras

        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.cam_dic = {}
        # self.index_dic = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            if camid not in self.cam_dic:
                self.cam_dic[camid] = {'indices':defaultdict(list)}
            self.cam_dic[camid]['indices'][pid].append(index)
        self.total_cams = len(self.cam_dic)

        # estimate number of examples in an epoch
        self.length = 0
        self.cam_counts = np.zeros( self.total_cams) )
        for camid in self.cam_dic.keys():
            self.cam_counts[camid] = len(self.cam_dic[camid]['indices'])
            pids = list(self.cam_dic[camid]['indices'].keys())
            self.cam_dic[camid]['pids'] = pids

            for pid in pids:
                idxs = self.cam_dic[camid]['indices'][pid]
                num = len(idxs)
                if num < self.num_instances:
                    num = self.num_instances
                self.length += num - num % self.num_instances
        
        # compute sampling weights of cameras based on frequency
        self.cam_probs = self.cam_counts / sum(self.cam_counts)
    
    def sample_for_triplet_loss(self, pids, indices):
        batch_idxs_dict = defaultdict(list)

        for pid in pids:
            idxs = copy.deepcopy(indices[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __iter__(self):
        # Sample cameras
        cams = np.random.choice(self.total_cams, size=self.num_cameras, replace=True, p=self.cam_probs)

        # Merge the pids and indices from the selected cameras into a large one
        pids = [] # all the relevant person-ids to sample from
        indices = defaultdict(list) # mapping from person-ids to indices in the master data. 
        # (This must exclude the indices of cameras that were not selected)

        # Collect all pids observed in chosen cameras.
        # Collect all indices of observed pids within the chosen cameras.
        for cam in cams:
            cam_pids = copy.deepcopy(self.cam_dic[cam]['pids'])
            pids.extend( cam_pids )
            for pid in cam_pids:
                indices[pid].extend( copy.deepcopy(self.cam_dic[cam]['indices'][pid] )
        # Remove duplicates for both the pid list, since there are bound to be pids seen in multiple cameras
        pids = list(set(pids))
        
        # There should not be a need for duplicate removal for the indices data structure.
        # Because, indices are unique with respect to camera-id and person-id. You will not get
        # the same index in a different camera. TODO: make sure this reasoning is correct.
        # for pid in pids:
        #     idxs = indices[pid]
        #     indices[pid] = list(set(idxs))

        # Sub-contract the work back to ye olde triplet sampler
        return self.sample_for_triplet_loss( pids, indices )

    def __len__(self):
        return self.length
