# encoding: utf-8
from .BaseDataset import BaseImageDataset
from collections import OrderedDict
import numpy as np


class Combined(BaseImageDataset):
    def __init__(self, datasets, merge=False, verbose=True, **kwargs):
        """
        Assume:
            1. No negetive ids;
            2. The new arragement of ids will be (for both camid and pid):
                [ D1:train D2:train ... D1:others D2:others ... ]
            3. Train camids do not intersect with others' camids

        If merge, merge all train,query,gallery into one train.
        """
        super(Combined, self).__init__()
        self.datasets = OrderedDict(datasets.items())
        self.order = [key for key in self.datasets.keys()]

        train_pid_range = [dataset.num_train_pids for _, dataset in self.datasets.items()]
        train_camid_max = [np.asarray(dataset.train)[:,2].astype(int).max(0)+1 if len(dataset.train)!=0 else 0 for _, dataset in self.datasets.items()]
        others_stat = [np.asarray(dataset.query+dataset.gallery)[:,1:].astype(int).max(0)+1 if len(dataset.query+dataset.gallery)!=0 else (0,0) for _, dataset in self.datasets.items()]
        others_pid_max = [stat[0] for stat in others_stat]
        others_camid_max = [stat[1] for stat in others_stat]

        self.train_pid_offset = [sum(train_pid_range[:self.order.index(name)]) for name in self.order]
        self.train_camid_offset = [sum(train_camid_max[:self.order.index(name)]) for name in self.order]
        upper_train_pid = sum(train_pid_range)
        upper_train_camid = sum(train_camid_max)
        self.others_pid_offset = [upper_train_pid+sum(others_pid_max[:self.order.index(name)]) for name in self.order]
        self.others_camid_offset = [upper_train_camid+sum(others_camid_max[:self.order.index(name)]) for name in self.order]

        train = [(t[0], t[1]+self.train_pid_offset[self.order.index(name)], t[2]+self.train_camid_offset[self.order.index(name)])
                    for name, dataset in self.datasets.items() for t in dataset.train]
        query = [(t[0], t[1]+self.others_pid_offset[self.order.index(name)], t[2]+self.others_camid_offset[self.order.index(name)])
                    for name, dataset in self.datasets.items() for t in dataset.query]
        gallery = [(t[0], t[1]+self.others_pid_offset[self.order.index(name)], t[2]+self.others_camid_offset[self.order.index(name)])
                    for name, dataset in self.datasets.items() for t in dataset.gallery]

        if merge:
            pid_container = set()
            self.others_pid_offset = np.array(self.others_pid_offset)
            train = query+gallery+train
            gallery = []
            query = []
            for t in train:
                pid_container.add(t[1])
            pids = [pid for pid in pid_container]
            pids.sort()
            pid2label={}
            for label, pid in enumerate(pids):
                pid2label[pid] = label
                if (pid+1) in self.others_pid_offset:
                    self.others_pid_offset[self.others_pid_offset==(pid+1)] = label+1
            train = [(t[0], pid2label[t[1]], t[2]) for t in train]
            self.others_pid_offset=self.others_pid_offset.tolist()


        if verbose:
            print("=> Combined{}: {} Loaded".format(' Merged' if merge else '',','.join(self.order)))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
