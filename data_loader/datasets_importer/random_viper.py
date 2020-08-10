import numpy as np
import os
import glob
import re
from .BaseDataset import BasePlainDataset, BaseImageDataset

class VIPeR(BasePlainDataset):

    dataset_dir = 'VIPeR'

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(store_dir, self.dataset_dir)

        self._check_before_run()

        data = self._process_dir(self.dataset_dir, relabel=True)

        if verbose:
            print("=> VIPeR Loaded")
            self.print_dataset_statistics(data)

        self.data = data

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.bmp'), recursive=True)
        pattern = re.compile(r'cam_(.)\/([\d]+)_[\d]+.bmp')

        dataset = []
        pid2label = set()
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            camid, pid = filename.split('.')[0].split('_')
            # camid, pid = pattern.search(img_path).groups()
            pid = int(pid)
            pid2label.add(pid)
            camid = 0 if camid=='a' else 1
            dataset.append((img_path, pid, camid))

        if relabel:
            temp = []
            pid2label = {pid: label for label, pid in enumerate(pid2label)}
            for data in dataset:
                temp.append((data[0], pid2label[data[1]], data[2]))
            dataset = temp
        return dataset

class Random_VIPeR(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = VIPeR(cfg.DATASETS.STORE_DIR, verbose=False).data
        query_ids = np.arange(632)
        np.random.shuffle(query_ids)
        query_ids = query_ids[:316]

        self.train = []
        self.query = []
        self.gallery = []
        for ele in data:
            if ele[1] in query_ids:
                if ele[2]==0:
                    self.query.append(ele)
                elif ele[2]==1:
                    self.gallery.append(ele)

        if verbose:
            print("=> Random VIPeR Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
