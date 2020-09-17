# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import glob
import re
from .BaseDataset import BaseImageDataset


class NTU_Outdoor_Night_V1_Enlighten(BaseImageDataset):
    """
    NTUIndoor
    Reference:

    Dataset statistics:
    # identities: N/A
    # images: N/A (train) + N/A (query) + N/A (gallery)
    """
    dataset_dir = 'NTU_Outdoor_Night_V1_Enlighten'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(NTU_Outdoor_Night_V1_Enlighten, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = []
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> NTU Outdoor Night V1.0 Enlighten Loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.png'))

        pid_container = set()
        for img_path in img_paths:
            pid = os.path.basename(img_path).split('_')[4]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in img_paths:
            pid = os.path.basename(img_path).split('_')[4]
            camid = os.path.basename(img_path).split('_')[0]

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

