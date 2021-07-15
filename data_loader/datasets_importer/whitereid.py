# encoding: utf-8
import os
import glob
import re
from .BaseDataset import BaseImageDataset

class WhiteReID(BaseImageDataset):
    """
    White-reID 

    Dataset statistics:    
    # identities: 1214 (train + query)
    # images:10040 (train) + 2756 (query) + 10336 (gallery)    
    """

    dataset_dir = 'White-reID'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(WhiteReID, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> White-reID loaded")
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
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(.*\d.*)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = pattern.search(img_path).groups()
            pid = pid.split('/')[-1]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pids, camid = pattern.search(img_path).groups()
            pids = pids.split('/')[-1]
            pid = pids            
            if relabel: 
                pid = pid2label[pids]
            else:
                if pids.startswith('b'):
                    pid = pids.split('_')[1]
                else:
                    pid = pids
            pid = int(pid)
            camid = int(camid)            
            dataset.append((img_path, pid, camid))

        return dataset