import numpy as np
import os
import glob
import re
from .BaseDataset import BasePlainDataset, BaseImageDataset

class iLIDS(BasePlainDataset):

    """No Camera"""

    dataset_dir = 'i-LIDS'

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(store_dir, self.dataset_dir, 'images')

        self._check_before_run()

        data = self._process_dir(self.dataset_dir, relabel=True)

        if verbose:
            print("=> i-LIDS Loaded")
            self.print_dataset_statistics(data)

        self.data = data


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.jpg'), recursive=True)
        pattern = re.compile(r'([\d]{4})([\d]{3}).jpg')

        # Example: ./P1/cam2/238_0324.png

        dataset = []
        pid2label = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid2label.add(pid)
            dataset.append((img_path, pid, None))

        if relabel:
            temp = []
            pid2label = {pid: label for label, pid in enumerate(pid2label)}
            for data in dataset:
                temp.append((data[0], pid2label[data[1]], data[2]))
            dataset = temp
        return dataset


class Random_iLIDS(BaseImageDataset):
    """A to B"""
    def __init__(self, cfg, verbose=True):
        data = iLIDS(cfg.DATASETS.STORE_DIR, verbose=False).data

        query_ids = np.arange(119)
        np.random.shuffle(query_ids)
        query_ids = query_ids[:60]

        self.train = []
        self.query = []
        self.gallery = []
        counter = {idx: 0 for idx in query_ids}
        for ele in data:
            if ele[1] in query_ids and counter[ele[1]]<2:
                counter[ele[1]]+=1
                if counter[ele[1]] == 1:
                    self.query.append((ele[0], ele[1], 0))
                else:
                    self.gallery.append((ele[0], ele[1], 1))

        if verbose:
            print("=> Random i-LIDS Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
