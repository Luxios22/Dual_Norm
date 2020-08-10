# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BasePlainDataset, BaseImageDataset
import pickle
import os
import cv2
from scipy.io import loadmat

class CUHK_SYSU_Raw(BasePlainDataset):

    """No camera"""

    dataset_name = 'CUHK-SYSU'

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.store_dir = store_dir
        self.dataset_dir = os.path.join(self.store_dir, self.dataset_name, 'processed')

        self._check_process()
        self._check_before_run()

        dataset_pth = os.path.join(self.dataset_dir, 'dataset.list')
        if os.path.exists(dataset_pth):
            with open(dataset_pth, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._process_dir(self.dataset_dir, relabel=True)
            with open(dataset_pth, 'wb') as f:
                pickle.dump(data, f)

        if verbose:
            print("=> CUHK-SYSU Loaded")
            self.print_dataset_statistics(data)

        self.data = data

    def _check_process(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            print('CUHK-SYSU Image Extraction Processing')
            person_pth =  os.path.join(self.store_dir, self.dataset_name, 'annotation', 'Person.mat')
            print(person_pth)
            img_dir = os.path.join(self.store_dir, self.dataset_name,'Image', 'SSM')
            save_dir = os.path.join(self.store_dir, self.dataset_name,'processed')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            persons = loadmat(person_pth)['Person'][0]
            for person in persons:
                process(person, img_dir, save_dir)
            print('CUHK-SYSU Image Extraction Completed')

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.jpg'), recursive=True)
        # pattern = re.compile(r'p([\d]+)\/s([\d]+).jpg')

        # Example: ./P1/cam2/238_0324.png

        pid2label_pth = os.path.join(self.dataset_dir, 'pid2label.dict')
        if os.path.exists(pid2label_pth):
            with open(pid2label_pth, 'rb') as f:
                pid2label = pickle.load(f)
        else:
            pid_container = set()
            for img_path in img_paths:
                print(img_path)
                pid = os.path.basename(img_path).split('_')[0][1:]
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            with open(pid2label_pth, 'wb') as f:
                pickle.dump(pid2label, f)

        dataset = []
        for img_path in img_paths:
            pid = os.path.basename(img_path).split('_')[0][1:]
            camid = 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset


class CUHK_SYSU(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = CUHK_SYSU_Raw(cfg.DATASETS.STORE_DIR, verbose=False).data

        self.train = data
        self.query = []
        self.gallery = []


        if verbose:
            print("=> CUHK-SYSU Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


def process(person,img_dir,save_dir):
    pid, _, pics = person
    pid = pid.item()
    pics = pics[0]
    for num, pic in enumerate(pics):
        file, bbox, _ = pic
        file = file.item()
        bbox = bbox[0]

        x, y, w, h = map(int, bbox)

        img = cv2.imread(os.path.join(img_dir, file))

        crop_img = img[y:y+h+1, x:x+w+1]

        if not os.path.exists(os.path.join(save_dir, pid)):
            os.mkdir(os.path.join(save_dir, pid))
        cv2.imwrite(os.path.join(save_dir, pid, pid+'_'+str(num)+'_'+file), crop_img)

