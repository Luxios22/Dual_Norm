import os 
import glob
from .BaseDataset import BaseImageDataset

class NTUOutdoorv2_white(BaseImageDataset):
    
    dataset_dir = "NTUOutdoor_Color_V2/white"

    def __init__(self, cfg, verbose=True, **kwargs):
        super(NTUOutdoorv2_white, self).__init__()        
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)        
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')        
        self._check_before_run()

        train = []
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False) 

        if verbose:
            print("=> NTUOutdoor/white Loaded")
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

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            base_path = os.path.basename(img_path)
            pid = int(base_path.split('_')[4])            
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            base_path = os.path.basename(img_path)
            pids = int(base_path.split('_')[4])    
            camid = int(base_path.split('_')[0])
            pid = pids
            if relabel:
                pid = pid2label[pids]                         
            dataset.append((img_path, pid, camid))

        return dataset
