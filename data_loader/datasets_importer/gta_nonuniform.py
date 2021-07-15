# encoding: utf-8
"""
ROSE Lab's GTA ReID Dataset, v3 alpha. This is the train-full version, meaning that there is no separation between
training set, query set and gallery set. The training set consists of all annotated images available. This is to
facillitate cross domain experiments.
"""
import os
import glob
import re
import numpy as np
from numpy import random
from .BaseDataset import BaseImageDataset
from PID_finder import finder

class GTA_NonUniform(BaseImageDataset):
    """
    GTA Long-Term

    Dataset statistics:
    # identities: 690 unique PID
    # appearances: 1972
    # images 11,358,928
    
    """
    dataset_dir = 'GTA_Long'

    def __init__(self, cfg, verbose = True, relabel = False, **kwargs):
        
        super(GTA_NonUniform, self).__init__()

        # self.filters = [
        #     cfg.FILTERS.LOCATION,
        #     cfg.FILTERS.WEATHER,
        #     cfg.FILTERS.TIME,
        #     cfg.FILTERS.ANIMATION,
        #     cfg.FILTERS.CAM_HEIGHT,
        #     cfg.FILTERS.PID_RANGE,
        #     cfg.FILTERS.YAW_ANGLE
        # ]
        
        # self.q_angles = cfg.FILTERS.QUERY_ANGLE
        # if cfg.FILTERS.PID_RANGE:
        #     pid_intlist = list(range(cfg.FILTERS.PID_RANGE[0],cfg.FILTERS.PID_RANGE[1]+1))
        #     self.filters[-2] = [str(pid) for pid in pid_intlist]
        # if cfg.FILTERS.YAW_ANGLE:
        #     #self.q_angles = [str(ang) for ang in range(1,9) if ang not in cfg.FILTERS.YAW_ANGLE]
        #     yaw_intlist = list(range(cfg.FILTERS.YAW_ANGLE[0],cfg.FILTERS.YAW_ANGLE[1]+1))
        #     self.filters[-1] = [str(yaw) for yaw in yaw_intlist if yaw not in self.q_angles]
        #     print('query angle:',self.q_angles)
        #     print('gallery angles:',self.filters[-1])
        
        ## FILTER PIDS ACCORDING TO GTAUNIFORM
        self.gta_uniform = True
        if self.gta_uniform:
            # self.filters[-2] = finder()
            pass


        # self.qname = '_'.join(map(str,self.q_angles))

        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.train_list = os.path.join(self.dataset_dir, 'train.txt')
        self.query_list = os.path.join(self.dataset_dir, f'query_1_nonuniform.txt')
        self.gallery_list = os.path.join(self.dataset_dir, f'gallery_1_nonuniform.txt')
        print(f'The dataset directory is: {self.dataset_dir}')
        
        pid_files = ['train.txt',f'gallery_1_nonuniform.txt']

        for fil in pid_files:
            if not os.path.exists(os.path.join(self.dataset_dir, fil)):
                print(f'Labels for {fil} have not been generated.')
                print(f'Labels will be generated into dir {self.dataset_dir}')
                self.generate_label(fil)
        # if cfg.FILTERS.QUERY_ANGLE:
        #     print(f"You are currently doing query on {self.qname}")

        self.check_before_run()
        self.cam_dict = {
            'person' : 0,
            'surveillance' : 1,
            'drone' : 2
        }

        train = self._process_list(self.train_list, relabel=True)
        gallery = self._process_list(self.gallery_list, relabel=False, camid=0)
        query = self._process_list(self.query_list, relabel=False, camid=1)

        if verbose:
            print(f"=> {self.dataset_dir} loaded. If you did not intend for the training set to consist of the full images, ABORT NOW!")
            self.print_dataset_statistics(train, query, gallery)
            

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def generate_label(self,fil):
        #This is where to add filters
        save_dir = self.dataset_dir
        img_format = '.jpg'
        loc_filter = self.filters[0]
        f = open(os.path.join(save_dir,fil),'w')
        if fil == f'gallery_1_nonuniform.txt':
            q_file =  open(os.path.join(save_dir,f'query_1_nonuniform.txt'),'w')

        if not os.path.exists(self.dataset_dir):
            print('Dataset not downloaded')
        
        for root, subdir, files in os.walk(self.dataset_dir):
            ped_list = []
            ped_list_s = []
            if self.gta_uniform:
                if self.problematic(root):
                    continue
            for _file in files:
                if _file.endswith('.txt') or _file.endswith('.xml'):
                    continue
                if _file.endswith(img_format):
                    _filestr = _file.replace(img_format,"") #removes the .jpg portion of the filename
                    _file_split = _filestr.split('_')
                    ped = self.create_pedlist(root, _file, fil, _file_split)
                    if ped == 0:
                        continue
                    else:
                        ped_list.append(ped)
            ped_list = np.array(ped_list)
            if ped_list.size == 0:
                continue
            
            #ped_list = np.random.choice(ped_list,100, replace = False)
            # ped_list = np.random.choice(ped_list,int(0.125*len(ped_list)), replace = False) #chooses a random sample of 50 images for the dataset, with replacement (i.e. there can be duplicates.)
            # q_list = np.random.choice(ped_list,int(0.2*len(ped_list)))

            for q in ped_list:
                q_str = q.replace(img_format,"")
                q = q.replace(img_format,"")
                q_split = q_str.split('_')
                if self.filters[-1] != []:
                    min_yaw = str(min(self.filters[-1]))
                else:
                    min_yaw = '1'
                # if fil == f'gallery_{self.qname}.txt':
                if q_split[-1] == min_yaw: #This should only execute once for each set of conditions.

                    if self.q_angles:
                        for ang in self.q_angles:
                            q_split[-1] = str(ang)
                            q = '_'.join(q_split)+'.jpg'
                            if fil == f'gallery_1_nonuniform.txt':
                                print(q, file = q_file)

                # while int(q_split[-1]) == int(q.split('_')[-1]):
                #     q_split[-1] = str(np.random.randint(1, high=9))
                # if (q.split('_')[-3]) == 'person':
                #     q_split[-3] = 'surveillance'
                # else:
                #     q_split[-3] = 'person'



            for ped in ped_list:
                print(ped, file = f)
            


        f.close()
        if fil == f'gallery_1_nonuniform.txt':
            q_file.close()
        # if self.gta_uniform:
        #     self.remove_duplicates(fil)
        #     self.remove_duplicates(f'query_{self.qname}.txt')


    def create_pedlist(self, root, _file, _filestr, _file_split):
        filters = self.filters
        for idx, fil in enumerate(filters):
            if fil:
                if _file_split[idx] not in fil:
                    return 0
            ped = os.path.join(''.join(root.split('/')[-2:-1]),_file_split[0], _file)
        return ped
        

    def check_before_run(self):
        '''Check that all files are available before going deeper'''
        checklist = [self.dataset_dir, self.train_list, self.query_list, self.gallery_list]
        for item in checklist:
            if not os.path.exists(item):
                raise RuntimeError(f"'{item}' is not available.")

    def _process_list(self, list_path, relabel=False, camid=-1):
        with open(list_path, 'r') as f: 
            img_paths = [l.strip() for l in f.readlines()]
        dataset = [] 
        if relabel: 
            pid_container = set()
            for img_path in img_paths: 
                #example img_path: 99/subway/subway_RAIN_13_walk_person_99_3.jpg
                pid = int(img_path.split('/')[-3]) 
                #pid = 99 for the above example
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            pid = int(img_path.split('/')[-3])
            filename = '/'.join(img_path.split('/')[-3:])
            dataset.append((os.path.join(self.dataset_dir, filename), int(pid), int(camid))) # camid :: person = 0, surveillance = 1, drone = 2
        return dataset

    def remove_duplicates(self,fil):
        pid_dict = {}
        with open(os.path.join(self.dataset_dir, fil),'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                line_us_split = line.split('_')
                line_slash_split = line.split('/')
                if line_us_split[-2] not in pid_dict:
                    pid_dict[line_us_split[-2]] = set()
                if not self.problematic(line_slash_split[0]):
                    pid_dict[line_us_split[-2]].add(int(line_slash_split[0]))
        print(pid_dict)

        with open(os.path.join(self.dataset_dir, fil),'w') as f:
            for line in lines:
                line = line.strip('\n')
                line_us_split = line.split('_')
                line_slash_split = line.split('/')
                try:
                    long_pid = str(min(pid_dict[line_us_split[-2]]))
                except ValueError:
                    print(f'{pid_dict[line_us_split[-2]]=}')
                    continue
                if line_slash_split[0] == long_pid:
                    print(line,file = f)
    
    def problematic(self,longid):
        problem_files = [1006,1056,1068,1078,1105,1111,1153,1171,1180,1192,1204,1209,
        1215,1218,1234,1239,1253,1286,1296,1324,1326,1331,1356,1373,1397,1405,1437,1479,
        1509,1513,1518,1524,1531,1547,1576,1598,1605,1614,1620,1631,1640,1645,1678,1707,
        1731,1745,1810,1830,1890,1895,1901,1906,1938,1949,1961,940,938,937]
        problem_files_str = [str(f) for f in problem_files]
        
        return longid in problem_files_str




    
