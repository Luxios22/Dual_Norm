# encoding: utf-8
from .cuhk03 import CUHK03
from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .msmt17 import MSMT17
from .ImageDataset import ImageDataset
from .random_viper import Random_VIPeR
from .random_grid import Random_GRID
from .random_prid import Random_PRID
from .random_ilids import Random_iLIDS
from .cuhk02 import CUHK02
from .cuhk_sysu import CUHK_SYSU
from .combined import Combined
from collections import OrderedDict
from .ntu_outdoor_night_v1 import NTU_Outdoor_Night_V1
from .ntu_outdoor_night_v2 import NTU_Outdoor_Night_V2
from .ntu_outdoor_night_v1_enlighten import NTU_Outdoor_Night_V1_Enlighten


__factory = {
    'CUHK03': CUHK03,
    'Market-1501': Market1501,
    'DukeMTMC-reID': DukeMTMC,
    'MSMT17_V2': MSMT17,
    'VIPeR': Random_VIPeR,
    'GRID': Random_GRID,
    'PRID2011': Random_PRID,
    'i-LIDS': Random_iLIDS,
    'CUHK02': CUHK02,
    'CUHK-SYSU': CUHK_SYSU,
    'NTU_Outdoor_Night_V1': NTU_Outdoor_Night_V1,
    'NTU_Outdoor_Night_V2': NTU_Outdoor_Night_V2,
    'NTU_Outdoor_Night_V1_Enlighten': NTU_Outdoor_Night_V1_Enlighten
}

def get_names():
    return __factory.keys()


def init_dataset(cfg,dataset_names, merge, *args, **kwargs):
    for dataset_name in dataset_names:
        if dataset_name not in __factory.keys():
            raise KeyError("Unknown datasets: {}".format(dataset_name))
    if len(dataset_names) == 1:
        return __factory[dataset_names[0]](cfg,*args, **kwargs)
    else:
        datasets = OrderedDict([(name, __factory[name](cfg,*args, verbose=False, **kwargs)) for name in dataset_names])
        return Combined(datasets, merge)