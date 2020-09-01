from __future__ import absolute_import

from .ResNet_IFN import *
from .ResNet import *
from .MobileNet_IFN import *
from .MobileNet import *
# from .MMFA_AAE import resnet50_IFN as MMFA_AAE



__factory = {  
    'resnet50': resnet50,
    'resnet50_ifn': resnet50_IFN,
	'mobilenet':MobileNetV2,
    'mobilenet_ifn': MobileNetV2_IFN,
    # 'mmfa_aae': MMFA_AAE,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)