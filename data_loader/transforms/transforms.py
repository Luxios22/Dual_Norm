import torchvision.transforms as T
from .RandomPatch import RandomPatch
from .autoaugment import *

def transforms(cfg, is_train=True):
    if is_train:
        transform_list = [T.Resize(cfg.INPUT.SIZE_TRAIN),
                        T.RandomHorizontalFlip(p=cfg.INPUT.HF_PROB),
                        T.Pad(cfg.INPUT.PADDING),
                        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                        T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS, contrast=cfg.INPUT.CONTRAST, saturation=cfg.INPUT.SATURATION, hue=cfg.INPUT.HUE)]
        if cfg.INPUT.DO_AUTOAUG:
            transform_list.insert(0, ImageNetPolicy(cfg.SOLVER.MAX_ITER))
        if cfg.INPUT.RP:
            transform_list.append(RandomPatch(prob_happen=cfg.INPUT.RP_PROB))
        transform_list.append(T.ToTensor())        
        transform_list.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
        if cfg.INPUT.RE:
            transform_list.append(T.RandomErasing(p=cfg.INPUT.RE_PROB))

    else:
        transform_list = [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]
    return T.Compose(transform_list)
