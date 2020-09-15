# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.optim import lr_scheduler
from .WarmupMultiStepLR import WarmupMultiStepLR
from .DelayedCosineAnnealingLR import DelayedCosineAnnealingLR


def make_scheduler(cfg,optimizer):
    
    if cfg.SOLVER.SCHED=="WarmupMultiStepLR":
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.WARMUP_STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    elif cfg.SOLVER.SCHED=="DelayedCosineAnnealingLR":
        scheduler = DelayedCosineAnnealingLR(optimizer,cfg.SOLVER.DELAY_ITERS,cfg.SOLVER.MAX_EPOCHS,cfg.SOLVER.ETA_MIN_LR,cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_ITERS,cfg.SOLVER.WARMUP_METHOD)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, cfg.SOLVER.STEP, cfg.SOLVER.GAMMA)
    return scheduler
