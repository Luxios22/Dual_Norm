from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'ResNet50'
_C.MODEL.LAST_STRIDE = 2
_C.MODEL.POOL = "AVG"

_C.MODEL.CLASS_PARAM = 1.
_C.MODEL.D_PARAM = 1.
_C.MODEL.G_PARAM = 1.

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.HF_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
#_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
#_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# Random probability for image random erasing
_C.INPUT.RE = True
# Random probability for image random erasing
_C.INPUT.RE_PROB = 0.5

# Value of colorjitter brightness
_C.INPUT.BRIGHTNESS = 0.0
# Value of colorjitter contrast
_C.INPUT.CONTRAST = 0.0
# Value of colorjitter saturation
_C.INPUT.SATURATION = 0.0
# Value of colorjitter hue
_C.INPUT.HUE = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Setup storage directroy for dataset
_C.DATASETS.STORE_DIR = ('./datasets')
# Source domains for cross-domain experiments
_C.DATASETS.SOURCE = ['Market1501', 'DukeMTMC']
# Target domains for cross-domain experiments
_C.DATASETS.TARGET = ['CUHK03']

_C.DATASETS.MERGE = True

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 32

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Sampler for data loading
_C.SOLVER.LOSS = 'softmax'
_C.SOLVER.LAMBDA1 = 1.
_C.SOLVER.LAMBDA2 = 1.

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.BASE_LR = 0.1

# SGD
_C.SOLVER.NESTEROV = True

# Adam
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.TRI = CN()
_C.SOLVER.TRI.MARGIN = 0.3
_C.SOLVER.TRI.NORM_FEAT = False
_C.SOLVER.TRI.HARD_MINING = True
_C.SOLVER.TRI.USE_COSINE_DIST = False
_C.SOLVER.TRI.SCALE = 1.0


_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.STEP = 40

_C.SOLVER.SCHED = "StepLR"
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.DELAY_ITERS = 100
_C.SOLVER.ETA_MIN_LR = 3e-7

_C.SOLVER.WARMUP_STEPS = [30, 55]
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.LOAD_EPOCH = 120


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.DEVICE = "cuda:0"
_C.RE_RANKING = False
