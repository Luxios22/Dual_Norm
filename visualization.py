import fire
import os
import time
import torch
import numpy as np
import models
from config import cfg
from logger import make_logger
from evaluation import evaluation
from datasets import Person_ReID_Dataset_Downloader
from torch import mode, optim
from torch.optim import lr_scheduler
from loss import CrossEntropyLabelSmooth
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.datasets_importer import init_dataset, ImageDataset
from data_loader.samplers import RandomIdentitySampler
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from utils import check_jupyter_run
import scipy.ndimage as ndimage
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def visualization(config_file = 'config/dual_norm.yaml', num_classes = 18530, n = 8 ,image_path = 'image/VIPER.bmp',  **kwargs):
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device(cfg.DEVICE)

    model = models.init_model(name=cfg.MODEL.NAME, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(output_dir,'net_'+str(cfg.TEST.LOAD_EPOCH)+'.pth'),map_location=torch.device('cpu')), strict=False)
    model.to(device)
    model = model.eval()

    
    preprocess = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    raw_img = Image.open(image_path)
    img = preprocess(raw_img)
    img = img.unsqueeze(0)

    my_embedding = torch.zeros(2048,n,4)
    def hook(module, input, output):
        outputs.append(output)

    avgpool_layer = model._modules.get('global_avgpool')
    def fun(m, i, o): my_embedding.copy_(i[0].data[0])
    h = avgpool_layer.register_forward_hook(fun)


    with torch.no_grad():
        if device:
            model.to(device)
            images = img.to(device)
        heatmap = model(images)
        h.remove()

    value = []
    i = 0
    for data in my_embedding:
        value.append(torch.max(data).item())

    sort_index = np.argsort(value) 

    map_index = sort_index[-1]


    width, height = raw_img.size
    heat_map = Image.fromarray(my_embedding[map_index].numpy()*255)

    heat_map_resized = heat_map.resize((width,height))
    heat_map_resized = ndimage.gaussian_filter(heat_map_resized, sigma=(10, 10), order=0)
    heat_map_resized = np.asarray(heat_map_resized)

    plt.axis('off')
    plt.imshow(raw_img)
    plt.imshow(heat_map_resized, alpha=0.6, cmap='bwr')
    plt.pause(10)

    # print(heatmap)
    




if __name__=='__main__':
    fire.Fire(visualization)
