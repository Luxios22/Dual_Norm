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
from torch import optim
from torch.optim import lr_scheduler
from loss import CrossEntropyLabelSmooth
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.datasets_importer import init_dataset, ImageDataset
from data_loader.samplers import RandomIdentitySampler
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

def data_loader(cfg,dataset_names,merge):
    train_transforms = data_transforms['train']
    val_transforms = data_transforms['val']
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg, dataset_names,merge)
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset_names, dataset.train, train_transforms)
    if len(dataset.train) == 0:
        train_loader = None
    else:
        if cfg.SOLVER.LOSS == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

    if len(dataset.query + dataset.gallery) == 0:
        val_loader = None
    else:
        val_set = ImageDataset(dataset_names, dataset.query + dataset.gallery, val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    return train_loader, val_loader, len(dataset.query), num_classes



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def save_network(network, epoch, output_dir):
    save_filename = 'net_%s.pth'% (epoch+1)
    save_path = os.path.join(output_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)



class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, names, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids
def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W N:TensornSamples in minibatch, i.e., batchsize x nChannels x Height x Width
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def test(config_file = 'config/dual_norm.yaml', num_classes = 18530, number_fold = 10,  **kwargs):
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    resume=False

    [Person_ReID_Dataset_Downloader(cfg.DATASETS.STORE_DIR,dataset) for dataset in cfg.DATASETS.TARGET]
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = make_logger("Reid_Baseline", output_dir,'result_new', resume)
    if not resume:
        logger.info("Using {} GPUS".format(1))
        logger.info("Loaded configuration file {}".format(config_file))
        logger.info("Running with config:\n{}".format(cfg))

    output_dir = cfg.OUTPUT_DIR
    device = torch.device(cfg.DEVICE)

    # train_loader, _, num_query, num_classes = data_loader(cfg,cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE)
    # _, val_loader, _, _ = data_loader(cfg,cfg.DATASETS.TARGET,merge=False)
    val_stats = [data_loader(cfg,(target,),merge=False)[1:3] for target in cfg.DATASETS.TARGET]

    # model = models.init_model(name='mobilenet_ifn', num_classes=num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes, device=cfg.DEVICE)

    # for resnet50
    model = models.init_model(name=cfg.MODEL.NAME, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(output_dir,'net_'+str(cfg.TEST.LOAD_EPOCH)+'.pth'),map_location=torch.device('cpu')), strict=False)
    model.to(device)
    model = model.eval()



          # must be done before the optimizer generation
    # ignored_params = list(map(id, model.fn.parameters() ))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    # optimizer = optim.SGD([
    #              {'params': base_params, 'lr': 0.1*0.05},
    #              {'params': model.fn.parameters(), 'lr': 0.05}
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    base_epo = 0
    # if resume:
    #     optimizer.load_state_dict(torch.load(checkpoints['opt']))
    #     sch_dict = torch.load(checkpoints['sch'])
    #     scheduler.load_state_dict(sch_dict)
    #     base_epo = checkpoints['epo']


    # logger.info("Start training")
    since = time.time()
    # for epoch in range(epochs):
    #     count = 0
    #     running_loss = 0.0
    #     running_acc = 0
    #     for data in tqdm(train_loader, desc='Iteration', leave=False):
    #         model.train()
    #         images, labels = data
    #         if device:
    #             model.to(device)
    #             images, labels = images.to(device), labels.to(device)

    #         optimizer.zero_grad()

    #         scores = model(images)
    #         loss = loss_fn(scores, labels)

    #         loss.backward()
    #         optimizer.step()

    #         count = count + 1
    #         running_loss += loss.item()
    #         running_acc += (scores.max(1)[1] == labels).float().mean().item()


    #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
    #                                 .format(base_epo+epoch+1, count, len(train_loader),
    #                                 running_loss/count, running_acc/count,
    #                                 scheduler.get_lr()[0]))
    #     scheduler.step()

    #     if (base_epo+epoch+1) % checkpoint_period == 0:
    #         save_network(model, base_epo+epoch, output_dir)
    #         torch.save(optimizer.state_dict(), os.path.join(output_dir, 'opt_epo'+str(base_epo+epoch+1)+'.pth'))
    #         torch.save(scheduler.state_dict(), os.path.join(output_dir, 'sch_epo'+str(base_epo+epoch+1)+'.pth'))

        # Validation
    logger.info("Start testing")
    for i, (val_loader, num_query) in enumerate(val_stats):
        fold = number_fold
        all_cmc = [0.0,0.0,0.0]
        all_mAP = 0.0
        for j in tqdm(range(fold)):
            all_feats = []
            all_pids = []
            all_camids = []
            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data
                    n, c, h, w = images.size()
                    ff = torch.FloatTensor(n,2048).zero_()
                    for i in range(2):
                        if(i==1):
                            images = fliplr(images)
                        model.to(device)
                        input_img = images.to(device)
                        outputs = model(input_img)
                        f = outputs.data.cpu()
                        ff = ff+f   
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))

                all_feats.append(ff)
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query)
            all_cmc[0] = all_cmc[0] + cmc[0]
            all_cmc[1] = all_cmc[1] + cmc[4]
            all_cmc[2] = all_cmc[2] + cmc[9]
            all_mAP = all_mAP + mAP
        logger.info("Validation Results: {}".format(cfg.DATASETS.TARGET[i]))
        logger.info("mAP: {:.1%}".format(all_mAP/n))
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(1, all_cmc[0]/n))
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(5, all_cmc[1]/n))
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(10, all_cmc[2]/n))


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('-' * 10)


if __name__=='__main__':
    fire.Fire(test)
