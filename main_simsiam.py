import os
import random
import math
import shutil
import mlflow
import hydra
import logging
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset

import simsiam.simsiam.loader as loader
import simsiam.simsiam.builder as builder
from src.utils.models.resnet import Original_ResNet
from src.utils.models.resnet import ResNet18

def prepare_train_dataset(cfg):
    if cfg.dataset.name == 'CIFAR10':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/data/dataset/cifar10/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'EuroSAT':
        augmentation = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/data/dataset/eurosat/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'BrainTumor':
        augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/data/dataset/brain_tumor/BrainTumor'
        train_dataset = datasets.ImageFolder(root=traindir+'/'+str(1), transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
        for i in range(2, 5):
            train_dataset = ConcatDataset([train_dataset, (datasets.ImageFolder(root=traindir+'/'+str(i), transform=loader.TwoCropsTransform(transforms.Compose(augmentation))))])
    
    elif cfg.dataset.name == 'OCT':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/data/dataset/oct_modified/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'GAPs':
        augmentation = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/data/dataset/gaps/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'ImageNet':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/imagenet_dataset/ILSVRC2012_img_train'
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_parametres.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)
    return train_loader


# introduce hydra, mlflow for logging tool
@hydra.main(config_name='ssl_config', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
        mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
        mlflow.set_experiment(cfg.mlflow_runname)
        with mlflow.start_run():
            os.chdir(hydra.utils.get_original_cwd())
            mlflow.log_params(cfg.dataset)
            mlflow.log_params(cfg.train_parameters)

            # fix seed
            random_seed = cfg.train_parmeters.seed
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gpu_ids=[]
            for i in range(torch.cuda.device_count()):
                gpu_ids.append(i)
            
            print(f'use {gpu_ids} gpu')

            # fit batch_size
            cfg.train_parameters.batch_size = int(cfg.train_parameters.batch_size/8*len(gpu_ids))
            
            main_worker(device, gpu_ids, cfg)


def main_worker(device, gpu_ids, cfg):

    # model
    if cfg.dataset.name == 'CIFAR10':
        model = builder.SimSiam(
            ResNet18,
            cfg.train_parameters.dim, cfg.train_parameters.pred_dim)
    else:
        model = builder.SimSiam(
            Original_ResNet,
            cfg.train_parameters.dim, cfg.train_parameters.pred_dim)
    if cfg.train_parameters.start_epoch > 0:
        model.load_state_dict(torch.load('weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed)+'/checkpoint.pth.tar')['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    # infer learning rate before changing batch size
    init_lr = cfg.train_parameters.lr * cfg.train_parameters.batch_size / 256

    # loss
    criterion = nn.CosineSimilarity(dim=1).to(device)

    # learning rate setting
    if cfg.train_parameters.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    
    # optimizer
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9, weight_decay=1e-4)
    print(optimizer)

    # dataset
    train_loader = prepare_train_dataset(cfg)
    
    os.makedirs('weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed), exist_ok=True)

    for epoch in range(cfg.train_parameters.start_epoch, cfg.train_parameters.n_epoch):
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename='weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed)+'/checkpoint.pth.tar')

from tqdm import tqdm
def train(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # end = time.time()
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(train_loader, unit='batch', desc='| Pretrain |', dynamic_ncols=True)
    for i, (images, _) in enumerate(loop):
        # measure data loading time
        image0 = images[0].to(device, non_blocking=True)
        image1 = images[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # compute output and loss
            p1, p2, z1, z2 = model(x1=image0, x2=image1)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    progress.display(0)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, cfg):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.train_parameters.n_epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

if __name__ == '__main__':

    main()
