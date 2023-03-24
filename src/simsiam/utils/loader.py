from PIL import ImageFilter
import random

import torch
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_dataloader(cfg):
    if cfg.dataset.name == 'CIFAR10':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        traindir = '/localdata/cifar10/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'EuroSAT':
        augmentation = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/localdata/eurosat/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'BrainTumor':
        augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/localdata/brain_tumor/BrainTumor'
        train_dataset = datasets.ImageFolder(root=traindir+'/'+str(1), transform=TwoCropsTransform(transforms.Compose(augmentation)))
        for i in range(2, 5):
            train_dataset = ConcatDataset([train_dataset, (datasets.ImageFolder(root=traindir+'/'+str(i), transform=TwoCropsTransform(transforms.Compose(augmentation))))])
    
    elif cfg.dataset.name == 'OCT':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/localdata/oct_modified/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))
    
    elif cfg.dataset.name == 'GAPs':
        augmentation = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/localdata/gaps/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))

    elif cfg.dataset.name == 'KSDD2':
        augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/localdata/ksdd2/train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))

    elif cfg.dataset.name == 'ImageNet':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        traindir = '/imagenet_dataset/ILSVRC2012_img_train'
        train_dataset = datasets.ImageFolder(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)
    return train_loader