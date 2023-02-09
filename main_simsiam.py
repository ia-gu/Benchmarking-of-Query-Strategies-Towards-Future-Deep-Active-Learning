import os
import random
import mlflow
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import src.simsiam.utils.builder as builder
from src.simsiam.utils.loader import get_dataloader
from src.simsiam.utils.train import train
from src.simsiam.utils.utils import adjust_learning_rate, save_checkpoint
from src.utils.models.resnet import OriginalResNet, ResNet18

@hydra.main(config_name='ssl_config', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
        mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
        mlflow.set_experiment(cfg.mlflow_runname)
        with mlflow.start_run():
            os.chdir(hydra.utils.get_original_cwd())
            print('dataset: ', end='')
            mlflow.log_params(cfg.dataset)
            mlflow.log_params(cfg.train_parameters)

            random_seed = cfg.train_parameters.seed
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            os.makedirs('weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed), exist_ok=True)
            gpu_ids=[]
            for i in range(torch.cuda.device_count()):
                gpu_ids.append(i)

            main_worker(device, gpu_ids, cfg)

def main_worker(device, gpu_ids, cfg):

    # model
    model = builder.SimSiam(ResNet18, cfg.train_parameters.dim, cfg.train_parameters.pred_dim) if cfg.dataset.name == 'CIFAR10' \
    else    builder.SimSiam(OriginalResNet, cfg.train_parameters.dim, cfg.train_parameters.pred_dim)

    # resume pretraining
    if cfg.train_parameters.start_epoch > 0:
        model.load_state_dict(torch.load('weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed)+'/checkpoint.pth.tar')['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    # learning rate setting
    init_lr = cfg.train_parameters.lr * cfg.train_parameters.batch_size / 256
    if cfg.train_parameters.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9, weight_decay=1e-4)
    train_loader = get_dataloader(cfg)
    criterion = nn.CosineSimilarity(dim=1).to(device)

    # train
    for epoch in range(cfg.train_parameters.start_epoch, cfg.train_parameters.n_epoch):
        print('adjust_learning_rate')
        adjust_learning_rate(optimizer, init_lr, epoch, cfg)

        train(train_loader, model, criterion, optimizer, epoch, device)

        save_checkpoint({'state_dict': model.module.state_dict(), 'optimizer' : optimizer.state_dict(),}, 
                          is_best=False, filename='weights/'+cfg.dataset.name+'/'+str(cfg.train_parameters.seed)+'/checkpoint')

if __name__ == '__main__':

    main()
