import os
import random
import numpy as np
import sys
import mlflow
import hydra
import logging
import gc
from omegaconf import DictConfig

sys.path.append('./')
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset, ConcatDataset

from src.utils.models.resnet import ResNet18
from src.utils.models.resnet import OriginalResNet
from src.utils.logger import Logger
from src.utils.dataset import get_data
from src.utils.query_strategy import get_strategy
from src.utils.train_helper import data_train
from src.utils.utils import LabeledToUnlabeledDataset

class TrainClassifier:

    def __init__(self, cfg, log_path,):
        self.cfg = cfg
        self.log_path = log_path
        self.model_name = 'resnet'
        self.channels = 1

    def getModel(self,):
        net = OriginalResNet(num_classes=len(self.classes), channels=self.channels) if self.model_name=='resnet' \
        else  ResNet18(num_classes=len(self.classes), channels=self.channels)

        return net

    def train_classifier(self):

        mlflow.log_params(self.cfg.al_method)
        mlflow.log_params(self.cfg.dataset)
        mlflow.log_params(self.cfg.train_parameters)
        random_seed = self.cfg.train_parameters.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

        logger = Logger(self.log_path)
        full_train_dataset, test_dataset, self.model_name, self.channels, self.classes = get_data(self.cfg.dataset)
        net = self.getModel()
        start_idxs = np.random.choice(len(full_train_dataset), size=self.cfg.dataset.initial_points, replace=False)
        train_dataset = Subset(full_train_dataset, start_idxs)
        unlabeled_dataset = Subset(full_train_dataset, list(set(range(len(full_train_dataset))) -  set(start_idxs)))
        strategy = get_strategy(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, len(self.classes), self.cfg.al_method)

        logging.info('#########################round0#########################')
        dt = data_train(train_dataset, net, self.cfg.train_parameters, self.cfg.dataset, logger)

        # Use DDP when your dataset is ImageNet and use multi-GPU
        if self.cfg.dataset.name == 'ImageNet' and torch.cuda.device_count()>1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            mp.spawn(dt.ddp_train, cfg=(self.classes), nprocs=torch.cuda.device_count())
        else:
            dt.train(self.classes)

        # test
        clf = self.getModel()
        clf.load_state_dict(torch.load(self.log_path+'/weight0.pth', map_location="cpu"))
        strategy.update_model(clf)
        dt.get_acc_on_set(test_dataset, self.classes, clf)

        for rd in range(1, self.cfg.dataset.rounds):
            logging.info(f'#########################round{rd}#########################')
            logger.rd = rd
            gc.collect()

            # query data
            idx = strategy.select(self.cfg.dataset.budget)
            train_dataset = ConcatDataset([train_dataset, Subset(unlabeled_dataset, idx)])
            remain_idx = list(set(range(len(unlabeled_dataset)))-set(idx))
            unlabeled_dataset = Subset(unlabeled_dataset, remain_idx)

            # update train dataset, then train
            strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
            dt = data_train(train_dataset, net, self.cfg.train_parameters, self.cfg.dataset, logger)
            if self.cfg.dataset.name == 'ImageNet' and torch.cuda.device_count()>1:
                mp.spawn(dt.ddp_train, cfg=(self.classes), nprocs=torch.cuda.device_count())
            else:
                dt.train(self.classes)

            # test
            clf = self.getModel()
            clf.load_state_dict(torch.load(self.log_path+'/weight'+str(rd)+'.pth', map_location="cpu"), strict=False)
            strategy.update_model(clf)
            dt.get_acc_on_set(test_dataset, self.classes, clf)

        print('Training Completed!')
        logger.show_result(self.cfg.train_parameters.seed)

@hydra.main(config_name='base', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow_runname)
    log_path = os.getcwd()
    with mlflow.start_run():
        os.chdir(hydra.utils.get_original_cwd())
        tc = TrainClassifier(cfg, log_path)
        tc.train_classifier()

if __name__ == '__main__':
    main()