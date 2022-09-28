import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import mlflow
import hydra
import logging
import csv
import gc
from omegaconf import DictConfig

sys.path.append('./')
import torch
from torch import utils
import torch.multiprocessing as mp
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets, transforms

from src.utils.models.resnet import ResNet18
from src.utils.models.resnet import ResNet
from src.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, LeastConfidenceSampling, \
                                        MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
                                        BALDDropout, FASS, BatchBALDDropout, SubmodularSampling, ClusterMarginSampling
from src.utils.train_helper import data_train
from src.utils.utils import LabeledToUnlabeledDataset

class TrainClassifier:
	
    def __init__(self, cfg, log_path,):
        self.cfg = cfg
        self.log_path = log_path
        self.model_name = 'resnet'
        self.channels = 1

    def getModel(self,):

        if self.model_name == 'resnet':
            net = ResNet(num_classes = self.num_classes, channels = self.channels)

        else:
            # ResNet18 for CIFAR10
            net = ResNet18(num_classes = self.num_classes, channels = self.channels)

        return net

    def getData(self, data_cfg):
        # download_path needs to be modified
        # data_path also needs to be modified

        if data_cfg.name == 'CIFAR10':
            self.model_name = 'modified_resnet'
            self.channels = 3
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            download_path = './downloaded_data/'
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = datasets.CIFAR10(download_path, download=True, train=True, transform=train_transform, target_transform=torch.tensor)
            test_dataset = datasets.CIFAR10(download_path, download=True, train=False, transform=test_transform, target_transform=torch.tensor)

        elif data_cfg.name == 'EuroSAT':
            self.channels = 3
            self.classes = ('AnnualCrop', 'Forest', 'Herbaceous', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')
            data_path = '/data/dataset/eurosat/2750'
            train_transform = transforms.Compose([transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            data = datasets.ImageFolder(root=data_path)
            train_size = 22000
            test_size = len(data) - train_size
            train_dataset, test_dataset = utils.data.random_split(data, [train_size, test_size])
            train_dataset.dataset.transform = train_transform
            test_dataset.dataset.transform = test_transform

        elif data_cfg.name == 'BrainTumor':
            self.classes = ('Glioma', 'Meningioma', 'Pituitary')
            train_data_path = '/data/dataset/brain_tumor/BrainTumor'
            test_data_path = '/data/dataset/brain_tumor/BrainTumor/5'
            train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = datasets.ImageFolder(root=train_data_path+'/'+str(1), transform=train_transform)
            for i in range(2, 5):
                train_dataset = ConcatDataset([train_dataset, (datasets.ImageFolder(root=train_data_path+'/'+str(i), transform=train_transform))])
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

        elif data_cfg.name == 'OCT':
            self.classes = ('CNV', 'DME', 'DRUSEN', 'NORMAL')
            train_data_path = '/data/dataset/oct_modified/train'
            test_data_path = '/data/dataset/oct_modified/test'
            train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        
        elif data_cfg.name == 'GAPs':
            self.classes = ('AppliedPatch', 'Crack', 'InlaidPatch', 'IntactRoad', 'OpenJoint', 'Pothol')
            train_data_path = '/data/dataset/gaps/train'
            test_data_path = '/data/dataset/gaps/test'
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

        elif data_cfg.name == 'ImageNet':
            self.channels = 3
            self.classes = list()
            for i in range(1000):
                self.classes.append(str(i))
            print(f'ImageNet_Classes: {len(self.classes)}')
            train_data_path = '/imagenet_dataset/imagenet_2012/ILSVRC2012_img_train'
            test_data_path = '/imagenet_dataset/imagenet_2012/ILSVRC2012_img_val_for_ImageFolder'
            train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

        self.num_classes = len(self.classes)
        return train_dataset, test_dataset

    def train_classifier(self):
        t = time.time()
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
        del random_seed
        
        full_train_dataset, test_dataset = self.getData(self.cfg.dataset)
        net = self.getModel()
        selected_strat = self.cfg.al_method.strategy
        budget = self.cfg.dataset.budget
        start = self.cfg.dataset.initial_points
        n_rounds = self.cfg.dataset.rounds
        strategy_args = self.cfg.al_method
        nSamps = len(full_train_dataset)
        start_idxs = np.random.choice(nSamps, size=start, replace=False)
        train_dataset = Subset(full_train_dataset, start_idxs)
        unlabeled_dataset = Subset(full_train_dataset, list(set(range(len(full_train_dataset))) -  set(start_idxs)))
        del full_train_dataset, start, nSamps, start_idxs

        if selected_strat == 'badge':
            strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'glister':
            strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args,validation_dataset=None,\
                    typeOf='Diversity',lam=10)
        elif selected_strat == 'entropy_sampling':
            strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'margin_sampling':
            strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'least_confidence':
            strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'coreset':
            strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'fass':
            strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'random_sampling':
            strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'bald_dropout':
            strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'adversarial_bim':
            strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'kmeans_sampling':
            strategy = KMeansSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'adversarial_deepfool':
            strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'batch_bald':
            strategy = BatchBALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'submodlib':
            strategy = SubmodularSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        elif selected_strat == 'cluster_margin':
            strategy = ClusterMarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, self.num_classes, strategy_args)
        else:
            raise IOError('Enter Valid Strategy!')
        
        logging.info('#########################')
        logging.info('round0')
        logging.info('#########################')
        dt = data_train(train_dataset, net, self.cfg.train_parameters, self.cfg.dataset)

        # Use DDP when your dataset is ImageNet and use multi-GPU
        if self.cfg.dataset.name == 'ImageNet' and torch.cuda.device_count()>1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            mp.spawn(dt.ddp_train, args=(self.classes, 0, self.log_path), nprocs=torch.cuda.device_count())
        else:
            dt.train(self.classes, 0, self.log_path)

        # test
        clf = self.getModel()
        clf.load_state_dict(torch.load(self.log_path+'/weight.pth', map_location="cpu"))
        strategy.update_model(clf)
        acc = np.zeros(n_rounds)
        acc[0], class_correct, class_total = dt.get_acc_on_set(test_dataset, self.classes, clf)
        for i in range(self.num_classes):
            mlflow.log_metric(self.classes[i], (class_correct[i]/class_total[i]), step=len(train_dataset))
        mlflow.log_metric('final_acc', acc[0], step=len(train_dataset))
        logging.info(f'training points: {len(train_dataset)}')
        logging.info(f'test accuracy: {round(acc[0]*100, 2)}')
        logging.info(f'trainning time: {time.time()-t}')

        print('***************************')
        print('Starting Training..')
        print('***************************')
        for rd in range(1, n_rounds):
            logging.info('#########################')
            logging.info(f'round{rd}')
            logging.info('#########################')
            t0 = time.time()
            idx = strategy.select(budget)
            t1 = time.time()

            # Adding new points to training set
            train_dataset = ConcatDataset([train_dataset, Subset(unlabeled_dataset, idx)])
            remain_idx = list(set(range(len(unlabeled_dataset)))-set(idx))
            unlabeled_dataset = Subset(unlabeled_dataset, remain_idx)

            strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
            dt = data_train(train_dataset, net, self.cfg.train_parameters, self.cfg.dataset)
            gc.collect()

            if self.cfg.dataset.name == 'ImageNet' and torch.cuda.device_count()>1:
                mp.spawn(dt.ddp_train, args=(self.classes, rd, self.log_path), nprocs=torch.cuda.device_count())
            else:
                dt.train(self.classes, rd, self.log_path)
            
            t2 = time.time()            
            clf = self.getModel()
            clf.load_state_dict(torch.load(self.log_path+'/weight.pth', map_location="cpu"), strict=False)
            strategy.update_model(clf)
            acc[rd], class_correct, class_total = dt.get_acc_on_set(test_dataset, self.classes, clf)
            mlflow.log_metric('final_acc', acc[rd], step=len(train_dataset))
            for i in range(self.num_classes):
                mlflow.log_metric(self.classes[i], 100*class_correct[i]/class_total[i], step=len(train_dataset))
            logging.info(f'training points: {len(train_dataset)}')
            logging.info(f'test accuracy: {round(acc[rd]*100, 2)}')
            logging.info(f'selection time: {t1-t0}')
            logging.info(f'training time: {t2-t1}')

        print('Training Completed!')

        torch.save(clf.state_dict(), self.log_path+'/weight.pth')
        print('Model Saved!')

        fig = plt.figure()
        plt.plot(acc)
        plt.title('accuracy')
        plt.ylabel('test Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True)
        fig.savefig(self.log_path+'/acc.png')

        with open(self.log_path+'/../../test.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([self.cfg.train_parameters.seed])
            writer.writerow(acc)

        for i in range(self.num_classes):
            logging.info('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))

# hydra
@hydra.main(config_name='base', config_path='configs', version_base='1.1')
def main(cfg : DictConfig):
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow_runname)
    log_path = os.getcwd()
    # mlflow
    with mlflow.start_run():
        # rechange cwd
        os.chdir(hydra.utils.get_original_cwd())
        tc = TrainClassifier(cfg, log_path)
        tc.train_classifier()

if __name__ == '__main__':
    main()