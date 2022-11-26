import os
import matplotlib.pyplot as plt
import sys
import logging
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


sys.path.append('./')  

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, label, index
    
    def __len__(self):
        return len(self.wrapped_dataset)

#custom training
class data_train:

    """
    Provides a configurable training loop for AL.
    
    Parameters
    ----------
    training_dataset: torch.utils.data.Dataset
        The training dataset to use
    net: torch.nn.Module
        The model to train
    cfg: DictConfig
        Additional arguments to control the training loop
    dataset_cfg: DictConfig
        use for ssl section
    """
    
    
    def __init__(self, training_dataset, net, cfg, dataset_cfg):

        self.training_dataset = AddIndexDataset(training_dataset)
        self.net = net
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        
        self.n_pool = len(training_dataset)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def get_acc_on_set(self, test_dataset, classes, clf):
        
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        classes: list
            The list of each class name
        clf: torch.nn.Module
            Model
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        class_correct: list
            The list of correct numbers of each class
        class_total: list
            The list of total numbers of each class
        """	
        
        self.clf = clf

        if test_dataset is None:
            raise ValueError("Test data not present")
        
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=self.cfg.batch_size, num_workers=8)
        self.clf.eval()

        class_correct = [0.]*len(classes)
        class_total = [0.]*len(classes)
        loop = tqdm(loader_te, unit='batch', desc='| Test |', dynamic_ncols=True)

        with torch.no_grad():
            self.clf = self.clf.to(self.device)
            for _, (x,y) in enumerate(loop): 
                x, y = x.to(device=self.device), y.to(device=self.device)
                outputs = self.clf(x)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == y)
                # Get acc of each class (for imbalanced data analysis)
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        accFinal = sum(class_correct) / sum(class_total)

        return accFinal, class_correct, class_total

    def get_binary_acc_on_set(self, test_dataset, classes, clf):
  
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        classes: list
            The list of each class name
        clf: torch.nn.Module
            Model
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        class_correct: list
            The list of correct numbers of each class
        class_total: list
            The list of total numbers of each class
        """	
        
        self.clf = clf

        if test_dataset is None:
            raise ValueError("Test data not present")
        
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=self.cfg.batch_size, num_workers=8)
        self.clf.eval()

        binary_correct = [0.]*2
        binary_total = [0.]*2
        class_correct = [0.]*len(classes)
        class_total = [0.]*len(classes)
        loop = tqdm(loader_te, unit='batch', desc='| Test |', dynamic_ncols=True)

        with torch.no_grad():
            self.clf = self.clf.to(self.device)
            for _, (x,y) in enumerate(loop): 
                x, y = x.to(device=self.device), y.to(device=self.device)
                outputs = self.clf(x)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == y)
                # GAPs
                if len(classes)>2:
                    for i in range(len(y)):
                        # OK data
                        if y[i]==3:
                            if predicted[i]==3:
                                binary_correct[0] += 1
                            binary_total[0] += 1
                        # NG data
                        else:
                            if not predicted[i]==3:
                                binary_correct[1] += 1
                            binary_total[1] += 1
                        # acc on each class
                        label = y[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                # KSDD2
                else:
                    for i in range(len(y)):
                        if y[i]==1:
                            if predicted[i]==1:
                                binary_correct[0] += 1
                            binary_total[0] += 1
                        else:
                            if predicted[i]==0:
                                binary_correct[1] += 1
                            binary_total[1] += 1
                        label = y[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

                    

        accFinal = sum(class_correct) / sum(class_total)
        binary_accFinal = sum(binary_correct) / sum(binary_total)
        del loader_te, loop, outputs, predicted
        return accFinal, class_correct, class_total, binary_accFinal, binary_correct, binary_total


    def _train(self, epoch, loader_tr, optimizer, criterion, scaler):
        accFinal = 0.
        lossFinal = 0.
        
        loop = tqdm(loader_tr, unit='batch', desc='| Training |', dynamic_ncols=True)
        for _, (x, y, _) in enumerate(loop):
            x, y = x.to(device=self.device), y.to(device=self.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = self.clf(x)
                loss = criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lossFinal += loss.item()

        self.acc.append(accFinal/len(loader_tr.dataset))
        self.loss.append(lossFinal/len(loader_tr.dataset))
        print('Epoch:' + str(epoch) + '- training accuracy:'+str(self.acc[-1])+'- training loss:'+str(self.loss[-1]))


    def train(self, classes, rd, log_path):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        classes: list
            The list of each class name
        rd: int
            Current round
        log_path: str
            path for log file
        """        

        self.acc = []
        self.loss = []

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        n_epoch = self.cfg.n_epoch

        if self.cfg.isreset:
            if self.cfg.ssl:
                checkpoint = torch.load('./weights/'+self.dataset_cfg.name+'/'+str(self.cfg.seed)+'/checkpoint.pth.tar', map_location="cpu")
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder up to before the embedding layer
                    if k.startswith('encoder') and not k.startswith('encoder.fc'):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                self.net.load_state_dict(state_dict, strict=False)
            else:
                self.clf = self.net.apply(weight_reset)
        
        # DataParallel
        device_ids = []
        for i in range(torch.cuda.device_count()):
            device_ids.append(i)
        self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
        self.clf = self.net.to(device=self.device)

        # batch size for each gpu
        batch_size = self.cfg.batch_size
        batch_size *= len(device_ids)

        optimizer = optim.SGD(self.clf.parameters(), lr = self.cfg.lr, momentum=0.9, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        epoch = 1
        
        # Make histgram of queried data
        hist = [0]*len(classes)
        os.makedirs(os.path.join(log_path, 'hist'), exist_ok=True)
        for _, (_, y, _) in enumerate(loader_tr):
            for i in y:
                hist[i] += 1
        fig = plt.figure()
        plt.bar(classes, hist, width=0.9)
        plt.xlabel('classes')
        plt.ylabel('number of queried data')
        fig.savefig(os.path.join(log_path, 'hist', str(rd)+'.png'))
        plt.close()

        self.clf.train()
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(n_epoch):
            self._train(epoch, loader_tr, optimizer, criterion, scaler)
            lr_sched.step()
            epoch += 1

        for i in range(n_epoch):
            mlflow.log_metric('acc', self.acc[i])
            mlflow.log_metric('loss', self.loss[i])
            logging.info('Epoch:' + str(i) + '- training accuracy:'+str(self.acc[i])+'- training loss:'+str(self.loss[i]))

        torch.save(self.clf.module.state_dict(), log_path+'/weight'+str(rd)+'.pth')


    def ddp_train(self, rank, classes, rd, log_path):

        """
        Initiates the training loop.

        rank: int
            The id of current GPU (Given automatically)
        Other requiered contents are same as the ddp_train

        """

        if rank==0:
            print('Training..')
        self.device = rank

        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        n_epoch = self.cfg.n_epoch

        if self.cfg.isreset:
            if self.cfg.ssl:
                checkpoint = torch.load('./weights/'+self.dataset_cfg.name+'/'+str(self.cfg.seed)+'/checkpoint.pth.tar', map_location="cpu")
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder up to before the embedding layer
                    if k.startswith('encoder') and not k.startswith('encoder.fc'):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                self.net.load_state_dict(state_dict, strict=False)
            else:
                self.clf = self.net.apply(weight_reset)

        # DDP
        dist.init_process_group(backend='nccl', rank=rank, world_size=torch.cuda.device_count())
        self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(device=self.device)
        self.clf = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[rank], find_unused_parameters=True)
        sampler = DistributedSampler(self.training_dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True)

        # batch size for each gpu
        batch_size = self.cfg.batch_size
        batch_size *= torch.cuda.device_count()

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, sampler=sampler)
        
        optimizer = optim.SGD(self.clf.parameters(), lr = self.cfg.lr, momentum=0.9, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_tr)*n_epoch/torch.cuda.device_count())
        
        # histgram of queried data
        if rank==0:
            hist = [0]*len(classes)
            os.makedirs(os.path.join(log_path, 'hist'), exist_ok=True)
            for _, (_, y, _) in enumerate(loader_tr):
                for i in y:
                    hist[i] += 1
            fig = plt.figure()
            plt.bar(classes, hist, width=0.9)
            plt.xlabel('classes')
            plt.ylabel('number of queried data')
            fig.savefig(os.path.join(log_path, 'hist', str(rd)+'.png'))
        
        epoch = 1
        
        dist.barrier()

        while epoch <= n_epoch: 

            self._train(loader_tr, optimizer)

            lr_sched.step()
            
            epoch += 1
        
        if rank==0:
            for i in range(n_epoch):
                mlflow.log_metric('acc', self.acc[i])
                mlflow.log_metric('loss', self.loss[i])
                logging.info('Epoch:' + str(i) + '- training accuracy:'+str(self.acc[i])+'- training loss:'+str(self.loss[i]))
            torch.save(self.clf.module.state_dict(), log_path+'/weight'+str(rd)+'.pth')