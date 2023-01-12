import sys

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
    logger: Logger
        logging manager
    """
    
    
    def __init__(self, training_dataset, net, cfg, dataset_cfg, logger):

        self.training_dataset = AddIndexDataset(training_dataset)
        self.net = net
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.logger = logger
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
        """	
        
        # If the dataset is visual inspection, calcurate binary metric
        if 'IntactRoad' in classes:
            ok_idx = 3
        elif 'OK' in classes:
            ok_idx = 1
        else:
            ok_idx = None
        self.clf = clf.eval().to(device=self.device)
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=self.cfg.batch_size, num_workers=8)
        loop = tqdm(loader_te, unit='batch', desc='| Test |', dynamic_ncols=True)

        binary_correct = [0.]*2; binary_total = [0.]*2
        class_correct = [0.]*len(classes); class_total = [0.]*len(classes)

        with torch.no_grad():
            for _, (x,y) in enumerate(loop):
                x, y = x.to(device=self.device), y.to(device=self.device)
                outputs = self.clf(x)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == y)
                if ok_idx==None:
                    for i in range(len(y)):
                        class_correct[y[i]] += c[i].item()
                        class_total[y[i]] += 1
                else:
                    for i in range(len(y)):
                        # OK data
                        if y[i]==ok_idx:
                            if predicted[i]==ok_idx:
                                binary_correct[0] += 1
                            binary_total[0] += 1
                        # NG data
                        else:
                            if not predicted[i]==ok_idx:
                                binary_correct[1] += 1
                            binary_total[1] += 1
                        # acc on each class
                        class_correct[y[i]] += c[i].item()
                        class_total[y[i]] += 1

        self.logger.write_test_log(class_correct, class_total, binary_correct, binary_total)


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

        self.logger.train_acc.append(accFinal/len(loader_tr.dataset))
        self.logger.train_loss.append(lossFinal/len(loader_tr.dataset))
        print('Epoch:' + str(epoch) + '- training accuracy:'+str(self.logger.train_acc[-1])+'- training loss:'+str(self.logger.train_loss[-1]))


    def train(self, classes):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        classes: list
            The list of each class name
        """

        print('Training..')

        # Make histgram of queried data
        self.logger.get_hist(self.training_dataset, classes)

        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

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
        print(f'use {len(device_ids)} gpus')
        self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
        self.clf = self.net.to(device=self.device)

        batch_size = self.cfg.batch_size
        batch_size *= len(device_ids)
        optimizer = optim.SGD(self.clf.parameters(), lr = self.cfg.lr, momentum=0.9, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.n_epoch)
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        self.clf.train()
        for epoch in range(self.cfg.n_epoch):
            self._train(epoch, loader_tr, optimizer, criterion, scaler)
            lr_sched.step()

        self.logger.write_train_log()
        self.logger.save_weight(self.clf)


    def ddp_train(self, rank, classes):

        """
        Initiates the training loop.

        rank: int
            The id of current GPU (Given automatically)
        Other requiered contents are same as the ddp_train

        """

        if rank==0:
            print('Training..')
            self.logger.get_hist(self.training_dataset, classes)
        self.device = rank
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

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

        batch_size = self.cfg.batch_size
        batch_size *= torch.cuda.device_count()
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, sampler=sampler)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        optimizer = optim.SGD(self.clf.parameters(), lr = self.cfg.lr, momentum=0.9, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_tr)*self.cfg.n_epoch/torch.cuda.device_count())

        dist.barrier()
        for epoch in range(self.cfg.n_epoch):
            self._train(epoch, loader_tr, optimizer, criterion, scaler)
            lr_sched.step()

        if rank==0:
            self.logger.write_train_log()
            self.logger.save_weight(self.clf)