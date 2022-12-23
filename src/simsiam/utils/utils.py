import torch

import shutil
import math


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')

def adjust_learning_rate(optimizer, init_lr, epoch, cfg):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.train_parameters.n_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] == init_lr if 'fix_lr' in param_group and param_group['fix_lr'] else cur_lr
    print(f'current lr: {param_group["lr"]}')
