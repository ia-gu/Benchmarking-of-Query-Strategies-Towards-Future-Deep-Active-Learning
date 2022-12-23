import torch
from tqdm import tqdm

from src.simsiam.utils.progress_meter import ProgressMeter
from src.simsiam.utils.average_meter import AverageMeter

def train(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(train_loader, unit='batch', desc='| Pretrain |', dynamic_ncols=True)
    for _, (images, _) in enumerate(loop):
        image0 = images[0].to(device, non_blocking=True)
        image1 = images[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            p1, p2, z1, z2 = model(x1=image0, x2=image1)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), images[0].size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    progress.display(0)
