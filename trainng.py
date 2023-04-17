import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import time
import os
from torchvision.utils import save_image


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def training_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.to(device)
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        image, label = data[0]['image'].to(device),data[1].to(device)
        # optimizer.zero_grad()
        batch_size = image.size(0)

        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()

        if (step + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss

def val_epoch(model,dataloader, device, epoch):
    model.to(device)
    model.eval()
    with torch.no_grad():

        dataset_size = 0
        running_loss = 0.0
        total_batch = 0
        total_acc = 0
        coll = 0
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            image, label = data[0]['image'].to(device),data[1].to(device)

            batch_size = image.size(0)

            outputs = model(image)
            loss = criterion(outputs, label)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            coll = torch.eq(outputs.argmax(dim=1), label).sum().float().item()
            running_acc = coll/batch_size

            epoch_loss = running_loss / dataset_size
            total_acc += coll
            total_batch += batch_size
            bar.set_postfix(Epoch=epoch, Val_Loss=epoch_loss, Run_acc = running_acc)
        gc.collect()

    return epoch_loss