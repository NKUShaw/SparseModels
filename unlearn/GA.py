import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def GA(data_loader, model, criterion, optimizer, epoch):
    model.train()
    for _ in range(epoch):
        for i, (image, target) in enumerate(data_loader):
            image = image.cuda()
            target = target.cuda()
    
            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            output = output_clean.float()
            loss = loss.float()
    return model