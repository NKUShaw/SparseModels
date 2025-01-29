import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def compute_fisher_matrix(model, data_loader, criterion, device):
    fisher_matrix = {}
    for name, param in model.named_parameters():
        fisher_matrix[name] = torch.zeros_like(param)

    model.eval() 
    for imgs, targets in data_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        loss = criterion(output, targets)
        
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            fisher_matrix[name] += param.grad.data.pow(2) / len(data_loader)

    return fisher_matrix

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = torch.autograd.grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)
    
def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr : curr + length].view(param.shape)
            curr += length

def woodfisher(model, train_dl, device, criterion, v, N=1000):
    model.eval()
    k_vec = torch.clone(v)
    o_vec = None
    for idx, (data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def Wfisher(retain_loader, forget_loader, model, criterion, device, alpha=1.0, N=1000):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()
    for i, (data, label) in enumerate(tqdm(forget_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, (data, label) in enumerate(tqdm(retain_loader)):
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num
    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2
    
    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
        N=N
    )
    
    apply_perturb(model, alpha * perturb)
    
    return model