from __future__ import print_function
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import grad
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()

def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param


def batch_grads_to_vec(parameters):
    vec = []
    for param in parameters:
        # vec.append(param.view(1, -1))
        vec.append(param.reshape(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(1, -1).size(1)
        grads.append(vec[:size].view(param.size()))
        vec = vec[size:]
    return grads
    
def grad_batch(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    params = [p for p in model.parameters() if p.requires_grad]
    grad_batch = [torch.zeros_like(p).cpu() for p in params]
    num = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        num += targets.shape[0]
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        grad_mini = list(grad(criterion(outputs, targets), params))
        for i in range(len(grad_batch)):
            grad_batch[i] += grad_mini[i].cpu().detach()

    for i in range(len(grad_batch)):
        grad_batch[i] /= num

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    grad_reg = list(grad(lam * l2_reg, params))
    for i in range(len(grad_batch)):
        grad_batch[i] += grad_reg[i].cpu().detach()
    return [p.to(device) for p in grad_batch]

def grad_batch_approx(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss += criterion(outputs, targets)

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss + lam * l2_reg, params))


def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)
    return return_grads


def inverse_hvp(y, w, v):

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    vec_first_grads = batch_grads_to_vec(first_grads)
    
    hessian_list = []
    for i in range(vec_first_grads.shape[0]):
        sec_grads = grad(vec_first_grads[i], w, retain_graph=True)
        hessian_list.append(batch_grads_to_vec(sec_grads).unsqueeze(0))
    
    hessian_mat = torch.cat(hessian_list, 0)
    return torch.linalg.solve(hessian_mat, v.view(-1, 1))


def newton_update(g, batch_size, res_set, lam, gamma, model, s1, s2, scale, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    H_res = [torch.zeros_like(p) for p in g]
    for i in tqdm(range(s1)):
        H = [p.clone() for p in g]
        sampler = RandomSampler(res_set, replacement=True, num_samples=batch_size * s2)
        # Create a data loader with the sampler
        res_loader = DataLoader(res_set, batch_size=batch_size, sampler=sampler)
        res_iter = iter(res_loader)
        for j in range(s2):
            data, target = next(res_iter)
            data, target = data.to(device), target.to(device)
            z = model(data)
            loss = criterion(z, target)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            # Add L2 regularization to the loss
            
            loss += (lam + gamma) * l2_reg
            H_s = hvp(loss, params, H)
            
            with torch.no_grad():
                for k in range(len(params)):
                    H[k] = H[k] + g[k] - H_s[k] / scale
                if j % int(s2 / 10) == 0:
                    print(f'Epoch: {j}, Sum: {sum([torch.norm(p, 2).item() for p in H])}')
        for k in range(len(params)):
            H_res[k] = H_res[k] + H[k] / scale
    return [p / s1 for p in H_res]

def CU(model_original, retrain_dataset, retrain_loader, weight_decay, device, unlearn_batch_size, gamma, s1, s2, scale, std):
    g = grad_batch(retrain_loader, weight_decay, model_original, device)
    delta = newton_update(g, unlearn_batch_size, retrain_dataset, weight_decay, gamma, model_original, s1, s2, scale, device)
    for i, param in enumerate(model_original.parameters()):
        param.data.add_(-delta[i] + std * torch.randn(param.data.size()).to(device))
    model = model_original
    return model
