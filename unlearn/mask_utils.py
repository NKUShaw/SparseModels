import numpy as np
import torch

def get_mask(model):
    mask = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            mask.append(np.where(tensor == 0, 0, 1))
    return mask

def compute_overlap(mask1, mask2):
    intersection_count = 0  
    union_count = 0    

    for m1, m2 in zip(mask1, mask2):
        if isinstance(m1, np.ndarray) or isinstance(m1, list):
            sub_intersection, sub_union = compute_overlap(m1, m2)
            intersection_count += sub_intersection
            union_count += sub_union
        else:
            if m1 == 1 and m2 == 1:
                intersection_count += 1
            if m1 == 1 or m2 == 1:
                union_count += 1
    return intersection_count, union_count

def count_parameters(model):
    total_params = 0  
    zero_params = 0   

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel() 
            zero_params += torch.sum(param == 0).item()

    return total_params, zero_params