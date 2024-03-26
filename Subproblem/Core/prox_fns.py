# prox_fns

import torch

from torch import Tensor

def prox_glasso(p: Tensor, 
                eta: Tensor, 
                alpha: float, 
                lambda_: float, 
                dim: tuple):
    if dim == (0,2) or dim == (0):
        group_size = p.numel()/p.shape[1]
    elif dim == (1,2) or dim == (1):
        group_size = p.numel()/p.shape[0]
    
    threshold = alpha*lambda_*(group_size**0.5)
    if eta is not None:
        threshold = eta.mul(threshold)
    norm = torch.nn.functional.relu(torch.linalg.norm(p, dim=dim, keepdim=True).sub(threshold))
    p.mul_(norm.div(norm.add(threshold)))

    return norm