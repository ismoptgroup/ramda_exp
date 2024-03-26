# prox_fns

import torch
import torch.nn.functional as F

from math import sqrt
from torch import Tensor
from torch.jit import Future
from typing import Tuple, List, Iterable

def prox_glasso(p: Tensor, 
                eta: Tensor, 
                alpha: float, 
                lambda_: float, 
                dim: tuple):  
    if dim == (0,2,3) or dim == (1,2,3):
        if dim == (0,2,3):
            dim = (0,2)
        elif dim == (1,2,3):
            dim == (1,2)
        p = p.view(p.shape[0], p.shape[1], -1)
        if eta is not None:
            eta = eta.view(eta.shape[0], eta.shape[1], -1)
            
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

def prox_quant(p: Tensor, eta: Tensor, alpha: float, lambda_: float):
    threshold = alpha*lambda_
    if eta is not None:
        threshold = eta.mul(threshold)
    p_sign = p.sign()
    diff_p_p_sign = p.sub(p_sign)
    norm = torch.nn.functional.relu(diff_p_p_sign.abs().sub(threshold))
    p.copy_(p_sign+diff_p_p_sign.sign().mul(norm))

    return norm

def prox_nuclear(p: Tensor, alpha: float, lambda_: float):
    threshold = alpha*lambda_
    U, S_hat, V_T = torch.linalg.svd(p, full_matrices=False)
    S = torch.nn.functional.relu(S_hat.sub(threshold))
    p.copy_(U.matmul(S.diagflat()).matmul(V_T))

    return S

# https://github.com/tristandeleu/pytorch-structured-sparsity/blob/master/proxssi/l1_l2.py
@torch.jit.script
def prox_l2(param: Tensor,
            lambda_: float) -> Tensor:
    norm = F.relu(torch.linalg.norm(param, dim=(0, 1)) - lambda_)
    norm.div_(norm + lambda_)

    param.data.mul_(norm)
    return param

@torch.jit.script
def prox_l1_l2(groups: List[Tensor],
               lr: float,
               lambda_: float,
               reweight: bool = True):
    futures: List[Future[Tensor]] = []

    if reweight:
        lambdas: List[float] = [sqrt(param.numel() / param.size(-1)) * lambda_
                                for param in groups]
    else:
        lambdas: List[float] = [lambda_ for _ in groups]

    for param, lambda_rw in zip(groups, lambdas):
        futures.append(torch.jit.fork(prox_l2, param, lr * lambda_rw))

    for future in futures:
        torch.jit.wait(future)


@torch.jit.script
def _newton_raphson_step(theta: Tensor,
                         weights: Tensor,
                         num: Tensor,
                         lambda_: float) -> Tensor:
    den: Tensor = theta + lambda_ / weights
    func: Tensor = torch.sum(num / (den ** 2), dim=(0, 1)) - 1
    step: Tensor = 0.5 * (func / torch.sum(num / (den ** 3), dim=(0, 1)))

    theta.add_(step).clamp(min=0.)
    return func

@torch.jit.script
def _newton_raphson(param: Tensor,
                    weights: Tensor,
                    lambda_: float,
                    atol: float = 1e-7,
                    rtol: float = 1e-7,
                    max_iters: int = 100) -> Tensor:
    num_groups: int = param.size(-1)

    d_max: Tensor = torch.max(weights.reshape((-1, num_groups)), dim=0)[0]
    norms_weighted: Tensor = torch.linalg.norm(param * weights, dim=(0, 1))

    theta: Tensor = (norms_weighted - lambda_) / d_max
    num: Tensor = param ** 2

    prev_value: Tensor = param.new_zeros((num_groups,))
    for _ in range(max_iters):
        value: Tensor = _newton_raphson_step(theta, weights, num, lambda_)
        if torch.all(value.abs() < atol):
            break
        if torch.all((prev_value - value).abs() < rtol):
            break
        prev_value = value

    return theta

@torch.jit.script
def prox_l2_weighted(param: Tensor,
                     weights: Tensor,
                     lambda_: float,
                     atol: float = 1e-7,
                     rtol: float = 1e-7,
                     max_iters: int = 100) -> Tensor:
    mask: Tensor = (torch.linalg.norm(param * weights, dim=(0, 1)) > lambda_)
    if torch.any(mask):
        theta: Tensor = _newton_raphson(param[:,:,mask], weights[:,:,mask], lambda_,
                                        atol=atol, rtol=rtol, max_iters=max_iters)

        factor: Tensor = torch.zeros_like(param)
        theta_weights: Tensor = weights[:,:,mask] * theta
        factor[:,:,mask] = theta_weights / (theta_weights + lambda_)

        param.data.mul_(factor)
    else:
        param.data.zero_()
    return param

@torch.jit.script
def prox_l1_l2_weighted(groups: List[Tensor],
                        weights: List[Tensor],
                        lr: float,
                        lambda_: float,
                        atol: float = 1e-7,
                        rtol: float = 1e-7,
                        max_iters: int = 100,
                        reweight: bool = True):
    futures: List[Future[Tensor]] = []

    if reweight:
        lambdas: List[float] = [sqrt(param.numel() / param.size(-1)) * lambda_
                                for param in groups]
    else:
        lambdas: List[float] = [lambda_ for _ in groups]

    for param, weight, lambda_rw in zip(groups, weights, lambdas):
        futures.append(torch.jit.fork(prox_l2_weighted, param,
            weight, lr * lambda_rw, atol=atol, rtol=rtol, max_iters=max_iters))

    for future in futures:
        torch.jit.wait(future)
