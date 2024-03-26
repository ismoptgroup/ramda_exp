# solver

import torch

from torch import Tensor
from Core.prox_fns import prox_glasso, prox_nuclear

def pgd_solver_glasso(p: Tensor, 
                      p0: Tensor, 
                      v: Tensor, 
                      denom: Tensor, 
                      alpha: float, 
                      lambda_: float, 
                      dim: tuple, 
                      dim_: tuple, 
                      max_iters: int, 
                      rtol: tuple): 
    p_shape = None
    if dim == (0,2,3) or dim == (1,2,3):
        if dim == (0,2,3):
            dim = (0,2)
        elif dim == (1,2,3):
            dim == (1,2)
        p_shape = p.shape
        p = p.view(p.shape[0], p.shape[1], -1)
        p0 = p0.view(p0.shape[0], p0.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        denom = denom.view(denom.shape[0], denom.shape[1], -1)
        
    if dim == (0,2) or dim == (0):
        group_size = p.numel()/p.shape[1]
    elif dim == (1,2) or dim == (1):
        group_size = p.numel()/p.shape[0]
    
    threshold = alpha*lambda_*(group_size**0.5)
    mask = torch.linalg.norm(denom.mul(p0).sub(v), dim=dim, keepdim=True).gt(threshold)
        
    p_tilde = torch.zeros_like(p)
    
    if mask.any().item():
        if dim == (0,2) or dim == (0):
            index = mask.float().nonzero(as_tuple=True)[1]
        elif dim == (1,2) or dim == (1):
            index = mask.float().nonzero(as_tuple=True)[0]
            
        p_ = p.index_select(dim_, index)
        v_ = v.index_select(dim_, index)
        p0_ = p0.index_select(dim_, index)
        denom_ = denom.index_select(dim_, index)
        
        eta_ = denom_.amax(dim=dim, keepdim=True).reciprocal()  

        diff_ = p_.sub(p0_)
        diff_denom_ = diff_.mul(denom_)
        norm_ = torch.linalg.norm(p_, dim=dim)
        previous = (diff_.mul(v_).sum()+
                    diff_denom_.mul(diff_).sum().mul(1/2)+
                    norm_.sum().mul(threshold)).item() 
        
        for i in range(max_iters):     
            p_.addcmul_(v_.add(diff_denom_), eta_, value=-1)
            norm_ = prox_glasso(p=p_, eta=eta_, alpha=alpha, lambda_=lambda_, dim=dim)
                                
            diff_ = p_.sub(p0_)
            diff_denom_ = diff_.mul(denom_)
            current = (diff_.mul(v_).sum()+
                       diff_denom_.mul(diff_).sum().mul(1/2)+
                       norm_.sum().mul(threshold)).item()
            
            if (previous-current)/(abs(current)+1.0) < rtol and i > 0:
                break
                                
            previous = current
            
        p_tilde.index_copy_(dim_, index, p_)
        
    if p_shape is not None:    
        p_tilde = p_tilde.view(p_shape)
        
    return p_tilde



def pgd_solver_nuclear(p: Tensor, 
                       p0: Tensor, 
                       v: Tensor, 
                       denom: Tensor, 
                       alpha: float, 
                       lambda_: float, 
                       max_iters: int, 
                       rtol: tuple): 
    threshold = alpha*lambda_
        
    eta = denom.amax().reciprocal()  

    diff = p.sub(p0)
    diff_denom = diff.mul(denom)
    U, S_hat, V_T = torch.linalg.svd(p)
    S = torch.nn.functional.relu(S_hat.sub(threshold))
    previous = (diff.mul(v).sum()+
                diff_denom.mul(diff).sum().mul(1/2)+
                S.sum().mul(threshold)).item() 
    p_tilde = p.clone().detach()
        
    for i in range(max_iters):     
        p_tilde.addcmul_(v.add(diff_denom), eta, value=-1.0)
        S = prox_nuclear(p=p_tilde, alpha=eta, lambda_=threshold)
                                
        diff = p_tilde.sub(p0)
        diff_denom = diff.mul(denom)
        current = (diff.mul(v).sum()+
                   diff_denom.mul(diff).sum().mul(1/2)+
                   S.sum().mul(threshold)).item()
            
        if (previous-current)/(abs(current)+1.0) < rtol and i > 0:
            break
                                
        previous = current
        
    return p_tilde, S
