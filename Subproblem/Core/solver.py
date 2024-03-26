# solver

import torch

from torch import Tensor
from Core.prox_fns import prox_glasso

def pgd_solver(p: Tensor, 
               p0: Tensor, 
               v: Tensor, 
               denom: Tensor, 
               alpha: float, 
               lambda_: float, 
               dim: tuple, 
               dim_: tuple, 
               max_iters: int,
               early_stopping: bool,
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
        
        if early_stopping:
            norm_ = torch.linalg.norm(p_, dim=dim)
            previous = (diff_.mul(v_).sum()+
                        diff_denom_.mul(diff_).sum().mul(1/2)+
                        norm_.sum().mul(threshold)).item() 
        
        for i in range(max_iters):     
            p_.addcmul_(v_.add(diff_denom_), eta_, value=-1)
            norm_ = prox_glasso(p=p_, eta=eta_, alpha=alpha, lambda_=lambda_, dim=dim)
                                
            diff_ = p_.sub(p0_)
            diff_denom_ = diff_.mul(denom_)
            
            if early_stopping:
                current = (diff_.mul(v_).sum()+
                           diff_denom_.mul(diff_).sum().mul(1/2)+
                           norm_.sum().mul(threshold)).item()
            
                if (previous-current)/(abs(current)+1.0) < rtol and i > 0:
                    break
                                
                previous = current
            
        p_tilde.index_copy_(dim_, index, p_)
        
    if p_shape is not None:    
        p_tilde = p_tilde.view(p_shape)
        
    if mask.any().item():        
        return p_tilde, v_.add(diff_denom_), p_.clone().detach(), i+1
    else:
        return p_tilde, None, None, 0
