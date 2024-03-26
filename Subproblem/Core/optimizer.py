# optimizer

import torch
import math
import time

from typing import Iterable, Tuple

from Core.solver import pgd_solver

class RAMDA(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, momentum: float = 1e-2, 
                 lambda_: float = 0.0, epsilon: float = 1e-6,
                 max_iters: int = 100, early_stopping: bool = True, rtol: float = 1e-8,
                 dim: tuple = None, dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if max_iters < 1:
            raise ValueError("Invalid max_iters value: {}".format(max_iters))
        if early_stopping != True and early_stopping != False:
            raise ValueError("Invalid early_stopping value: {}".format(early_stopping))
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))

        defaults = dict(lr=lr, momentum=momentum,
                        lambda_=lambda_, epsilon=epsilon,
                        max_iters=max_iters, early_stopping=early_stopping, rtol=rtol,
                        dim=dim, dim_=dim_)

        super(RAMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        grads = []
        ps = []
        iters = []
        times = [] 

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            epsilon = group['epsilon']
            max_iters = group['max_iters']
            early_stopping = group['early_stopping']
            rtol = group['rtol']
            dim = group['dim']
            dim_ = group['dim_']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['alpha'] = 0.0
                    p0 = state['initial_point'] = p.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    grad_sum_sq = state['grad_sum_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                else:
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                    grad_sum_sq = state['grad_sum_sq']
                state['step'] += 1
                step = state['step']
                step_size = lr*step**0.5
                state['alpha'] += step_size
                alpha = state['alpha']
                                    
                grad_sum.add_(grad, alpha=step_size)

                grad_sum_sq.add_(grad.square(), alpha=step_size)

                denom = grad_sum_sq.pow(1/3).add(epsilon)

                if lambda_ != 0.0 and dim is not None:  
                    start = time.time()
                    p_tilde, grad_, p_, iter_ = pgd_solver(p=p, 
                                                           p0=p0, 
                                                           v=grad_sum, 
                                                           denom=denom, 
                                                           alpha=alpha, 
                                                           lambda_=lambda_, 
                                                           dim=dim, 
                                                           dim_=dim_,
                                                           max_iters=max_iters, 
                                                           early_stopping=early_stopping, 
                                                           rtol=rtol)
                    end = time.time()
                    
                    grads.append(grad_)
                    ps.append(p_)
                    iters.append(iter_)
                    times.append(end-start)
                else:
                    p_tilde = p0.addcdiv(grad_sum, denom, value=-1.0)

                if momentum != 1.0:
                    p.mul_(1.0-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)
                
        return alpha, grads, ps, iters, times


class ProxAdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-8, weight_decay: float = 0.0, 
                 amsgrad: bool = False, lambda_: float = 0.0, 
                 max_iters: int = 100, early_stopping: bool = True, rtol: float = 1e-8,
                 dim: tuple = None, dim_: tuple = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= lambda_:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))
        if max_iters < 1:
            raise ValueError("Invalid max_iters value: {}".format(max_iters))
        if early_stopping != True and early_stopping != False:
            raise ValueError("Invalid early_stopping value: {}".format(early_stopping))
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))
            
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                        weight_decay=weight_decay, 
                        amsgrad=amsgrad, lambda_=lambda_,
                        max_iters=max_iters, early_stopping=early_stopping, rtol=rtol,
                        dim=dim, dim_=dim_)
        
        super(ProxAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            
    @torch.no_grad()
    def step(self):
        grads = []
        ps = []
        iters = []
        times = []
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            lambda_ = group['lambda_']
            max_iters = group['max_iters']
            early_stopping = group['early_stopping']
            rtol = group['rtol']
            dim = group['dim']
            dim_ = group['dim_']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                p.mul_(1.0-lr*weight_decay)

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    exp_avg = state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avg_sq = state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                bias_correction1 = 1.0-beta1**step
                bias_correction2 = 1.0-beta2**step

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(epsilon)
                else:
                    denom = (exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(epsilon)

                step_size = lr/bias_correction1
                    
                if lambda_ != 0.0 and dim is not None:
                    start = time.time()
                    p_tilde, grad_, p_, iter_ = pgd_solver(p=p, 
                                                           p0=p.clone().detach(), 
                                                           v=exp_avg.mul(step_size), 
                                                           denom=denom, 
                                                           alpha=lr, 
                                                           lambda_=lambda_, 
                                                           dim=dim, 
                                                           dim_=dim_, 
                                                           max_iters=max_iters, 
                                                           early_stopping=early_stopping,
                                                           rtol=rtol)  
                    end = time.time()
                        
                    p.copy_(p_tilde)   
                    
                    grads.append(grad_)
                    ps.append(p_)
                    iters.append(iter_)
                    times.append(end-start)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    
        return lr, grads, ps, iters, times