# optimizer

import torch
import math

from typing import Callable, Iterable, Optional, Tuple

from Core.prox_fns import prox_glasso, prox_quant, prox_nuclear, prox_l1_l2_weighted
from Core.solver import pgd_solver_glasso, pgd_solver_nuclear

class ProxSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-1, momentum: float = 1e-1, lambda_: float = 0.0, 
                 regularization: str = "glasso", dim: tuple = None, dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))

        defaults = dict(lr=lr, momentum=momentum, lambda_=lambda_, 
                        regularization=regularization, dim=dim, dim_=dim_)

        super(ProxSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            regularization = group['regularization']
            dim = group['dim']
            dim_ = group['dim_']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                    
                state = self.state[p]
                if len(state) == 0:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach().mul(momentum)
                    state['S'] = None
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(1.0-momentum).add_(d_p, alpha=momentum)
                d_p = buf

                p.sub_(d_p, alpha=lr)

                if lambda_ != 0.0 and dim is not None:
                    if regularization == "glasso":
                        prox_glasso(p=p, eta=None, alpha=lr, lambda_=lambda_, dim=dim)
                    elif regularization == "nuclear":
                        state['S'] = prox_nuclear(p=p, alpha=lr, lambda_=lambda_)

        return loss

class RMDA(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-1, momentum: float = 1e-2, lambda_: float = 0.0, 
                 regularization: str = "glasso", dim: tuple = None, dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))

        defaults = dict(lr=lr, momentum=momentum, lambda_=lambda_, 
                        regularization=regularization, dim=dim, dim_=dim_)

        super(RMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            regularization = group['regularization']
            dim = group['dim']
            dim_ = group['dim_']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    step = state['step'] = 0
                    alpha = state['alpha'] = 0.0
                    p0 = state['initial_point'] = p.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['S'] = None
                else:  
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                    
                state['step'] += 1
                step = state['step']
                scaling = step**0.5
                state['alpha'] += lr*scaling
                alpha = state['alpha']

                grad_sum.add_(grad, alpha=lr*scaling)

                p_tilde = p0.sub(grad_sum, alpha=1/scaling)

                if lambda_ != 0.0 and dim is not None:
                    if regularization == "glasso":
                        prox_glasso(p=p_tilde, eta=None, alpha=alpha/scaling, lambda_=lambda_, dim=dim)
                    elif regularization == "nuclear":
                        state['S'] = prox_nuclear(p=p_tilde, alpha=alpha/scaling, lambda_=lambda_)

                if momentum != 1.0:
                    p.mul_(1.0-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)

        return loss

class RAMDA(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, momentum: float = 1e-2, 
                 lambda_: float = 0.0, epsilon: float = 1e-6,
                 regularization: str = "glasso",
                 max_iters: int = 100, rtol: float = 1e-8,
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
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))

        defaults = dict(lr=lr, momentum=momentum,
                        lambda_=lambda_, epsilon=epsilon,
                        regularization=regularization,
                        max_iters=max_iters, rtol=rtol,
                        dim=dim, dim_=dim_)

        super(RAMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            epsilon = group['epsilon']
            regularization = group['regularization']
            max_iters = group['max_iters']
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
                    state['S'] = None
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
                    if regularization == "glasso":
                        p_tilde = pgd_solver_glasso(p=p, 
                                                    p0=p0, 
                                                    v=grad_sum, 
                                                    denom=denom, 
                                                    alpha=alpha, 
                                                    lambda_=lambda_, 
                                                    dim=dim, 
                                                    dim_=dim_,
                                                    max_iters=max_iters, 
                                                    rtol=rtol)
                    elif regularization == "nuclear":
                        p_tilde, state['S'] = pgd_solver_nuclear(p=p, 
                                                                 p0=p0, 
                                                                 v=grad_sum, 
                                                                 denom=denom, 
                                                                 alpha=alpha, 
                                                                 lambda_=lambda_, 
                                                                 max_iters=max_iters, 
                                                                 rtol=rtol)
                        
                else:
                    p_tilde = p0.addcdiv(grad_sum, denom, value=-1.0)

                if momentum != 1.0:
                    p.mul_(1.0-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)
                
        return loss


class ProxAdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-8, weight_decay: float = 0.0, 
                 amsgrad: bool = False, lambda_: float = 0.0, 
                 regularization: str = "glasso",
                 max_iters: int = 100, rtol: float = 1e-8,
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
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))
            
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                        weight_decay=weight_decay, 
                        amsgrad=amsgrad, lambda_=lambda_,
                        regularization=regularization,
                        max_iters=max_iters, rtol=rtol,
                        dim=dim, dim_=dim_)
        
        super(ProxAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            lambda_ = group['lambda_']
            regularization = group['regularization']
            max_iters = group['max_iters']
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
                    state['S'] = None
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
                    if regularization == "glasso":
                        p_tilde = pgd_solver_glasso(p=p, 
                                                    p0=p.clone().detach(), 
                                                    v=exp_avg.mul(step_size), 
                                                    denom=denom, 
                                                    alpha=lr, 
                                                    lambda_=lambda_, 
                                                    dim=dim, 
                                                    dim_=dim_, 
                                                    max_iters=max_iters, 
                                                    rtol=rtol)          
                        
                        p.copy_(p_tilde)
                    elif regularization == "nuclear":
                        p_tilde, state['S'] = pgd_solver_nuclear(p=p, 
                                                                 p0=p.clone().detach(), 
                                                                 v=exp_avg.mul(step_size), 
                                                                 denom=denom, 
                                                                 alpha=lr, 
                                                                 lambda_=lambda_, 
                                                                 max_iters=max_iters, 
                                                                 rtol=rtol) 
                        p.copy_(p_tilde)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    
        return loss

class RMDA_D(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-1, momentum: float = 1e-2,
                 T: int = 1, power: int = 1,
                 dim: tuple = None, dim_: tuple = None):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if T < 0:
            raise ValueError("Invalid T value: {}".format(T))        
        if power < 1:
            raise ValueError("Invalid power value: {}".format(power))
            
        defaults = dict(lr=lr, momentum=momentum, 
                        T=T, power=power,
                        dim=dim, dim_=dim_)

        super(RMDA_D, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        p_tildes = {}
        params = []
        for group in self.param_groups:
            lr = group['lr']
            dim = group['dim']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    t = state['t'] = 1
                    step = state['step'] = 0
                    alpha = state['alpha'] = 0.0
                    p0 = state['initial_point'] = p.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['norm'] = p.clone().detach()
                else:
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                state['t'] += 1
                t = state['t']
                state['step'] += 1
                step = state['step']
                scaling = step**0.5
                step_size = lr*scaling
                state['alpha'] += step_size
                alpha = state['alpha']

                grad_sum.add_(grad, alpha=step_size)

                p_tilde = p0.sub(grad_sum, alpha=1/scaling)
                p_tildes[p] = p_tilde
                
                if dim is not None:
                    params.append(p_tilde.view(-1))
                    
        T = group['T'] 
        power = group['power']
        params = torch.cat(params)
        k = math.floor(params.numel()*((1.0-t/T)**power))
        threshold = torch.topk(params.abs(), k)[0][-1] 
        lambda_ = threshold/(alpha/scaling)

        for group in self.param_groups:
            momentum = group['momentum']
            dim = group['dim']
            for p in group['params']:
                state = self.state[p]
                alpha = state['alpha']
                step = state['step']
                scaling = step**0.5
                p_tilde = p_tildes[p]
                if dim is not None:
                    state['norm'] = prox_quant(p=p_tilde, alpha=alpha/scaling, lambda_=lambda_)
                if momentum != 1.0:
                    p.mul_(1-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)

        return loss

class RAMDA_D(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, momentum: float = 1e-1, epsilon: float = 1e-6, 
                 T: int = 1, power: int = 1,
                 dim: tuple = None, dim_: tuple = None):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if T < 0:
            raise ValueError("Invalid T value: {}".format(T))        
        if power < 1:
            raise ValueError("Invalid power value: {}".format(power))
            
        defaults = dict(lr=lr, momentum=momentum, epsilon=epsilon, 
                        T=T, power=power,
                        dim=dim, dim_=dim_)

        super(RAMDA_D, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            epsilon = group['epsilon']
            T = group['T'] 
            power = group['power']
            dim = group['dim']

            p_tildes = {}
            denoms = {}
            params = []
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    t = state['t'] = 1
                    step = state['step'] = 0
                    alpha = state['alpha'] = 0.0
                    p0 = state['initial_point'] = p.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    grad_sum_sq = state['grad_sum_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['norm'] = p.clone().detach()
                else:
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                    grad_sum_sq = state['grad_sum_sq']
                state['t'] += 1
                t = state['t']
                state['step'] += 1
                step = state['step']
                scaling = step**0.5
                step_size = lr*scaling
                state['alpha'] += step_size
                alpha = state['alpha']

                grad_sum.add_(grad, alpha=step_size)
                
                grad_sum_sq.addcmul_(grad, grad, value=step_size)
                
                denom = grad_sum_sq.pow(1/3).add_(epsilon)
                denoms[p] = denom

                p_tilde = p0.mul(denom).sub(grad_sum)
                
                p_tildes[p] = p_tilde
                if dim is not None:
                    params.append(p_tilde.view(-1))
            
        T = group['T'] 
        power = group['power']
        params = torch.cat(params)
        k = math.floor(params.numel()*((1.0-t/T)**power))
        threshold = torch.topk(params.abs(), k)[0][-1] 
        lambda_ = threshold/alpha

        for group in self.param_groups:
            momentum = group['momentum']
            dim = group['dim']
            for p in group['params']:
                state = self.state[p]
                alpha = state['alpha']
                p_tilde = p_tildes[p]
                denom = denoms[p]
                if dim is not None:
                    state['norm'] = prox_quant(p=p_tilde, alpha=alpha, lambda_=lambda_)
                p_tilde.div_(denom)
                if momentum != 1.0:
                    p.mul_(1-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)

        return loss
    
# https://github.com/tristandeleu/pytorch-structured-sparsity/blob/master/proxssi/optimizers/adamw_hf.py
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        groups_fn: Optional[Callable] = None,
        penalty: str = 'group_mcp',
        prox_kwargs: dict = {},
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if penalty not in ['l1_l2', 'group_mcp']:
            raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(penalty))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        groups_fn=groups_fn, penalty=penalty, prox_kwargs=prox_kwargs)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            denoms, params = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                denoms.append(denom)
                params.append(p)

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

            if group['groups_fn'] is not None:
                groups_fn = group['groups_fn']
                params = groups_fn(params)
                weights = groups_fn(denoms)
                assert len(params) == len(weights)

                if group['penalty'] == 'l1_l2':
                    prox_fn = prox_l1_l2_weighted
                elif group['penalty'] == 'group_mcp':
                    prox_fn = prox_group_mcp_weighted
                else:
                    raise ValueError('Unknown penalty {0}'.format(group['penalty']))

                prox_kwargs = group['prox_kwargs']
                prox_kwargs['lr'] = group['lr']

                with torch.no_grad():
                    prox_fn(params, weights, **prox_kwargs)

        return loss