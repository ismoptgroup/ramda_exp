# Core code for the experiments.

This directory contains:
 - optimizer.py: ProxSGD, ProxSSI, ProxGen(ProxAdamW), RMDA and RAMDA optimizer.
 - prox_fns.py: Group-Lasso proximal operator and newton-raphson solver for Group-Lasso weighted proximal operator.
 - solver.py: proximal gradient descent solver for Group-Lasso weighted proximal operator.
 - scheduler.py: stage-wise learning rate scheduler, momentum scheduler and restarting.
 - group.py: grouping the models into channel-wise, input-wise or non-regularized group. 