# Reproducing subproblem solver experiments (Group LASSO norm)

This directory contains:
 - Core folder: core code for the subproblem experiments. 
 - train.py: training and evaluating logistic regression on MNIST, VGG19 on CIFAR10, and ResNet50 on CIFAR100. 
 - model.py: logistic regression for MNIST, VGG19 for CIFAR10, and ResNet50 for CIFAR100.
 - prepare.py: creating model initializations, downloading datasets and making folders.
 - optimizer_100_model_on_dataset_NoEarlyStopping.sh: configuration files to reproduce the no early stopping strategy experiments.
 - optimizer_100_model_on_dataset_EarlyStopping.sh: configuration files to reproduce the early stopping strategy experiments. 
 - run.sh: running all the experiments.

## Quick Start Guide
1. modify the path argument in both the train.py and prepare.py files to specify the location for storing model initializations, datasets, checkpoints, and logs, if necessary.

2. alter the GPU assignment in each configuration file (optimizer_100_model_on_dataset_NoEarlyStopping.sh and optimizer_100_model_on_dataset_EarlyStopping.sh) according to your machine.

3. run all the experiments

```
bash run.sh
``` 