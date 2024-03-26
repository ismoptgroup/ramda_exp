# Reproducing VGG19 and ResNet50 on CIFAR10 and CIFAR100 experiments

This directory contains:
 - train.py: training and evaluating VGG19 and ResNet50 on CIFAR10 and CIFAR100 experiments. 
 - model.py: VGG19 and ResNet50 for CIFAR10 and CIFAR100.
 - prepare.py: creating model initializations, downloading datasets and making folders.
 - optimizer_model_on_dataset.sh: configuration files to reproduce the experiments.
 - run.sh: running all the experiments.

## Quick Start Guide
1. modify the path argument in both the train.py and prepare.py files to specify the location for storing model initializations, datasets, checkpoints, and logs, if necessary.

2. alter the GPU assignment in each configuration file (optimizer_model_on_dataset.sh) according to your machine.

3. run all the experiments

```
bash run.sh
``` 