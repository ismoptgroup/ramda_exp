# Reproducing MLP on FashionMNIST experiments (Nuclear norm)

This directory contains:
 - train.py: training and evaluating MLP on FashionMNIST experiments. 
 - model.py: MLP for FashionMNIST.
 - prepare.py: creating model initializations, downloading datasets and making folders.
 - optimizer_MLP_on_FashionMNIST.sh: configuration files to reproduce the experiments.
 - run.sh: running all the experiments.

## Quick Start Guide
1. modify the path argument in both the train.py and prepare.py files to specify the location for storing model initializations, datasets, checkpoints, and logs, if necessary.

2. run all the experiments

```
bash run.sh
``` 