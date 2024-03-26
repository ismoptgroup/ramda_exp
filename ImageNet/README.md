# Reproducing ResNet50 on ImageNet experiments (Group LASSO norm)

This directory contains:
 - train.py: training and evaluating ResNet50 on ImageNet experiments. 
 - prepare.py: creating model initializations and making folders.
 - optimizer_ResNet50_on_ImageNet.sh: configuration files to reproduce the experiments.

## Quick Start Guide
1. download and extract the [ImageNet dataset](https://www.image-net.org/challenges/LSVRC/2012/index.php). We recommend this [code](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) to extract the dataset.

2. modify the path argument in both the train.py and prepare.py files to specify the location for storing model initializations, checkpoints, and logs, if necessary. Also, change data_root argument to the folder containing the downloaded ImageNet dataset.

3. alter the GPU assignment in each configuration file (optimizer_ResNet50_on_ImageNet.sh) according to your machine. You can refer this [link](https://horovod.readthedocs.io/en/latest/running_include.html). Please note that the batch_size argument means the batch size per GPU. Therefore, if you change the number of GPUs, you will also need to adjust the batch_size argument to maintain a consistent global batch size of 256.

4. run the experiments

```
python prepare.py
bash MSGD_ResNet50_on_ImageNet.sh
bash ProxAdamW_ResNet50_on_ImageNet.sh
bash ProxSGD_ResNet50_on_ImageNet.sh
bash RAMDA_ResNet50_on_ImageNet.sh
bash RMDA_ResNet50_on_ImageNet.sh
``` 