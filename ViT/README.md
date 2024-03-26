# Reproducing ViT on CIFAR10 experiments (Nuclear norm)

The original [code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining). If you want to learn more, please refer to their document.
Please also note that you will need to install [accelerate](https://huggingface.co/docs/accelerate/index) package to run this code.

## Quick Start Guide
1. alter the GPU assignment in each configuration file (optimizer_model_on_dataset.sh) according to your machine.

2. run all the experiments

```
bash run.sh
``` 