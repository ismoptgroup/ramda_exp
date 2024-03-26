# Reproducing Transformer-XL on WikiText-103 experiments (Group LASSO norm)

The original [code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL). If you want to learn more, please refer to their document.

## Quick Start Guide

1. download and preprocess the dataset.

```
bash getdata.sh
```

2. start training.

```
cd pytorch
bash Adam_Transformer-XL_on_WikiText-103.sh
bash ProxAdamW_Transformer-XL_on_WikiText-103.sh
bash ProxSGD_Transformer-XL_on_WikiText-103.sh
bash RAMDA_Transformer-XL_on_WikiText-103.sh
bash RMDA_Transformer-XL_on_WikiText-103.sh
```