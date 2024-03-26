# Reproducing Tacotron2 on LJSpeech experiments (Group LASSO norm)

The original [code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2). If you want to learn more, please refer to their document.

## Quick Start Guide

1. download and preprocess the dataset.
```
bash scripts/prepare_dataset.sh
```

2. preprocess the datasets for Tacotron 2 training
```
bash scripts/prepare_mels.sh
```

3. start training.
```
bash Adam_Tacotron2_on_LJSpeech.sh
bash ProxAdamW_Tacotron2_on_LJSpeech.sh
bash ProxSGD_Tacotron2_on_LJSpeech.sh
bash RAMDA_Tacotron2_on_LJSpeech.sh
bash RMDA_Tacotron2_on_LJSpeech.sh
```