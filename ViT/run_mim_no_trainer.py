#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import warnings
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForMaskedImageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import ViTConfig

import sys 
sys.path.append("..")

from Core.group import group_model
from Core.optimizer import ProxSGD, ProxAdamW, RMDA, RAMDA, RMDA_D, RAMDA_D


""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM)
without using HuggingFace Trainer.
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a simple Masked Image Modeling task"
    )
    parser.add_argument(
        "--tuning",
        action='store_true', 
        default=False
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help="Name of a dataset from the datasets package",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default=None,
        help="The column name of the images in the files. If not set, will try to use 'image' or 'img'.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--validation_dir",
        type=None,
        default=None,
        help="A folder containing the validation data.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation.",
    )
    parser.add_argument(
        "--mask_patch_size",
        type=int,
        default=32,
        help="The size of the square patches to use for masking.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.6,
        help="Percentage of patches to mask.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help=(
            "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--config_name_or_path",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=(
            "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store (cache) the pretrained models/datasets downloaded from the hub",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--image_processor_name",
        type=str,
        default=None,
        help="Name or path of preprocessor config.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
            "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
        ),
    )
    parser.add_argument(
        "--use_auth_token",
        type=bool,
        default=None,
        help="The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="The size (resolution) of each image. If not specified, will use `image_size` of the configuration.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration.",
    )
    parser.add_argument(
        "--encoder_stride",
        type=int,
        default=None,
        help={"help": "Stride to use for the encoder."},
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The initial learning rate for [`AdamW`] optimizer.",
    )
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument('--regularization', type=str, default='nuclear')
    parser.add_argument("--lambda_", type=float, default=0.0)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--max-iters', type=int, default=5)
    parser.add_argument("--momentum", type=float, default=1e-1)
    parser.add_argument('--milestones', type=int, nargs='+', default=[100, 200])
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=300,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "multistep"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    args = parser.parse_args()

    if args.tuning:
        if args.optimizer != "AdamW":
            args.output_dir = './Results/'+args.optimizer+'_'+str(args.learning_rate)+'_'+str(args.lambda_)+'_'+str(args.num_train_epochs)
        else:
            args.output_dir = './Results/'+args.optimizer+'_'+str(args.learning_rate)+'_'+str(args.weight_decay)+'_'+str(args.num_train_epochs)
    else:
        args.output_dir = './Results/'+args.optimizer+'_'+str(args.seed)

    # Sanity checks
    data_files = {}
    if args.train_dir is not None:
        data_files["train"] = args.train_dir
    if args.validation_dir is not None:
        data_files["val"] = args.validation_dir
    args.data_files = data_files if data_files else None

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


def main():
    args = parse_args()

    if args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        args.token = args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mim_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize our dataset.
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        data_files=args.data_files,
        cache_dir=args.cache_dir,
        token=args.token,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in ds.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = ds["train"].train_test_split(args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Create config
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name_or_path:
        config = AutoConfig.from_pretrained(args.config_name_or_path, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        if args.model_type == "vit":
            config = ViTConfig(hidden_size=192, num_hidden_layers=4, num_attention_heads=4, intermediate_size=768)
        else:
            config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")

    # make sure the decoder_type is "simmim" (only relevant for BEiT)
    if hasattr(config, "decoder_type"):
        config.decoder_type = "simmim"

    # adapt config
    args.image_size = args.image_size if args.image_size is not None else config.image_size
    args.patch_size = args.patch_size if args.patch_size is not None else config.patch_size
    args.encoder_stride = args.encoder_stride if args.encoder_stride is not None else config.encoder_stride

    config.update(
        {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "encoder_stride": args.encoder_stride,
        }
    )

    # create image processor
    if args.image_processor_name:
        image_processor = AutoImageProcessor.from_pretrained(args.image_processor_name, **config_kwargs)
    elif args.model_name_or_path:
        image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        IMAGE_PROCESSOR_TYPES = {
            conf.model_type: image_processor_class for conf, image_processor_class in IMAGE_PROCESSOR_MAPPING.items()
        }
        image_processor = IMAGE_PROCESSOR_TYPES[args.model_type]()

    # create model
    if args.model_name_or_path:
        model = AutoModelForMaskedImageModeling.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedImageModeling.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
        )

    column_names = ds["train"].column_names

    if args.image_column_name is not None:
        image_column_name = args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original SimMIM paper
    # source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB")),
            RandomResizedCrop(args.image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    # create mask generator
    mask_generator = MaskGenerator(
        input_size=args.image_size,
        mask_patch_size=args.mask_patch_size,
        model_patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
        which patches to mask."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        examples["mask"] = [mask_generator() for i in range(len(examples[image_column_name]))]

        return examples

    if args.max_train_samples is not None:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    ds["train"].set_transform(preprocess_images)

    if args.max_eval_samples is not None:
        ds["validation"] = ds["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
    # Set the validation transforms
    ds["validation"].set_transform(preprocess_images)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        ds["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        ds["validation"],
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if args.optimizer == "AdamW":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
    optimizer_grouped_parameters = group_model(model=model, name="ViT", lambda_=args.lambda_)
    if args.optimizer == "ProxSGD":
        optimizer = ProxSGD(params=optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            regularization=args.regularization)
    elif args.optimizer == "ProxAdamW":
        optimizer = ProxAdamW(params=optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              regularization=args.regularization,
                              rtol=args.rtol,
                              max_iters=args.max_iters)
    elif args.optimizer == "RMDA":
        optimizer = RMDA(params=optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         momentum=args.momentum,
                         regularization=args.regularization)
    elif args.optimizer == "RAMDA":
        optimizer = RAMDA(params=optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          regularization=args.regularization,
                          rtol=args.rtol,
                          max_iters=args.max_iters)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type != "multistep":
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mim_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    lrs = []
    momentums = []
    training_losses = []
    validation_losses = []
    low_rank_levels = []
    epoch_times = []
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(ds['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(args.max_train_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    momentum_step_count = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        start = time.time()
        model.train()
        if args.lr_scheduler_type == "multistep":    
            if epoch in args.milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.gamma
                    with torch.no_grad():
                        for p in param_group['params']:
                            optimizer.state[p]['step'] = 0
                            optimizer.state[p]['alpha'] = 0.0
                            optimizer.state[p]['initial_point'].copy_(p.clone().detach())
                            optimizer.state[p]['grad_sum'].copy_(torch.zeros_like(p, memory_format=torch.preserve_format))
                            if 'grad_sum_sq' in optimizer.state[p]:
                                optimizer.state[p]['grad_sum_sq'].copy_(torch.zeros_like(p, memory_format=torch.preserve_format)) 
                    if 'epsilon' in param_group:
                        param_group['epsilon'] *= args.gamma
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            if args.lr_scheduler_type == "multistep":
                if completed_steps < args.num_warmup_steps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = (args.learning_rate/args.num_warmup_steps)*(completed_steps+1)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                if args.lr_scheduler_type != "multistep":
                    lr_scheduler.step() 
                else:                       
                    if epoch+1 > args.milestones[-1]:
                        momentum_step_count += 1
                        for param_group in optimizer.param_groups:
                            param_group['momentum'] = min(args.momentum*momentum_step_count**0.5, 1.0) 
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        end = time.time()

        training_loss = total_loss/len(train_dataloader)

        logger.info(f"epoch {epoch}: training_loss: {training_loss}")

        model.eval()

        for param_group in optimizer.param_groups:
            lr = param_group['lr'] 
            if args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
                momentum = param_group['momentum']
            elif args.optimizer == "AdamW" or args.optimizer == "ProxAdamW":
                momentum = param_group['betas']
                
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        logger.info(f"epoch {epoch}: eval_loss: {eval_loss}")
        validation_loss = eval_loss

        if args.optimizer != "AdamW" and args.lambda_ != 0.0:
            S_hats = []
            for group in optimizer_grouped_parameters:
                dim = group["dim"]
                lambda_ = group["lambda_"]
                if dim == (0):
                    for p in group["params"]:
                        S_hats.append(optimizer.state[p]['S'])
     
            nonzero = 0.0
            num_el = 0.0
            for S_hat in S_hats:
                nonzero += S_hat.count_nonzero().item()
                num_el += S_hat.numel()
            low_rank_level = 1.0-(nonzero/num_el)
        else:
            low_rank_level = 0.0

        epoch_time = end-start

        logger.info(f"epoch {epoch}: low_rank_level: {low_rank_level}")

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        lrs.append(lr)
        momentums.append(momentum)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        low_rank_levels.append(low_rank_level)
        epoch_times.append(epoch_time)

        f = open(args.output_dir+'/'+args.optimizer+'_ViT_on_CIFAR10_presentation_'+str(args.seed)+'.txt', 'w+') 
        
        f.write("final training loss: {}".format(training_loss)+'\n')
        f.write("final validation loss: {}".format(validation_loss)+'\n')
        f.write("final low rank level: {}".format(low_rank_level)+'\n') 

        if epoch > (args.num_train_epochs-50):
            index = validation_losses.index(min(validation_losses[args.num_train_epochs-50:len(validation_losses)]))
            f.write("epoch with the best validation loss: {}".format(index+1)+'\n')
            f.write("the best validation loss: {}".format(validation_losses[index])+'\n')
            f.write("low rank level with the best validation loss: {}".format(low_rank_levels[index])+'\n')
        
        f.write("lr: {}".format(args.learning_rate)+'\n')
        if optimizer != "AdamW":
            f.write("lambda_: {}".format(args.lambda_)+'\n')
        else:
            f.write("weight decay: {}".format(args.weight_decay)+'\n')

        f.write("regularization: {}".format(args.regularization)+'\n')
        f.write("rtol: {}".format(args.rtol)+'\n')
        f.write("max iters: {}".format(args.max_iters)+'\n')

        if args.optimizer == "ProxSGD" or args.optimizer == "RMDA" or args.optimizer == "RAMDA":
            for i, r in enumerate(zip(lrs, momentums, training_losses, validation_losses, low_rank_levels, epoch_times)):
                f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f}\ttraining loss:{:<20.15f}\tvalidation loss:{:<20.15f}\tlow rank level:{:<20.15f}\tepoch time:{:<20.15f}".format((i+1), r[0], r[1], r[2], r[3], r[4], r[5])+'\n')

        elif args.optimizer == "AdamW" or args.optimizer == "ProxAdamW":
            for i, r in enumerate(zip(lrs, momentums, training_losses, validation_losses, low_rank_levels, epoch_times)):
                f.write("epoch:{:<5d}\tlearning rate:{:<20.15f}\tmomentum:{:<20.15f},{:<20.15f}\ttraining loss:{:<20.15f}\tvalidation loss:{:<20.15f}\tlow rank level:{:<20.15f}\tepoch time:{:<20.15f}".format((i+1), r[0], r[1][0], r[1][1], r[2], r[3], r[4], r[5])+'\n')    
        
        f.close()

        f = open(args.output_dir+'/'+args.optimizer+'_ViT_on_CIFAR10_forplotting_'+str(args.seed)+'.txt', 'w+')
        
        f.write('learning rate\n')
        for i, r in enumerate(lrs):
             f.write("epoch {}: {}".format((i+1), r)+'\n')
    
        f.write('momentum\n')
        for i, r in enumerate(momentums):
             f.write("epoch {}: {}".format((i+1), r)+'\n')

        f.write('training loss\n')
        for i, r in enumerate(training_losses):
             f.write("epoch {}: {}".format((i+1), r)+'\n')

        f.write('validation loss\n')
        for i, r in enumerate(validation_losses):
             f.write("epoch {}: {}".format((i+1), r)+'\n')

        f.write('low rank level\n')
        for i, r in enumerate(low_rank_levels):
            f.write("epoch {}: {}".format((i+1), r)+'\n')

        f.write('epoch time\n')
        for i, r in enumerate(epoch_times):
            f.write("epoch {}: {}".format((i+1), r)+'\n')
        
        f.close()

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()