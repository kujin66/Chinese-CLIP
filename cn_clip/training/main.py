from math import ceil
import os
import logging
from pathlib import Path
import json
import time
from time import gmtime, strftime
import importlib.util

import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from cn_clip.clip import load
from cn_clip.clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP
from cn_clip.training.train import train, evaluate
from cn_clip.training.data import get_data
from cn_clip.training.params import parse_args
from cn_clip.training.logger import setup_primary_logging, setup_worker_logging
from cn_clip.training.scheduler import cosine_lr
from transformers import CLIPModel, CLIPFeatureExtractor, AutoTokenizer, CLIPTextModel


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return args.rank == 0


# used to compare the pytorch version
def torch_version_str_compare_lessequal(version1, version2):
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]
    assert len(v1) == 3, "Cannot parse the version of your installed pytorch! ({})".format(version1)
    assert len(v2) == 3, "Illegal version specification ({}). Should be in 1.X.Y format.".format(version2)
    return sorted([v1, v2])[0] == v1


def main():
    args = parse_args()

    # Set distributed group
    args.local_device_rank = max(args.local_rank, 0)
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)

    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)    

    assert args.precision in ['amp', 'fp16', 'fp32']

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)

    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Build the CLIP model
    clip_model = CLIPKUN(args=args)
    clip_model.cuda(args.local_device_rank)
    clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.local_device_rank], find_unused_parameters=True)
    
    # Initialize dataset and dataloader
    data = get_data(args, epoch_id=0, max_txt_length=args.context_length)

    # Initialize optimizer and lr scheduler
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        num_batches = data["train"].dataloader.num_batches
        if args.max_steps is not None:
            args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
        else:
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
        total_steps = args.max_steps
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # Log and save hyper-params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params_{}.txt".format(time_suffix))
        with open(params_file, "w", encoding="utf-8") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    if args.local_device_rank == 0:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    logging.info(f"Use GPU: {args.local_device_rank} for training")

    # Note for mask_ratio
    if is_master(args) and args.mask_ratio > 0 and args.vision_model in ['RN50']:
        logging.info("Note: mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer. " + \
            "It will not function for ResNet backbone.")    

    # Optionally resume from a checkpoint
    start_epoch = 0
    steps = 0
    cudnn.benchmark = True
    cudnn.deterministic = False
    args.should_save = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and is_master(args)

    for epoch in range(start_epoch, args.max_epochs):
        if is_master(args) == 0:
            logging.info(f'Start epoch {epoch + 1}')
        num_steps_this_epoch = train(model, data, epoch, optimizer, scaler, scheduler, args, steps)
        steps += num_steps_this_epoch

        # Saving checkpoints.
        if args.should_save and num_steps_this_epoch > 0:
            if (epoch + 1) == args.max_epochs or (
                args.save_epoch_frequency > 0 and ((epoch + 1) % args.save_epoch_frequency) == 0
            ):
                t1 = time.time()
                clip_model.module.save_pretrained_model(save_dir)

        if epoch + 1 < args.max_epochs:
            data = get_data(args, epoch_id=epoch + 1, max_txt_length=args.context_length)
                
if __name__ == "__main__":
    main()
