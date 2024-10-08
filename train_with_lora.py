#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 MaPO authors and The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.

import logging
import math
import os
import shutil
from pathlib import Path
import random
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.nn.functional as F
import diffusers
from args import parse_args
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.training_utils import _set_state_dict_into_text_encoder
from utils import (
    collate_fn,
    compute_loss,
    compute_time_ids,
    encode_prompt,
    get_dataset_preprocessor,
    get_wandb_url,
    import_model_class_from_model_name_or_path,
    log_validation,
    compute_dpo_loss
)
import datetime

logger = get_logger(__name__)


def main(args):
    if args.lora_rank is None:
        raise ValueError("`--lora_rank` cannot be undefined when using LoRA training.")
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed+accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    if args.dpo_training or args.sft_training:
        if args.train_unet:
            ref_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
            )
            ref_unet.requires_grad_(False)
        if args.train_text_encoder:
            ref_text_encoder_one = text_encoder_cls_one.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
            ref_text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
            )
            ref_text_encoder_one.requires_grad_(False)
            ref_text_encoder_two.requires_grad_(False)
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet and text_encoders to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if args.dpo_training or args.sft_training:
        if args.train_unet:
            ref_unet.to(accelerator.device, dtype=weight_dtype)
        if args.train_text_encoder:
            ref_text_encoder_one.to(accelerator.device, dtype=weight_dtype)
            ref_text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    # Set up LoRA.
    if args.train_unet:
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
    
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
    
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        if args.train_unet:
            for param in unet.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

        if args.train_text_encoder:
            for param in text_encoder_one.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            for param in text_encoder_two.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    if args.gradient_checkpointing:
        if args.train_unet:
            unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.enable_gradient_checkpointing()
            text_encoder_two.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLLoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None
        
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(input_dir)
        if args.train_unet:
            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = []
    if args.train_unet:
        params_to_optimize += list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        split=args.dataset_split_name,
    )

    # Preprocessing the datasets.
    preprocess_train_fn = get_dataset_preprocessor(args, tokenizer_one, tokenizer_two)

    with accelerator.main_process_first():
        train_dataset = train_dataset.with_transform(preprocess_train_fn)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    if args.train_unet and args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_unet:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_text_encoder:
        text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if args.train_unet:
        unet.train()
    if args.train_text_encoder:
        text_encoder_one.train()
        text_encoder_two.train()
    implicit_acc_accumulated = 0.0
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            if args.train_text_encoder:
                model_accum = [text_encoder_one, text_encoder_two]
            if args.train_unet:
                model_accum = unet
            with accelerator.accumulate(model_accum):
                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                idx1 = random.randint(0, batch["pixel_values"].shape[1] // 6 - 1)
                value_w = batch["pixel_values"][:, idx1 * 3 : (idx1 + 1) * 3, :, :].to(dtype=vae.dtype)
                
                if args.dpo_training or args.mapo_training:
                    idx2 = random.randint(batch["pixel_values"].shape[1] // 6, batch["pixel_values"].shape[1] // 3 - 1)
                    value_l = batch["pixel_values"][:, idx2 * 3 : (idx2 + 1) * 3, :, :].to(dtype=vae.dtype)
                    feed_pixel_values = torch.cat((value_w, value_l), dim=0)
                if args.sft_training:
                    feed_pixel_values = value_w
                
                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.dpo_training or args.mapo_training:
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                bsz = latents.shape[0] 
                if args.dpo_training or args.mapo_training:
                    bsz = bsz // 2
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                )
                if args.dpo_training or args.mapo_training:
                    timesteps = timesteps.repeat(2)
                
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # time ids
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(args, accelerator, weight_dtype, s, c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )
                if args.dpo_training or args.mapo_training:
                    add_time_ids = add_time_ids.repeat(2, 1)
                
                # Get the text embedding for conditioning
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]
                )
                
                if args.dpo_training or args.mapo_training:
                    prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                    pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)
                    
                if (args.dpo_training or args.sft_training) and args.train_text_encoder:
                    ref_prompt_embeds, ref_pooled_prompt_embeds = encode_prompt(
                        [ref_text_encoder_one, ref_text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]
                    )
                    if args.dpo_training:
                        ref_prompt_embeds = ref_prompt_embeds.repeat(2, 1, 1)
                        ref_pooled_prompt_embeds = ref_pooled_prompt_embeds.repeat(2, 1)

                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                    ).sample
                    if args.dpo_training or args.sft_training:
                        if args.train_unet and args.train_text_encoder:
                            ref_pred = ref_unet(
                                noisy_model_input,
                                timesteps,
                                ref_prompt_embeds,
                                added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": ref_pooled_prompt_embeds},
                            ).sample
                        elif args.train_unet:
                            ref_pred = ref_unet(
                                noisy_model_input,
                                timesteps,
                                prompt_embeds,
                                added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                            ).sample
                        elif args.train_text_encoder:
                            ref_pred = unet(
                                noisy_model_input,
                                timesteps,
                                ref_prompt_embeds,
                                added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": ref_pooled_prompt_embeds},
                            ).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.sft_training:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss_ref = F.mse_loss(ref_pred.float(), target.float(), reduction="mean")
                elif args.dpo_training:
                    loss, model_losses_w, model_losses_l, implicit_acc, ref_losses_w, ref_losses_l = compute_dpo_loss(
                        args=args, model_pred=model_pred, target=target, ref_pred=ref_pred
                    )
                    avg_acc = accelerator.gather(implicit_acc).mean().detach().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps
                elif args.mapo_training:
                    loss, model_losses_w, model_losses_l, ratio_losses = compute_loss(
                        args=args, noise_scheduler=noise_scheduler, model_pred=model_pred, target=target
                    )
                else:
                    ValueError("Unknown training type")
                # loss = loss / args.gradient_accumulation_steps
                
                # Backprop.
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                
                # if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args, unet=unet, vae=vae, text_encoder_one=text_encoder_one, text_encoder_two=text_encoder_two, accelerator=accelerator, weight_dtype=weight_dtype, epoch=epoch
                        )
            if args.sft_training:
                logs = {
                    "epoch": args.num_train_epochs * progress_bar.n / args.max_train_steps,
                    "total loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "loss_ref": loss_ref.detach().item(),
                }
            elif args.dpo_training:
                logs = {
                    "epoch": args.num_train_epochs * progress_bar.n / args.max_train_steps,
                    "total loss": loss.detach().item(),
                    "implicit_acc": implicit_acc,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "model_losses_w": model_losses_w.mean().detach().item(),
                    "model_losses_l": model_losses_l.mean().detach().item(),
                    "losses_diff_w": model_losses_w.mean().detach().item() - ref_losses_w.mean().detach().item(),
                    "losses_diff_l": model_losses_l.mean().detach().item() - ref_losses_l.mean().detach().item(),
                }
                implicit_acc_accumulated = 0.0
            elif args.mapo_training:
                logs = {
                    "epoch": args.num_train_epochs * progress_bar.n / args.max_train_steps,
                    "total loss": loss.detach().item(),
                    "Win Score": ((args.snr_value * model_losses_w) / (torch.exp(args.snr_value * model_losses_w) - 1)).mean().detach().item(),
                    "Lose Score": ((args.snr_value * model_losses_l) / (torch.exp(args.snr_value * model_losses_l) - 1)).mean().detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "OR loss": -ratio_losses.mean().detach().item(),
                    "model_losses_w": model_losses_w.mean().detach().item(),
                    "model_losses_l": model_losses_l.mean().detach().item(),
                }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.train_unet:
            unet = accelerator.unwrap_model(unet)
            unet = unet.to(torch.float32)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        else:
            unet_lora_state_dict = None
        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)

            text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
            
        StableDiffusionXLLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        # Final validation?
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                vae=vae,
                text_encoder_one=None,
                text_encoder_two=None,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            wandb_info = get_wandb_url()
            with open(os.path.join(args.output_dir, "README.md"), "w") as f:
                f.write(wandb_info)

            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
