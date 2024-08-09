CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_with_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="mapo" \
  --mixed_precision="fp16" \
  --dataset_name=Jialuo21/Phys_data \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --lora_rank=8 \
  --use_8bit_adam \
  --learning_rate=1e-7 \
  --scale_lr \
  --mapo_training \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --max_train_steps=2000 \
  --checkpointing_steps=20 \
  --seed="0" \
  --beta_mapo=0.05 \
  --run_validation --validation_steps=50 \
  --report_to="wandb" \
  --train_unet
  # --resume_from_checkpoint="latest"
  # --gradient_checkpointing 