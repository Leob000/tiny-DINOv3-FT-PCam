make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_qk0.95 \
  PRUNE_AMOUNT=0.95 PRUNE_TARGETS=all PRUNE_METHOD=qk_product

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_qk0.94 \
  PRUNE_AMOUNT=0.94 PRUNE_TARGETS=all PRUNE_METHOD=qk_product

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_qk0.93 \
  PRUNE_AMOUNT=0.93 PRUNE_TARGETS=all PRUNE_METHOD=qk_product

# make seval CHECKPOINT=lora.pt \
#   WANDB_RUN_NAME=eval_lora_qk0.96 \
#   PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=qk_product
#
# make seval CHECKPOINT=head_only.pt \
#   WANDB_RUN_NAME=eval_head_qk0.96 \
#   PRUNE_AMOUNT=0.96 PRUNE_TARGETS=al PRUNE_METHOD=qk_product

###

# make sbaseline METHOD=head_only \
#   WARMUP_STEPS=400 \
#   TRAIN_NORMS_BIAS=both LR_HEAD=2.5e-4 LR_NORMS_BIAS=1e-4 \
#   RESOLUTION=224
#
# make sbaseline METHOD=lora \
#   LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
#   LORA_TARGETS="q_proj,k_proj,v_proj,o_proj" \
#   LR_HEAD=5e-4 LR_LORA=2.5e-4 LR_NORMS_BIAS=2.5e-4 \
#   WARMUP_STEPS=400 \
#   TRAIN_NORMS_BIAS=both \
#   RESOLUTION=224

# make sbaseline METHOD=fullft \
#   LR=5e-5 WEIGHT_DECAY=0.05 \
#   WARMUP_STEPS=400 GRAD_CLIP=1.0 LABEL_SMOOTHING=0.05 \
#   LR_HEAD=2.5e-4 \
#   RESOLUTION=224

#20250914-224440
# make sbaseline METHOD=head_only \
#   WARMUP_STEPS=400 \
#   TRAIN_NORMS_BIAS=both LR_HEAD=2.5e-4 LR_NORMS_BIAS=1e-4
#
# make sbaseline METHOD=lora \
#   LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
#   LORA_TARGETS="q_proj,k_proj,v_proj,o_proj" \
#   LR_HEAD=5e-4 LR_LORA=2.5e-4 LR_NORMS_BIAS=2.5e-4 \
#   WARMUP_STEPS=400 \
#   TRAIN_NORMS_BIAS=both
#
# make sbaseline METHOD=fullft \
#   LR=5e-5 WEIGHT_DECAY=0.05 \
#   WARMUP_STEPS=400 GRAD_CLIP=1.0 LABEL_SMOOTHING=0.05 \
#   LR_HEAD=2.5e-4

# Old runs
# make sbaseline METHOD=head_only \
#   TRAIN_NORMS_BIAS=both LR_HEAD=5e-4 LR_NORMS_BIAS=2.5e-4
#
# make sbaseline METHOD=lora \
#   LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
#   LORA_TARGETS="q_proj,k_proj,v_proj,o_proj" \
#   TRAIN_NORMS_BIAS=both
#
# make sbaseline METHOD=fullft \
#   LR=1e-4 WEIGHT_DECAY=0.05 \
#   WARMUP_STEPS=400 GRAD_CLIP=1.0 LABEL_SMOOTHING=0.05 \
#   LR_HEAD=5e-4
