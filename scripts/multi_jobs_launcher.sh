make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=truncated_svd

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=truncated_svd

make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=truncated_svd

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=mlp_neurons

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=all PRUNE_METHOD=mlp_neurons

make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_tsvd0.96 \
  PRUNE_AMOUNT=0.96 PRUNE_TARGETS=al PRUNE_METHOD=mlp_neuronsl

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_tsvd0.98 \
  PRUNE_AMOUNT=0.98 PRUNE_TARGETS=all PRUNE_METHOD=mlp_neurons

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_tsvd0.98 \
  PRUNE_AMOUNT=0.98 PRUNE_TARGETS=all PRUNE_METHOD=mlp_neurons

make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_tsvd0.98 \
  PRUNE_AMOUNT=0.98 PRUNE_TARGETS=al PRUNE_METHOD=mlp_neuronsl

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
