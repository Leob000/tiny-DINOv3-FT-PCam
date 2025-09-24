# WANDB_RUN_NAME=eval_lora_attn-mlp-tsvd_0.89-0.975-0.975 \
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=test_qint8_head_only \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
  QUANTIZE=int8

# WANDB_RUN_NAME=eval_head_only_attn-mlp-tsvd_0.89-0.975-0.975 \
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=test_qbf16_head_only \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
  QUANTIZE=bf16

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
