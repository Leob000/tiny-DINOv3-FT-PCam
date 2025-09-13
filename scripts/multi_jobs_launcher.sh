make smodel METHOD=head_only \
  TRAIN_NORMS_BIAS=both LR_HEAD=1e-3

make smodel METHOD=lora \
  LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.05 \
  LORA_TARGETS="q_proj,k_proj,v_proj,o_proj" \
  TRAIN_NORMS_BIAS=both LR_LORA=5e-4 LR_HEAD=1e-3

# make smodel METHOD=fullft \
#   LR=1e-4 WEIGHT_DECAY=0.05 \
#   WARMUP_STEPS=200 GRAD_CLIP=1.0 LABEL_SMOOTHING=0.05 \
#   LR_HEAD=5e-4
