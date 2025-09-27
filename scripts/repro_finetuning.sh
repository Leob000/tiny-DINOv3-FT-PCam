#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--wandb]" >&2
}

USE_WANDB=false
while [[ $# -gt 0 ]]; do
  case "$1" in
  --wandb)
    USE_WANDB=true
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage
    exit 1
    ;;
  esac
done

if $USE_WANDB; then
  export WANDB="--wandb"
else
  export WANDB=""
fi

make sbaseline METHOD=head_only \
  WARMUP_STEPS=400 \
  TRAIN_NORMS_BIAS=both LR_HEAD=2.5e-4 LR_NORMS_BIAS=1e-4 \
  RESOLUTION=224

make sbaseline METHOD=lora \
  LORA_R=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
  LORA_TARGETS="q_proj,k_proj,v_proj,o_proj" \
  LR_HEAD=5e-4 LR_LORA=2.5e-4 LR_NORMS_BIAS=2.5e-4 \
  WARMUP_STEPS=400 \
  TRAIN_NORMS_BIAS=both \
  RESOLUTION=224

make sbaseline METHOD=fullft \
  LR=5e-5 WEIGHT_DECAY=0.05 \
  WARMUP_STEPS=400 GRAD_CLIP=1.0 LABEL_SMOOTHING=0.05 \
  LR_HEAD=2.5e-4 \
  RESOLUTION=224
