# Large XPs
## No pruning
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_only_noPrune_noQ \
  PRUNE_METHOD=none

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_noPrune_noQ \
  PRUNE_METHOD=none

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_noPrune_noQ \
  PRUNE_METHOD=none

## Pruning
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_only_attn-mlp-tsvd_0.89-0.975-0.975_noQ \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_attn-mlp-tsvd_0.89-0.975-0.975_noQ \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_attn-mlp-tsvd_0.89-0.975-0.975_noQ \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975

## Quantize
### int8
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_only_noPrune_int8Q \
  QUANTIZE=int8 \
  PRUNE_METHOD=none

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_noPrune_int8Q \
  QUANTIZE=int8 \
  PRUNE_METHOD=none

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_noPrune_int8Q \
  QUANTIZE=int8 \
  PRUNE_METHOD=none

### bf16
#### No prune
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_only_noPrune_bf16Q \
  QUANTIZE=bf16 \
  PRUNE_METHOD=none

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_noPrune_bf16Q \
  QUANTIZE=bf16 \
  PRUNE_METHOD=none

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_noPrune_bf16Q \
  QUANTIZE=bf16 \
  PRUNE_METHOD=none

#### Prune
make seval CHECKPOINT=head_only.pt \
  WANDB_RUN_NAME=eval_head_only_attn-mlp-tsvd_0.89-0.975-0.975_bf16Q \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
  QUANTIZE=bf16

make seval CHECKPOINT=lora.pt \
  WANDB_RUN_NAME=eval_lora_attn-mlp-tsvd_0.89-0.975-0.975_bf16Q \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
  QUANTIZE=bf16

make seval CHECKPOINT=fullft.pt \
  WANDB_RUN_NAME=eval_fullft_attn-mlp-tsvd_0.89-0.975-0.975_bf16Q \
  PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
  PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
  QUANTIZE=bf16

###

# # WANDB_RUN_NAME=eval_lora_attn-mlp-tsvd_0.89-0.975-0.975 \
# make seval CHECKPOINT=head_only.pt \
#   WANDB_RUN_NAME=test_qint8_head_only \
#   PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
#   PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
#   QUANTIZE=int8
#
# # WANDB_RUN_NAME=eval_head_only_attn-mlp-tsvd_0.89-0.975-0.975 \
# make seval CHECKPOINT=head_only.pt \
#   WANDB_RUN_NAME=test_qbf16_head_only \
#   PRUNE_TARGETS=all PRUNE_METHOD=attention_heads,mlp_neurons,truncated_svd \
#   PRUNE_AMOUNT=attention_heads=0.89,mlp_neurons=0.975,truncated_svd=0.975 \
#   QUANTIZE=bf16
