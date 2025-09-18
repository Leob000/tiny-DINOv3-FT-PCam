# local: `make baseline METHOD=head_only` (default METHOD=head_only, other: lora, fullft)
# Slurm GPU cluster: `make sbaseline METHOD=head_only` (default METHOD=head_only, other: lora, fullft)
PY=python
DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m # Backbone model

# Download PCam, run once before other commands
get-data:
	$(PY) -m scripts.download_pcam --out $(DATA_DIR)

METHOD?=head_only
SELECT_METRIC?=auroc
WANDB?=--wandb
NUM_WORKERS?=4
EPOCHS?=8
RESOLUTION?=224
WARMUP_STEPS?=200
GRAD_CLIP?=1.0
LABEL_SMOOTHING?=0.05
BATCH_SIZE?=256
VAL_BATCH_SIZE?=512
LR?=1.0e-3
WEIGHT_DECAY?=1.0e-4
LORA_R?=8
LORA_ALPHA?=16
LORA_DROPOUT?=0.05
LORA_TARGETS?=q_proj,k_proj,v_proj,o_proj
LR_HEAD?=1.0e-3
LR_LORA?=5.0e-4
LR_NORMS_BIAS?=5.0e-4
VAL_EVAL_FRAC?=0.5
# Possible VAL_FLAGS: val_mid_epoch, val_epoch_end, val_heavy_end, val_heavy_mid
VAL_FLAGS=--val_mid_epoch --val_epoch_end --val_heavy_end
VAL_FLAGS_HUGE=--val_mid_epoch --val_epoch_end --val_heavy_end --val_heavy_mid
VAL_FLAGS_NO_MID=--val_epoch_end --val_heavy_end
TRAIN_NORMS_BIAS?=none # [none, norms, bias, both] train the LayerNorms params for head_only/LoRA methods

CHECKPOINT?=lora.pt

eval:
	$(PY) -m src.train.eval_checkpoint \
		--checkpoint $(CHECKPOINT) \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution $(RESOLUTION) \
		--val_batch_size $(VAL_BATCH_SIZE) \
		--num_workers $(NUM_WORKERS) \
		--tta_eval

baseline:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
    --select_metric $(SELECT_METRIC) \
		--resolution $(RESOLUTION) \
		--batch_size $(BATCH_SIZE) \
		--val_batch_size $(VAL_BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--weight_decay $(WEIGHT_DECAY) \
		--num_workers $(NUM_WORKERS) \
		--method $(METHOD) \
		--train_log_every_steps 4 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS_NO_MID) \
		--lora_r $(LORA_R) --lora_alpha $(LORA_ALPHA) --lora_dropout $(LORA_DROPOUT) \
		--lora_targets $(LORA_TARGETS) \
		--lora_include_mlp \
		--lr_head $(LR_HEAD) --lr_lora $(LR_LORA) \
		--train_norms_bias $(TRAIN_NORMS_BIAS) --lr_norms_bias $(LR_NORMS_BIAS) \
		--warmup_steps $(WARMUP_STEPS) --grad_clip $(GRAD_CLIP) \
		--label_smoothing $(LABEL_SMOOTHING) \
		--aug_histology --tta_eval \
		--save_best

debug:
	$(PY) -m src.train.train_linear \
		--method $(METHOD) \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution 96 \
		--batch_size 64 \
		--val_batch_size 64 \
		--epochs 1 \
		--lr 1e-3 \
		--weight_decay 1e-4 \
		--num_workers $(NUM_WORKERS) \
		--max_train_batches 1 \
		--max_eval_batches 1 \
		--skip_bench \
		--warmup_steps 0 --grad_clip $(GRAD_CLIP) \
		--label_smoothing $(LABEL_SMOOTHING)

SLURM_PARTITION ?= tau
SLURM_TIME ?= 24:00:00
SLURM_GPUS ?= 1
SLURM_CPUS ?= 11
SLURM_JOB_NAME ?= tiny-dino-pcam

SBATCH = sbatch \
  --job-name=$(SLURM_JOB_NAME) \
  --partition=$(SLURM_PARTITION) \
  --gres=gpu:$(SLURM_GPUS) \
  --cpus-per-task=$(SLURM_CPUS) \
  --time=$(SLURM_TIME) \
  --output=slurm/%x_%j.out \
  --error=slurm/%x_%j.out \
	--nodes=1 \
	--ntasks-per-node=1

COMMON = $(PY) -m src.train.train_linear \
  --data_dir $(DATA_DIR) \
  --model_id $(MODEL_ID) \
  --select_metric $(SELECT_METRIC) \
	--method $(METHOD) \
  --resolution $(RESOLUTION) \
  --num_workers $(NUM_WORKERS) \
  --batch_size $(BATCH_SIZE) --val_batch_size $(VAL_BATCH_SIZE) \
  --epochs $(EPOCHS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) \
  $(WANDB) --wandb_project dinov3-pcam-compress \
  --train_log_every_steps 4 \
  --val_eval_frac $(VAL_EVAL_FRAC) \
  $(VAL_FLAGS_NO_MID) \
  --lora_r $(LORA_R) --lora_alpha $(LORA_ALPHA) --lora_dropout $(LORA_DROPOUT) \
  --lora_targets $(LORA_TARGETS) \
  --lora_include_mlp \
  --lr_head $(LR_HEAD) --lr_lora $(LR_LORA) \
	--train_norms_bias $(TRAIN_NORMS_BIAS) --lr_norms_bias $(LR_NORMS_BIAS) \
	--warmup_steps $(WARMUP_STEPS) --grad_clip $(GRAD_CLIP) \
	--label_smoothing $(LABEL_SMOOTHING) \
	--aug_histology --tta_eval \
  --save_best

COMMON2 = $(PY) -m src.train.eval_checkpoint \
	--checkpoint $(CHECKPOINT) \
  $(WANDB) --wandb_project dinov3-pcam-compress \
	--data_dir $(DATA_DIR) \
	--model_id $(MODEL_ID) \
	--resolution $(RESOLUTION) \
	--val_batch_size $(VAL_BATCH_SIZE) \
	--num_workers $(NUM_WORKERS) \
	--tta_eval

.PHONY: common common2

# Where the project & venv live on the cluster
CLUSTER_DIR ?= $(HOME)/Tiny-DINOv3-PCam
VENV ?= .venv

define WRAP_CMD
bash -lc "source /home/tau/lburgund/.bashrc; \
  cd $(CLUSTER_DIR); \
  source $(VENV)/bin/activate; \
  which python; python --version; \
  echo Running on partition: $$SLURM_JOB_PARTITION; \
  echo Running: $(1); \
  $(1)"
endef

sbaseline:
	mkdir -p slurm
	$(SBATCH) --wrap='$(call WRAP_CMD,$(COMMON))'

seval:
	mkdir -p slurm
	$(SBATCH) --wrap='$(call WRAP_CMD,$(COMMON2))'
