# local: `make model METHOD=head_only` (default METHOD=head_only, other: lora, fullft)
# GPU cluster: `make smodel METHOD=head_only` (default METHOD=head_only, other: lora, fullft)
PY=python
DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m

# Download PCam, run once before other commands
get-data:
	$(PY) -m scripts.download_pcam --out $(DATA_DIR)

METHOD?=head_only
WANDB?=--wandb
NUM_WORKERS?=4
EPOCHS?=16
RESOLUTION?=96
VAL_EVAL_FRAC?=0.5
# Possible VAL_FLAGS: val_mid_epoch, val_epoch_end, val_heavy_end, val_heavy_mid
VAL_FLAGS=--val_mid_epoch --val_epoch_end --val_heavy_end
VAL_FLAGS_HUGE=--val_mid_epoch --val_epoch_end --val_heavy_end --val_heavy_mid
TRAIN_NORMS_BIAS=none # [none, norms, bias, both] train the LayerNorms params for head_only/LoRA methods

baseline:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution $(RESOLUTION) \
		--batch_size 256 \
		--val_batch_size 512 \
		--epochs $(EPOCHS) \
		--lr 1e-3 \
		--weight_decay 1e-4 \
		--num_workers $(NUM_WORKERS) \
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--skip_bench \
		--method $(METHOD) \
		--train_log_every_steps 4 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS) \
		--lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
		--lora_targets q_proj,k_proj,v_proj,o_proj \
		--lora_include_mlp \
		--lr_head 1e-3 --lr_lora 1e-3 \
		--train_norms_bias $(TRAIN_NORMS_BIAS) \
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
		--save_last

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
	--method $(METHOD) \
  --resolution $(RESOLUTION) \
  --num_workers $(NUM_WORKERS) \
  --batch_size 256 --val_batch_size 512 \
  --epochs $(EPOCHS) --lr 1e-3 --weight_decay 1e-4 \
  $(WANDB) --wandb_project dinov3-pcam-compress \
  --skip_bench \
  --train_log_every_steps 4 \
  --val_eval_frac $(VAL_EVAL_FRAC) \
  $(VAL_FLAGS_HUGE) \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj \
  --lora_include_mlp \
  --lr_head 1e-3 --lr_lora 1e-3 \
  --train_norms_bias $(TRAIN_NORMS_BIAS) \
  --save_best

.PHONY: common

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

smodel:
	mkdir -p slurm
	$(SBATCH) --wrap='$(call WRAP_CMD,$(COMMON))'
