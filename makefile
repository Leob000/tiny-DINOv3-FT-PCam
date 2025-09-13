# local: `make baseline`, `make lora`, ...
# GPU cluster: `make sbaseline`, `make slora`, ...
PY=python
DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m

# Download PCam, run once before other commands
get-data:
	$(PY) -m scripts.download_pcam --out $(DATA_DIR)

WANDB?=--wandb
NUM_WORKERS?=4
EPOCHS?=2
VAL_EVAL_FRAC?=0.5
# Possible VAL_FLAGS: val_mid_epoch, val_epoch_end, val_heavy_end, val_heavy_mid
VAL_FLAGS=--val_mid_epoch --val_epoch_end --val_heavy_end
VAL_FLAGS_HUGE=--val_mid_epoch --val_epoch_end --val_heavy_end --val_heavy_mid
# Also possible to have resolution 224

baseline:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution 96 \
		--batch_size 256 \
		--val_batch_size 512 \
		--epochs $(EPOCHS) \
		--lr 1e-3 \
		--weight_decay 1e-4 \
		--num_workers $(NUM_WORKERS) \
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--skip_bench \
		--method head_only \
		--train_log_every_steps 4 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS) \
		--save_best

serious_baseline:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution 96 \
		--batch_size 256 \
		--val_batch_size 512 \
		--epochs 10 \
		--lr 1e-3 \
		--weight_decay 1e-4 \
		--num_workers $(NUM_WORKERS) \
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--method head_only \
		--train_log_every_steps 2 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS_HUGE) \
		--save_best

lora:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution 96 \
		--batch_size 256 \
		--val_batch_size 512 \
		--epochs $(EPOCHS) \
		--num_workers $(NUM_WORKERS) \
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--skip_bench \
		--method lora\
		--lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
		--lora_targets q_proj,k_proj,v_proj,o_proj \
		--lora_include_mlp \
		--lr_head 1e-3 --lr_lora 1e-3 \
		--train_log_every_steps 2 \
		$(VAL_FLAGS) \
		--save_best


debug:
	$(PY) -m src.train.train_linear \
		--method head_only \
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

# Common part of your python command
COMMON = -m src.train.train_linear \
  --data_dir $(DATA_DIR) \
  --model_id $(MODEL_ID) \
  --resolution 96 \
  --num_workers $(NUM_WORKERS) \
  --batch_size 256 --val_batch_size 512 \
  --epochs $(EPOCHS) --lr 1e-3 --weight_decay 1e-4 \
  $(WANDB) --wandb_project dinov3-pcam-compress \
  --skip_bench \
  --train_log_every_steps 4 \
  --val_eval_frac $(VAL_EVAL_FRAC) \
  $(VAL_FLAGS_HUGE) \
  --save_best

BASELINE_CMD = $(PY) $(COMMON) \
  --method head_only

LORA_CMD = $(PY) $(COMMON) \
  --method lora \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_targets q_proj,k_proj,v_proj,o_proj \
  --lora_include_mlp \
  --lr_head 1e-3 --lr_lora 1e-3

.PHONY: sbaseline slora

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
	$(SBATCH) --wrap='$(call WRAP_CMD,$(BASELINE_CMD))'

slora:
	mkdir -p slurm
	$(SBATCH) --wrap='$(call WRAP_CMD,$(LORA_CMD))'
