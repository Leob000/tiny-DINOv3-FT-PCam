PY=python
DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m

WANDB?=--wandb
NUM_WORKERS?=4
EPOCHS?=1
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
		--method linear_probe \
		--train_log_every_steps 4 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS) \

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
		--method linear_probe \
		--train_log_every_steps 2 \
		--val_eval_frac $(VAL_EVAL_FRAC) \
		$(VAL_FLAGS_HUGE) \

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


debug:
	$(PY) -m src.train.train_linear \
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
		--method linear_probe \
