PY=python

DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m

WANDB?=--wandb
EVAL_FRAC?=0.05
MID_EVAL_BATCHES?=32
NUM_WORKERS?=5
EPOCHS?=1

baseline_224:
	$(PY) -m src.train.train_linear \
		--data_dir $(DATA_DIR) \
		--model_id $(MODEL_ID) \
		--resolution 224 \
		--batch_size 128 \
		--val_batch_size 256 \
		--epochs $(EPOCHS) \
		--lr 1e-3 \
		--weight_decay 1e-4 \
		--num_workers $(NUM_WORKERS) \
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--skip_bench \
		--method linear_probe \
    --eval_frac $(EVAL_FRAC) \
    --mid_eval_batches $(MID_EVAL_BATCHES) \
		--results_csv results.csv

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
    --eval_frac $(EVAL_FRAC) \
    --mid_eval_batches $(MID_EVAL_BATCHES) \
		--results_csv results.csv

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
		$(WANDB) --wandb_project dinov3-pcam-compress \
		--skip_bench \
		--method debug \
		--results_csv results.csv
