PY=python

DATA_DIR=src/data/pcam
MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m

NUM_WORKERS?=5
EPOCHS?=2

baseline:
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
		--results_csv results.csv

baseline_96:
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
		--results_csv results.csv
