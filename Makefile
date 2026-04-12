.PHONY: data train evaluate app test all clean

# ─────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────

all: data train evaluate app

data:
	python src/data/make_dataset.py

train:
	python src/training/train.py --model hybrid

train-all:
	python src/training/train.py --model all

evaluate:
	python src/evaluation/evaluate.py

app:
	python app/app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*.parquet
	rm -rf models/*.keras
	rm -rf models/pipeline
	rm -rf reports/figures/*.png
	rm -rf mlruns/

# ─────────────────────────────────────────────────────────────
# Entraînement individuel
# ─────────────────────────────────────────────────────────────

train-lstm:
	python src/training/train.py --model lstm_only

train-features:
	python src/training/train.py --model features_only

# ─────────────────────────────────────────────────────────────
# MLflow UI
# ─────────────────────────────────────────────────────────────

mlflow:
	mlflow ui --port 5000
