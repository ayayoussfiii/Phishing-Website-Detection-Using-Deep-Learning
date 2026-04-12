"""
train.py
--------
Script d'entraînement principal.

Usage:
    python src/training/train.py --model hybrid
    python src/training/train.py --model lstm_only
    python src/training/train.py --model features_only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import mlflow
import mlflow.keras

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seeds
from src.features.feature_pipeline import FeaturePipeline
from src.models.hybrid_model import (
    build_hybrid_model,
    build_lstm_only_model,
    build_features_only_model,
)

logger = get_logger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_DIR = PROJECT_ROOT / "models" / "pipeline"


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = 5):
    checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(PROJECT_ROOT / "reports" / "tensorboard" / model_name),
        ),
    ]


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────

def load_data():
    logger.info("📂 Chargement des données...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────

def train_hybrid(cfg):
    train_df, val_df, test_df = load_data()

    # Pipeline
    pipeline = FeaturePipeline(max_len=cfg["features"]["max_seq_len"])
    X_seq_train, X_feat_train, y_train = pipeline.fit_transform(train_df)
    X_seq_val, X_feat_val, y_val = pipeline.transform(val_df)
    X_seq_test, X_feat_test, y_test = pipeline.transform(test_df)

    pipeline.save(PIPELINE_DIR)

    # Modèle
    model = build_hybrid_model(
        vocab_size=pipeline.tokenizer.vocab_size,
        max_len=cfg["features"]["max_seq_len"],
        n_features=cfg["features"]["url_features"],
        embedding_dim=cfg["models"]["lstm"]["embedding_dim"],
        lstm_units=cfg["models"]["lstm"]["lstm_units"],
        hidden_units=cfg["models"]["features_branch"]["hidden_units"],
        fusion_units=cfg["models"]["hybrid"]["fusion_units"],
        dropout=cfg["models"]["hybrid"]["dropout"],
        learning_rate=cfg["training"]["learning_rate"],
    )

    # MLflow
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    with mlflow.start_run(run_name="hybrid"):
        mlflow.log_params({
            "model": "hybrid",
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "lstm_units": cfg["models"]["lstm"]["lstm_units"],
        })

        history = model.fit(
            [X_seq_train, X_feat_train], y_train,
            validation_data=([X_seq_val, X_feat_val], y_val),
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            callbacks=get_callbacks("hybrid", cfg["training"]["early_stopping_patience"]),
            verbose=1,
        )

        # Évaluation finale
        results = model.evaluate([X_seq_test, X_feat_test], y_test, verbose=0)
        metrics = dict(zip(model.metrics_names, results))
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        logger.info("📊 Résultats test :")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    return model, history


def train_lstm_only(cfg):
    train_df, val_df, test_df = load_data()
    pipeline = FeaturePipeline(max_len=cfg["features"]["max_seq_len"])
    X_seq_train, _, y_train = pipeline.fit_transform(train_df)
    X_seq_val, _, y_val = pipeline.transform(val_df)
    X_seq_test, _, y_test = pipeline.transform(test_df)

    model = build_lstm_only_model(
        vocab_size=pipeline.tokenizer.vocab_size,
        max_len=cfg["features"]["max_seq_len"],
        lstm_units=cfg["models"]["lstm"]["lstm_units"],
        dropout=cfg["models"]["lstm"]["dropout"],
        learning_rate=cfg["training"]["learning_rate"],
    )

    with mlflow.start_run(run_name="lstm_only"):
        history = model.fit(
            X_seq_train, y_train,
            validation_data=(X_seq_val, y_val),
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            callbacks=get_callbacks("lstm_only", cfg["training"]["early_stopping_patience"]),
            verbose=1,
        )

    return model, history


def train_features_only(cfg):
    train_df, val_df, test_df = load_data()
    pipeline = FeaturePipeline(max_len=cfg["features"]["max_seq_len"])
    _, X_feat_train, y_train = pipeline.fit_transform(train_df)
    _, X_feat_val, y_val = pipeline.transform(val_df)
    _, X_feat_test, y_test = pipeline.transform(test_df)

    model = build_features_only_model(
        n_features=cfg["features"]["url_features"],
        dropout=cfg["models"]["features_branch"]["dropout"],
        learning_rate=cfg["training"]["learning_rate"],
    )

    with mlflow.start_run(run_name="features_only"):
        history = model.fit(
            X_feat_train, y_train,
            validation_data=(X_feat_val, y_val),
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            callbacks=get_callbacks("features_only", cfg["training"]["early_stopping_patience"]),
            verbose=1,
        )

    return model, history


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["hybrid", "lstm_only", "features_only", "all"],
        default="hybrid",
        help="Modèle à entraîner",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seeds(cfg["seed"])

    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])

    if args.model == "hybrid" or args.model == "all":
        logger.info("🚀 Entraînement modèle HYBRIDE...")
        train_hybrid(cfg)

    if args.model == "lstm_only" or args.model == "all":
        logger.info("🚀 Entraînement LSTM only...")
        train_lstm_only(cfg)

    if args.model == "features_only" or args.model == "all":
        logger.info("🚀 Entraînement Features only...")
        train_features_only(cfg)

    logger.info("🎉 Entraînement terminé !")


if __name__ == "__main__":
    main()
