"""
evaluate.py
-----------
Évaluation complète : métriques, courbes ROC/PR, matrice de confusion, SHAP

Usage:
    python src/evaluation/evaluate.py 
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.feature_pipeline import FeaturePipeline

logger = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_inputs, y_true, model_name: str):
    """Calcule et affiche toutes les métriques."""
    y_prob = model.predict(X_inputs, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"  📊 Résultats : {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Légitimes", "Phishing"]))

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}")

    return y_prob, y_pred, auc_roc, auc_pr


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Légitimes", "Phishing"],
        yticklabels=["Légitimes", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"💾 confusion_{model_name}.png")


def plot_roc_curves(results: dict):
    """Affiche les courbes ROC de tous les modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

    for (name, y_true, y_prob), color in zip(results.values(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title("Courbes ROC — Comparaison modèles")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()
    logger.info("💾 roc_curves.png")


def plot_training_history(history, model_name: str):
    """Courbes loss et accuracy pendant l'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle(f"Courbes d'entraînement — {model_name}")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"history_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"💾 history_{model_name}.png")


def plot_feature_importance(model, feature_names: list, X_feat: np.ndarray):
    """Importance des features via perturbation."""
    baseline = model.predict(X_feat[:500], verbose=0).mean()
    importances = []

    for i in range(X_feat.shape[1]):
        X_perturbed = X_feat[:500].copy()
        X_perturbed[:, i] = 0
        score = model.predict(X_perturbed, verbose=0).mean()
        importances.append(abs(score - baseline))

    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    ax.barh(
        [feature_names[i] for i in indices],
        [importances[i] for i in indices],
        color="#e63946",
    )
    ax.set_xlabel("Importance relative")
    ax.set_title("Importance des features URL")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
    logger.info("💾 feature_importance.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    # Pipeline
    pipeline = FeaturePipeline.load(MODELS_DIR / "pipeline")
    X_seq_test, X_feat_test, y_test = pipeline.transform(test_df)

    results = {}

    # Hybrid
    hybrid_path = MODELS_DIR / "hybrid_best.keras"
    if hybrid_path.exists():
        model = tf.keras.models.load_model(str(hybrid_path))
        y_prob, y_pred, auc_roc, _ = evaluate_model(
            model, [X_seq_test, X_feat_test], y_test, "Hybrid"
        )
        plot_confusion_matrix(y_test, y_pred, "hybrid")
        results["hybrid"] = ("Hybrid", y_test, y_prob)

    # LSTM only
    lstm_path = MODELS_DIR / "lstm_only_best.keras"
    if lstm_path.exists():
        model_lstm = tf.keras.models.load_model(str(lstm_path))
        y_prob_lstm, y_pred_lstm, _, _ = evaluate_model(
            model_lstm, X_seq_test, y_test, "LSTM Only"
        )
        results["lstm"] = ("LSTM Only", y_test, y_prob_lstm)

    # Features only
    feat_path = MODELS_DIR / "features_only_best.keras"
    if feat_path.exists():
        model_feat = tf.keras.models.load_model(str(feat_path))
        y_prob_feat, y_pred_feat, _, _ = evaluate_model(
            model_feat, X_feat_test, y_test, "Features Only"
        )
        results["features"] = ("Features Only", y_test, y_prob_feat)

    if results:
        plot_roc_curves(results)
        logger.info(f"\n✅ Figures sauvegardées dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()
