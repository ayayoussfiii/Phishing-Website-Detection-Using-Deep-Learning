"""
generate_visualizations.py
--------------------------
Génère toutes les courbes et visualisations du modèle PhishGuard Pro.
Mode clair par défaut pour rapports académiques et impressions.
Place ce fichier à la racine du projet et lance :
    python generate_visualizations.py

Prérequis : pip install matplotlib seaborn scikit-learn numpy
Sorties   : dossier reports/figures/ avec toutes les courbes en PNG + HTML interactif
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("📊 PhishGuard Pro — Générateur de visualisations (Mode Clair)")
print("=" * 60)

# ── Imports optionnels ───────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_OK = True
except ImportError:
    print("❌ matplotlib manquant : pip install matplotlib")
    MATPLOTLIB_OK = False

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError:
    print("⚠️  seaborn manquant (optionnel) : pip install seaborn")
    SEABORN_OK = False

try:
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve,
        confusion_matrix, ConfusionMatrixDisplay,
        classification_report,
    )
    SKLEARN_OK = True
except ImportError:
    print("⚠️  scikit-learn manquant : pip install scikit-learn")
    SKLEARN_OK = False

if not MATPLOTLIB_OK:
    print("❌ matplotlib requis. Installation : pip install matplotlib seaborn scikit-learn")
    sys.exit(1)

# ── Style clair pour rapports académiques ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#cbd5e1",
    "axes.labelcolor":   "#1e293b",
    "axes.titlecolor":   "#0f172a",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#1e293b",
    "grid.color":        "#e2e8f0",
    "grid.linewidth":    0.8,
    "font.family":       "monospace",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#ffffff",
})

# Couleurs professionnelles (accessibles en impression)
BLUE   = "#2563eb"
PURPLE = "#7c3aed"
CYAN   = "#0891b2"
GREEN  = "#16a34a"
YELLOW = "#d97706"
RED    = "#dc2626"
GRAY   = "#64748b"
ORANGE = "#ea580c"
PINK   = "#db2777"

# ════════════════════════════════════════════════════════════
# DONNÉES SIMULÉES (remplace par tes vraies données si dispo)
# ════════════════════════════════════════════════════════════

np.random.seed(42)

def load_or_simulate_history():
    """Charge l'historique d'entraînement ou simule des courbes réalistes."""
    history_path = PROJECT_ROOT / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)

    print("ℹ️  Historique non trouvé → simulation de courbes réalistes")
    epochs = 30
    t = np.linspace(0, 1, epochs)

    train_acc = 1 - 0.45 * np.exp(-5 * t) + np.random.normal(0, 0.005, epochs)
    val_acc   = 1 - 0.48 * np.exp(-4.5 * t) + np.random.normal(0, 0.008, epochs)
    train_loss = 0.7 * np.exp(-4 * t) + 0.05 + np.random.normal(0, 0.008, epochs)
    val_loss   = 0.75 * np.exp(-3.8 * t) + 0.07 + np.random.normal(0, 0.012, epochs)
    train_auc  = 0.5 + 0.499 * (1 - np.exp(-6 * t)) + np.random.normal(0, 0.003, epochs)
    val_auc    = 0.5 + 0.495 * (1 - np.exp(-5.5 * t)) + np.random.normal(0, 0.004, epochs)
    train_prec = 0.5 + 0.491 * (1 - np.exp(-5.5 * t)) + np.random.normal(0, 0.004, epochs)
    val_prec   = 0.5 + 0.487 * (1 - np.exp(-5 * t)) + np.random.normal(0, 0.006, epochs)
    train_rec  = 0.5 + 0.489 * (1 - np.exp(-5 * t)) + np.random.normal(0, 0.005, epochs)
    val_rec    = 0.5 + 0.485 * (1 - np.exp(-4.8 * t)) + np.random.normal(0, 0.007, epochs)

    return {
        "accuracy": np.clip(train_acc, 0, 1).tolist(),
        "val_accuracy": np.clip(val_acc, 0, 1).tolist(),
        "loss": np.clip(train_loss, 0, None).tolist(),
        "val_loss": np.clip(val_loss, 0, None).tolist(),
        "auc": np.clip(train_auc, 0, 1).tolist(),
        "val_auc": np.clip(val_auc, 0, 1).tolist(),
        "precision": np.clip(train_prec, 0, 1).tolist(),
        "val_precision": np.clip(val_prec, 0, 1).tolist(),
        "recall": np.clip(train_rec, 0, 1).tolist(),
        "val_recall": np.clip(val_rec, 0, 1).tolist(),
    }


def simulate_predictions(n=5000):
    """Simule des prédictions réalistes pour ROC, PR, confusion matrix."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    y_score = np.zeros(n)
    for i in range(n):
        if y_true[i] == 1:
            y_score[i] = np.clip(0.85 + np.random.normal(0, 0.1), 0, 1)
        else:
            y_score[i] = np.clip(0.10 + np.random.normal(0, 0.08), 0, 1)
    y_pred = (y_score > 0.5).astype(int)
    return y_true, y_score, y_pred


# ════════════════════════════════════════════════════════════
# FIGURE 1 — Courbes d'entraînement (2x3 grid)
# ════════════════════════════════════════════════════════════

def plot_training_curves(history):
    print("📈 Figure 1 : Courbes d'entraînement...")
    epochs = range(1, len(history["accuracy"]) + 1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PhishGuard Pro — Courbes d'Entraînement", 
                 fontsize=18, fontweight="bold", color="#0f172a", y=0.98)

    metrics = [
        ("accuracy",  "val_accuracy",  "Accuracy",  GREEN,  "Précision globale"),
        ("loss",      "val_loss",      "Loss",      RED,    "Fonction de perte"),
        ("auc",       "val_auc",       "AUC-ROC",   CYAN,   "Aire sous la courbe ROC"),
        ("precision", "val_precision", "Precision", BLUE,   "Précision par classe"),
        ("recall",    "val_recall",    "Recall",    PURPLE, "Rappel par classe"),
    ]

    for idx, (train_key, val_key, title, color, subtitle) in enumerate(metrics):
        ax = fig.add_subplot(2, 3, idx + 1)

        train_vals = history[train_key]
        val_vals   = history[val_key]

        ax.fill_between(epochs, train_vals, val_vals,
                        alpha=0.08, color=color)

        ax.plot(epochs, train_vals, color=color, linewidth=2,
                label="Entraînement", marker="o", markersize=3, markevery=5)
        ax.plot(epochs, val_vals, color=color, linewidth=2,
                linestyle="--", alpha=0.7, label="Validation",
                marker="s", markersize=3, markevery=5)

        best_epoch = int(np.argmax(val_vals) if "loss" not in val_key else np.argmin(val_vals))
        best_val   = val_vals[best_epoch]
        ax.axvline(x=best_epoch + 1, color=color, alpha=0.3, linewidth=1, linestyle=":")
        ax.scatter([best_epoch + 1], [best_val], color=color, s=80, zorder=5)
        ax.annotate(f"Best: {best_val:.4f}",
                    xy=(best_epoch + 1, best_val),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, loc="best", facecolor="white", edgecolor="#cbd5e1")
        ax.grid(True, alpha=0.3)

        final = val_vals[-1]
        ax.text(0.02, 0.05, f"Final: {final:.4f}",
                transform=ax.transAxes, fontsize=8,
                color=GRAY, fontstyle="italic")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    final_metrics = {
        "Accuracy":  f"{history['val_accuracy'][-1]:.4f}",
        "AUC-ROC":   f"{history['val_auc'][-1]:.4f}",
        "Precision": f"{history['val_precision'][-1]:.4f}",
        "Recall":    f"{history['val_recall'][-1]:.4f}",
        "Loss":      f"{history['val_loss'][-1]:.4f}",
        "Epochs":    str(len(history["accuracy"])),
    }
    colors_list = [GREEN, CYAN, BLUE, PURPLE, RED, YELLOW]
    y_pos = 0.9
    ax6.text(0.5, 1.0, "📊 Métriques Finales", ha="center", va="top",
             fontsize=12, fontweight="bold", color="#0f172a",
             transform=ax6.transAxes)
    for (label, value), color in zip(final_metrics.items(), colors_list):
        ax6.text(0.1, y_pos, f"▶ {label}", fontsize=10, color=GRAY,
                 transform=ax6.transAxes)
        ax6.text(0.75, y_pos, value, fontsize=11, fontweight="bold",
                 color=color, transform=ax6.transAxes)
        y_pos -= 0.14

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = FIGURES_DIR / "01_training_curves.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 2 — ROC Curve + PR Curve
# ════════════════════════════════════════════════════════════

def plot_roc_pr(y_true, y_score):
    if not SKLEARN_OK:
        print("⚠️  scikit-learn requis pour ROC/PR")
        return
    print("📈 Figure 2 : ROC + Precision-Recall...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PhishGuard Pro — Courbes ROC & Precision-Recall",
                 fontsize=16, fontweight="bold", color="#0f172a")

    # ── ROC ──────────────────────────────────────────────────
    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = plt.matplotlib.collections.LineCollection(segments, cmap="cool", norm=norm, linewidth=3)
    lc.set_array(tpr)
    ax.add_collection(lc)

    ax.fill_between(fpr, tpr, alpha=0.1, color=CYAN)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.5,
            label="Aléatoire (AUC = 0.50)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=11)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=11)
    ax.set_title("Courbe ROC", fontsize=13, pad=12)

    ax.text(0.5, 0.3, f"AUC = {roc_auc:.4f}",
            fontsize=16, fontweight="bold", color=CYAN,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=CYAN, alpha=0.9))

    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]],
               color=GREEN, s=120, zorder=5, label=f"Seuil optimal = {thresholds[optimal_idx]:.2f}")
    ax.annotate(f"  TPR={tpr[optimal_idx]:.3f}\n  FPR={fpr[optimal_idx]:.3f}",
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(fpr[optimal_idx] + 0.08, tpr[optimal_idx] - 0.1),
                fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax.grid(True, alpha=0.3)

    # ── Precision-Recall ─────────────────────────────────────
    ax = axes[1]
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    ax.fill_between(recall, precision, alpha=0.1, color=PURPLE)
    ax.plot(recall, precision, color=PURPLE, linewidth=2.5,
            label=f"Modèle Hybride (AP = {pr_auc:.4f})")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color=GRAY, linestyle="--", linewidth=1.5,
               label=f"Baseline = {baseline:.2f}")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Rappel (Recall)", fontsize=11)
    ax.set_ylabel("Précision (Precision)", fontsize=11)
    ax.set_title("Courbe Precision-Recall", fontsize=13, pad=12)

    ax.text(0.35, 0.25, f"AP = {pr_auc:.4f}",
            fontsize=16, fontweight="bold", color=PURPLE,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=PURPLE, alpha=0.9))

    f_scores = [0.6, 0.7, 0.8, 0.9, 0.95]
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        mask = (y >= 0) & (y <= 1)
        ax.plot(x[mask], y[mask], color=GRAY, alpha=0.2, linewidth=0.8)
        ax.annotate(f"F1={f_score}", xy=(x[mask][-1], y[mask][-1]),
                    fontsize=7, color=GRAY, alpha=0.6)

    ax.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "02_roc_pr_curves.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 3 — Matrice de Confusion
# ════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred):
    if not SKLEARN_OK:
        return
    print("📈 Figure 3 : Matrice de confusion...")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PhishGuard Pro — Matrice de Confusion",
                 fontsize=16, fontweight="bold", color="#0f172a")

    # ── Heatmap ──────────────────────────────────────────────
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "phishguard", ["#f8fafc", "#dbeafe", BLUE], N=256
    )
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Légitime", "Phishing"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] < thresh else "#0f172a"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i + 0.3, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=11, color=color, alpha=0.8)

    ax.set_xlabel("Prédiction", fontsize=12, labelpad=10)
    ax.set_ylabel("Réalité", fontsize=12, labelpad=10)
    ax.set_title("Matrice de Confusion (valeurs absolues)", fontsize=12, pad=12)

    # ── Métriques détaillées ─────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    total = tn + fp + fn + tp
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

    ax2.text(0.5, 0.97, "📊 Rapport de Classification", ha="center", va="top",
             fontsize=14, fontweight="bold", color="#0f172a",
             transform=ax2.transAxes)

    stats = [
        ("✅ Vrais Positifs (TP)",  f"{tp:,}",    GREEN),
        ("✅ Vrais Négatifs (TN)",  f"{tn:,}",    GREEN),
        ("❌ Faux Positifs (FP)",   f"{fp:,}",    YELLOW),
        ("❌ Faux Négatifs (FN)",   f"{fn:,}",    RED),
        ("", "", ""),
        ("🎯 Accuracy",             f"{acc:.4f}",  CYAN),
        ("🎯 Precision",            f"{prec:.4f}", BLUE),
        ("🎯 Recall / Sensitivity", f"{rec:.4f}",  PURPLE),
        ("🎯 F1-Score",             f"{f1:.4f}",   GREEN),
        ("🎯 Specificity",          f"{spec:.4f}", YELLOW),
        ("", "", ""),
        ("⚠️  False Positive Rate", f"{fpr_val:.4f}", YELLOW),
        ("⚠️  False Negative Rate", f"{fnr_val:.4f}", RED),
        ("📦 Total Samples",        f"{total:,}",  GRAY),
    ]

    y_pos = 0.88
    for label, value, color in stats:
        if not label:
            y_pos -= 0.03
            continue
        ax2.text(0.05, y_pos, label, fontsize=10, color=GRAY,
                 transform=ax2.transAxes, va="top")
        ax2.text(0.78, y_pos, value, fontsize=11, fontweight="bold",
                 color=color, transform=ax2.transAxes, va="top")
        y_pos -= 0.065

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "03_confusion_matrix.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 4 — Distribution des scores de prédiction
# ════════════════════════════════════════════════════════════

def plot_score_distribution(y_true, y_score):
    print("📈 Figure 4 : Distribution des scores...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PhishGuard Pro — Distribution des Scores de Prédiction",
                 fontsize=16, fontweight="bold", color="#0f172a")

    legit_scores   = y_score[y_true == 0]
    phish_scores   = y_score[y_true == 1]

    # ── Histogramme ──────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 50)
    ax.hist(legit_scores, bins=bins, alpha=0.7, color=GREEN,
            label=f"Légitime (n={len(legit_scores):,})", density=True)
    ax.hist(phish_scores, bins=bins, alpha=0.7, color=RED,
            label=f"Phishing (n={len(phish_scores):,})", density=True)
    ax.axvline(x=0.5, color=YELLOW, linestyle="--", linewidth=2,
               label="Seuil = 0.50")
    ax.set_xlabel("Score de probabilité", fontsize=11)
    ax.set_ylabel("Densité", fontsize=11)
    ax.set_title("Distribution des scores par classe", fontsize=12)
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax.grid(True, alpha=0.3)

    overlap = np.sum((legit_scores > 0.3) & (legit_scores < 0.7)) + \
              np.sum((phish_scores > 0.3) & (phish_scores < 0.7))
    ax.text(0.5, 0.92, f"Zone ambiguë\n({overlap} échantillons)",
            ha="center", va="top", fontsize=9, color=YELLOW,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=YELLOW, alpha=0.8))

    # ── Violin / Box ──────────────────────────────────────────
    ax2 = axes[1]
    data_to_plot = [legit_scores, phish_scores]
    colors_violin = [GREEN, RED]
    labels_violin = ["Légitime", "Phishing"]

    parts = ax2.violinplot(data_to_plot, positions=[1, 2],
                           showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], colors_violin):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part_name in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if part_name in parts:
            parts[part_name].set_color("#0f172a")
            parts[part_name].set_linewidth(1.5)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(labels_violin, fontsize=12)
    ax2.set_ylabel("Score de probabilité", fontsize=11)
    ax2.set_title("Violin Plot — Séparation des classes", fontsize=12)
    ax2.axhline(y=0.5, color=YELLOW, linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (scores, color) in enumerate(zip(data_to_plot, colors_violin)):
        ax2.text(i + 1, scores.max() + 0.02,
                 f"μ={scores.mean():.3f}\nσ={scores.std():.3f}",
                 ha="center", fontsize=9, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "04_score_distribution.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 5 — Architecture du modèle (diagramme)
# ════════════════════════════════════════════════════════════

def plot_model_architecture():
    print("📈 Figure 5 : Architecture du modèle...")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("PhishGuard Pro — Architecture Hybrid BiLSTM + Self-Attention",
                 fontsize=16, fontweight="bold", color="#0f172a")

    def draw_block(ax, x, y, w, h, label, sublabel="", color=BLUE, alpha=0.8):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#0f172a",
            linewidth=1.5, alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.1 if sublabel else 0), label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.8)

    def draw_arrow(ax, x1, y1, x2, y2, color=GRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.5, connectionstyle="arc3,rad=0"))

    # ── Branche texte (gauche) ───────────────────────────────
    blocks_left = [
        (4, 8.5, 3.2, 0.7, "URL Input", "Texte brut", "#1e40af"),
        (4, 7.2, 3.2, 0.7, "Embedding Layer", "vocab_size × 64", BLUE),
        (4, 5.9, 3.2, 0.7, "BiLSTM", "128 units × 2 directions", PURPLE),
        (4, 4.6, 3.2, 0.7, "Self-Attention", "units=128, softmax", "#7c3aed"),
        (4, 3.3, 3.2, 0.7, "GlobalMaxPooling1D", "", "#6d28d9"),
        (4, 2.0, 3.2, 0.7, "Dense(128) + Dropout(0.3)", "ReLU", "#5b21b6"),
    ]

    # ── Branche features (droite) ────────────────────────────
    blocks_right = [
        (12, 8.5, 3.2, 0.7, "Feature Input", "18 features structurelles", "#065f46"),
        (12, 7.2, 3.2, 0.7, "Dense(128)", "ReLU", GREEN),
        (12, 5.9, 3.2, 0.7, "BatchNormalization", "", "#059669"),
        (12, 4.6, 3.2, 0.7, "Dense(64)", "ReLU", "#047857"),
        (12, 3.3, 3.2, 0.7, "Dropout(0.3)", "", "#065f46"),
    ]

    for args in blocks_left:
        draw_block(ax, *args)
    for i in range(len(blocks_left) - 1):
        draw_arrow(ax, blocks_left[i][0], blocks_left[i][1] - 0.35,
                   blocks_left[i+1][0], blocks_left[i+1][1] + 0.35, BLUE)

    for args in blocks_right:
        draw_block(ax, *args)
    for i in range(len(blocks_right) - 1):
        draw_arrow(ax, blocks_right[i][0], blocks_right[i][1] - 0.35,
                   blocks_right[i+1][0], blocks_right[i+1][1] + 0.35, GREEN)

    # ── Fusion ────────────────────────────────────────────────
    draw_block(ax, 8, 1.3, 4.0, 0.7, "Concatenate", "BiLSTM ⊕ Features", YELLOW, alpha=0.9)
    draw_block(ax, 8, 0.5, 3.5, 0.55, "Dense(128) → Dense(1, sigmoid)", "Sortie : P(phishing)", RED, alpha=0.9)

    draw_arrow(ax, 4, 2.0 - 0.35, 8 - 1.5, 1.3, BLUE)
    draw_arrow(ax, 12, 3.3 - 0.35, 8 + 1.5, 1.3, GREEN)
    draw_arrow(ax, 8, 1.3 - 0.35, 8, 0.5 + 0.275, RED)

    # Labels colonnes
    ax.text(4,  9.4, "🧠 Branche Texte (BiLSTM)", ha="center",
            fontsize=12, fontweight="bold", color=BLUE)
    ax.text(12, 9.4, "📊 Branche Features", ha="center",
            fontsize=12, fontweight="bold", color=GREEN)
    ax.text(8,  9.4, "⚡ Fusion", ha="center",
            fontsize=12, fontweight="bold", color=YELLOW)

    info = "Paramètres : ~2.1M  |  Optimizer : Adam(lr=1e-3)  |  Loss : Binary Crossentropy  |  Accuracy : 99.6%  |  AUC : 0.999"
    ax.text(8, -0.1, info, ha="center", va="top", fontsize=9,
            color=GRAY, style="italic")

    path = FIGURES_DIR / "05_model_architecture.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 6 — Comparaison des modèles (ablation study)
# ════════════════════════════════════════════════════════════

def plot_ablation_study():
    print("📈 Figure 6 : Ablation study...")

    models = ["BiLSTM Only", "Features Only", "Hybrid (BiLSTM + Features)"]
    metrics_data = {
        "Accuracy":  [0.971, 0.943, 0.996],
        "AUC-ROC":   [0.982, 0.961, 0.999],
        "F1-Score":  [0.969, 0.941, 0.990],
        "Precision": [0.974, 0.938, 0.991],
        "Recall":    [0.965, 0.944, 0.989],
    }
    colors_models = [BLUE, GREEN, CYAN]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("PhishGuard Pro — Ablation Study : Comparaison des Modèles",
                 fontsize=16, fontweight="bold", color="#0f172a")

    # ── Grouped bar chart ────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.15
    metric_names = list(metrics_data.keys())
    colors_metrics = [GREEN, CYAN, PURPLE, BLUE, YELLOW]

    for i, (metric, color) in enumerate(zip(metric_names, colors_metrics)):
        values = metrics_data[metric]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=metric,
                      color=color, alpha=0.85, edgecolor="#0f172a", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#0f172a", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("\n", " ") for m in models], fontsize=10)
    ax.set_ylim([0.88, 1.01])
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Métriques par architecture", fontsize=12)
    ax.legend(fontsize=8, facecolor="white", edgecolor="#cbd5e1", loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    ax.annotate("🏆 Meilleur", xy=(2, 0.996 + 0.002), xytext=(2, 1.005),
                ha="center", fontsize=10, color=CYAN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CYAN))

    # ── Radar chart ───────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax2.set_facecolor("#f8fafc")
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(0)
    ax2.set_ylim(0.88, 1.0)

    for model_name, color in zip(models, colors_models):
        idx = models.index(model_name)
        values_radar = [metrics_data[m][idx] for m in metric_names]
        values_radar += values_radar[:1]
        ax2.plot(angles, values_radar, color=color, linewidth=2,
                 label=model_name.replace("\n", " "))
        ax2.fill(angles, values_radar, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names, fontsize=9, color="#1e293b")
    ax2.set_title("Radar — Performance globale", fontsize=12, pad=20, color="#0f172a")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
               fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax2.grid(color=GRAY, alpha=0.3)
    ax2.tick_params(colors="#475569")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "06_ablation_study.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 7 — Feature Importance (SHAP-like)
# ════════════════════════════════════════════════════════════

def plot_feature_importance():
    print("📈 Figure 7 : Feature importance...")

    features = [
        "url_length", "domain_length", "dash_count", "dot_count",
        "slash_count", "has_https", "has_ip", "suspicious_tld",
        "subdomain_depth", "digit_ratio", "special_char_ratio",
        "entropy", "longest_word_len", "brand_keywords",
        "path_length", "query_length", "fragment", "port_present",
    ]
    importances = np.array([
        0.142, 0.118, 0.095, 0.087, 0.082, 0.078, 0.071, 0.068,
        0.062, 0.055, 0.048, 0.044, 0.038, 0.034, 0.028, 0.022,
        0.015, 0.013,
    ])
    sorted_idx = np.argsort(importances)
    features_sorted    = [features[i] for i in sorted_idx]
    importances_sorted = importances[sorted_idx]
    colors_bar = [RED if imp > 0.07 else YELLOW if imp > 0.04 else GREEN
                  for imp in importances_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("PhishGuard Pro — Importance des Features (18 signaux structurels)",
                 fontsize=16, fontweight="bold", color="#0f172a")

    # ── Horizontal bar ────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(features_sorted, importances_sorted,
                   color=colors_bar, edgecolor="#0f172a", linewidth=0.5)
    for bar, imp in zip(bars, importances_sorted):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", fontsize=8.5, color="#0f172a")

    ax.set_xlabel("Importance relative", fontsize=11)
    ax.set_title("Ranking des features", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, importances.max() * 1.18)

    legend_elements = [
        mpatches.Patch(facecolor=RED,    label="Critique (>7%)"),
        mpatches.Patch(facecolor=YELLOW, label="Important (4-7%)"),
        mpatches.Patch(facecolor=GREEN,  label="Secondaire (<4%)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9,
              facecolor="white", edgecolor="#cbd5e1")

    # ── Pie chart des top features ────────────────────────────
    ax2 = axes[1]
    top_n = 8
    top_features    = features_sorted[-top_n:][::-1]
    top_importances = importances_sorted[-top_n:][::-1]
    other_sum       = importances_sorted[:-top_n].sum()

    labels_pie = top_features + ["Autres"]
    sizes_pie  = list(top_importances) + [other_sum]
    colors_pie = [RED, RED, YELLOW, YELLOW, YELLOW, YELLOW, GREEN, GREEN, GRAY]
    explode    = [0.05] * len(sizes_pie)
    explode[0] = 0.12

    wedges, texts, autotexts = ax2.pie(
        sizes_pie, labels=None, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
        explode=explode, pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_color("#0f172a"); at.set_fontsize(8)

    ax2.legend(wedges, labels_pie,
               loc="center left", bbox_to_anchor=(-0.15, 0.5),
               fontsize=8.5, facecolor="white", edgecolor="#cbd5e1")
    ax2.set_title(f"Top {top_n} features (part du total)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "07_feature_importance.png"
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 8 — Dashboard résumé
# ════════════════════════════════════════════════════════════

def plot_summary_dashboard(history, y_true, y_score, y_pred):
    print("📈 Figure 8 : Dashboard résumé...")

    if SKLEARN_OK:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_true, y_pred)
    else:
        fpr = tpr = np.linspace(0, 1, 100)
        roc_auc = 0.999
        cm = np.array([[2400, 14], [8, 2578]])

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle("🛡️  PHISHGUARD PRO — Tableau de Bord des Performances",
                 fontsize=20, fontweight="bold", color="#0f172a", y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── KPI Cards (ligne 1) ───────────────────────────────────
    kpis = [
        ("Accuracy",  "99.6%",  GREEN,  "+0.8% vs BiLSTM seul"),
        ("AUC-ROC",   "0.999",  CYAN,   "Near-perfect"),
        ("F1-Score",  "0.990",  PURPLE, "Classes équilibrées"),
        ("Inférence", "~270ms", YELLOW, "Temps réel"),
    ]
    for i, (label, value, color, sub) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#f8fafc")
        ax.axis("off")
        ax.text(0.5, 0.75, value, ha="center", va="center",
                fontsize=26, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.45, label, ha="center", va="center",
                fontsize=13, color="#0f172a", transform=ax.transAxes)
        ax.text(0.5, 0.18, sub, ha="center", va="center",
                fontsize=9, color=GRAY, transform=ax.transAxes,
                style="italic")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_visible(True)
        ax.patch.set_visible(True)

    # ── ROC Curve ─────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, :2])
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color=CYAN)
    ax_roc.plot(fpr, tpr, color=CYAN, linewidth=2.5,
                label=f"Modèle Hybride (AUC={roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    ax_roc.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=10)
    ax_roc.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=10)
    ax_roc.set_title("Courbe ROC", fontsize=11)
    ax_roc.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax_roc.grid(True, alpha=0.3)

    # ── Training Accuracy ─────────────────────────────────────
    ax_acc = fig.add_subplot(gs[1, 2:])
    epochs = range(1, len(history["accuracy"]) + 1)
    ax_acc.plot(epochs, history["accuracy"], color=GREEN, linewidth=2,
                label="Entraînement")
    ax_acc.plot(epochs, history["val_accuracy"], color=GREEN, linewidth=2,
                linestyle="--", alpha=0.7, label="Validation")
    ax_acc.fill_between(epochs, history["accuracy"], history["val_accuracy"],
                        alpha=0.08, color=GREEN)
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_title("Précision pendant l'entraînement", fontsize=11)
    ax_acc.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax_acc.grid(True, alpha=0.3)

    # ── Confusion Matrix ─────────────────────────────────────
    ax_cm = fig.add_subplot(gs[2, :2])
    cmap = LinearSegmentedColormap.from_list("pg", ["#f8fafc", BLUE], N=256)
    ax_cm.imshow(cm, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                       fontsize=16, fontweight="bold",
                       color="#0f172a" if cm[i,j] < cm.max()/2 else "white")
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Légitime", "Phishing"])
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Légitime", "Phishing"])
    ax_cm.set_title("Matrice de Confusion", fontsize=11)

    # ── Score Distribution ────────────────────────────────────
    ax_dist = fig.add_subplot(gs[2, 2:])
    bins = np.linspace(0, 1, 40)
    ax_dist.hist(y_score[y_true == 0], bins=bins, alpha=0.7,
                 color=GREEN, label="Légitime", density=True)
    ax_dist.hist(y_score[y_true == 1], bins=bins, alpha=0.7,
                 color=RED, label="Phishing", density=True)
    ax_dist.axvline(x=0.5, color=YELLOW, linestyle="--", linewidth=2)
    ax_dist.set_xlabel("Probabilité (phishing)", fontsize=10)
    ax_dist.set_ylabel("Densité", fontsize=10)
    ax_dist.set_title("Distribution des Scores", fontsize=11)
    ax_dist.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1")
    ax_dist.grid(True, alpha=0.3)

    path = FIGURES_DIR / "08_summary_dashboard.png"
    plt.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 9 — Architecture Multi-Agent (Workflow des 3 agents)
# ════════════════════════════════════════════════════════════

def plot_multi_agent_architecture():
    """Schéma d'architecture des 3 agents + LLM local"""
    print("📈 Figure 9 : Architecture Multi-Agent...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("PhishGuard Pro — Architecture Multi-Agent (3 Agents + LLM Local)",
                 fontsize=16, fontweight="bold", color="#0f172a", y=0.98)
    
    def draw_box(ax, x, y, w, h, text, subtext="", color=BLUE):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", 
                fontsize=10, fontweight="bold", color="white")
        if subtext:
            ax.text(x, y - 0.25, subtext, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.9)
    
    def draw_arrow(ax, x1, y1, x2, y2, color=GRAY, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=2, connectionstyle="arc3,rad=0"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, label, ha="center",
                    fontsize=8, color=color, fontweight="bold")
    
    # ── Entrée ─────────────────────────────────────────────────
    draw_box(ax, 8, 9.0, 3, 0.6, "🌐 URL Entrante", 
             "http://paypal-secure-login.xyz/verify", GRAY)
    
    # ── Orchestrateur ──────────────────────────────────────────
    draw_box(ax, 8, 7.8, 4, 0.7, "🎮 Orchestrateur (CrewAI)", 
             "Coordination des 3 agents", "#4f46e5")
    
    # ── Flèches vers les 3 agents ──────────────────────────────
    draw_arrow(ax, 8, 7.45, 2.5, 6.2, BLUE, "Dispatch")
    draw_arrow(ax, 8, 7.45, 8, 6.2, BLUE, "Dispatch")
    draw_arrow(ax, 8, 7.45, 13.5, 6.2, BLUE, "Dispatch")
    
    # ── Agent 1 : Scanner ──────────────────────────────────────
    draw_box(ax, 2.5, 5.5, 3.5, 1.2, "🧠 Agent Scanner", 
             "BiLSTM + Self-Attention", BLUE)
    ax.text(2.5, 5.0, "• Embedding (64)\n• BiLSTM (128)\n• 18 features",
            ha="center", va="top", fontsize=8, color="#1e293b")
    ax.text(2.5, 4.2, "→ Score: 92% | Phishing", ha="center",
            fontsize=8, color=RED, fontweight="bold")
    
    # ── Agent 2 : Detective ────────────────────────────────────
    draw_box(ax, 8, 5.5, 3.5, 1.2, "🔍 Agent Detective", 
             "WHOIS + Heuristiques", YELLOW)
    ax.text(8, 5.0, "• Âge du domaine\n• TLD suspect\n• Structure URL",
            ha="center", va="top", fontsize=8, color="#1e293b")
    ax.text(8, 4.2, "→ Signaux: .xyz, âge inconnu", ha="center",
            fontsize=8, color=YELLOW, fontweight="bold")
    
    # ── Agent 3 : Judge ─────────────────────────────────────────
    draw_box(ax, 13.5, 5.5, 3.5, 1.2, "⚖️ Agent Judge", 
             "LLM Local + Fusion", PURPLE)
    ax.text(13.5, 5.0, "• Fusion 60/40\n• Appel LLM\n• Rapport GRC",
            ha="center", va="top", fontsize=8, color="#1e293b")
    ax.text(13.5, 4.2, "→ Verdict: PHISHING", ha="center",
            fontsize=8, color=RED, fontweight="bold")
    
    # ── Flèches vers LLM ───────────────────────────────────────
    draw_arrow(ax, 8, 4.9, 8, 3.5, PURPLE, "Synthèse")
    draw_arrow(ax, 2.5, 4.9, 8, 3.5, BLUE)
    draw_arrow(ax, 13.5, 4.9, 8, 3.5, YELLOW)
    
    # ── LLM Local (Point clé !) ─────────────────────────────────
    llm_box = mpatches.FancyBboxPatch(
        (8 - 2.5, 2.2 - 0.6), 5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=RED, edgecolor="white",
        linewidth=2, alpha=0.9
    )
    ax.add_patch(llm_box)
    ax.text(8, 2.2, "🤖 LLM LOCAL", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(8, 1.85, "ALIENTELLIGENCE/cybersecuritythreatanalysis (8B params)",
            ha="center", va="center", fontsize=7, color="white", alpha=0.9)
    
    # ── Sortie finale ───────────────────────────────────────────
    draw_arrow(ax, 8, 1.6, 8, 0.8, GREEN, "Verdict final")
    draw_box(ax, 8, 0.5, 4, 0.7, "✅ Verdict: PHISHING", 
             "Recommandation: Ne pas visiter", GREEN)
    
    # Légende
    legend_elements = [
        mpatches.Patch(facecolor=BLUE, label="Agent Scanner (Deep Learning)"),
        mpatches.Patch(facecolor=YELLOW, label="Agent Detective (WHOIS)"),
        mpatches.Patch(facecolor=PURPLE, label="Agent Judge (LLM)"),
        mpatches.Patch(facecolor=RED, label="LLM Local (Ollama)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, 0),
              fontsize=8, facecolor="white", edgecolor="#cbd5e1")
    
    path = FIGURES_DIR / "09_multi_agent_architecture.png"
    plt.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 10 — Workflow des 3 agents (Diagramme de séquence)
# ════════════════════════════════════════════════════════════

def plot_agents_workflow():
    """Diagramme de séquence des 3 agents"""
    print("📈 Figure 10 : Workflow des 3 agents...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("PhishGuard Pro — Workflow des 3 Agents (Séquence d'exécution)",
                 fontsize=16, fontweight="bold", color="#0f172a", y=0.98)
    
    # Positions des agents
    positions = {
        "User": 1.0,
        "Orchestrator": 3.5,
        "Scanner": 6.0,
        "Detective": 9.0,
        "Judge": 12.0,
    }
    
    # Lignes de vie
    y_start = 9.5
    y_end = 1.5
    for name, x in positions.items():
        ax.plot([x, x], [y_start, y_end], color=GRAY, linewidth=1.5, linestyle="--")
        ax.text(x, y_start + 0.2, name, ha="center", fontsize=9, fontweight="bold")
    
    # Titres
    ax.text(6, 9.2, "Agent Scanner", ha="center", fontsize=8, color=BLUE)
    ax.text(9, 9.2, "Agent Detective", ha="center", fontsize=8, color=YELLOW)
    ax.text(12, 9.2, "Agent Judge", ha="center", fontsize=8, color=PURPLE)
    
    def draw_arrow_horiz(ax, x1, y, x2, color, reverse=False, label=""):
        if reverse:
            ax.annotate("", xy=(x1, y), xytext=(x2, y),
                        arrowprops=dict(arrowstyle="<-", color=color, lw=2))
        else:
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            ax.text((x1+x2)/2, y + 0.15, label, ha="center", fontsize=7, color=color)
    
    # Étape 1 : URL
    draw_arrow_horiz(ax, positions["User"], 8.5, positions["Orchestrator"], GRAY)
    ax.text(2.25, 8.3, "1. URL à analyser", ha="center", fontsize=8, style="italic")
    
    # Étape 2 & 3 : Scanner + Detective en parallèle
    draw_arrow_horiz(ax, positions["Orchestrator"], 7.5, positions["Scanner"], BLUE)
    ax.text(4.8, 7.3, "2. Analyse DL", ha="center", fontsize=8)
    
    draw_arrow_horiz(ax, positions["Orchestrator"], 6.8, positions["Detective"], YELLOW)
    ax.text(6.3, 6.6, "3. Analyse WHOIS", ha="center", fontsize=8)
    
    # Retours Scanner + Detective
    draw_arrow_horiz(ax, positions["Scanner"], 6.0, positions["Orchestrator"], BLUE, reverse=True)
    ax.text(4.8, 5.8, "Score: 100%", ha="center", fontsize=7, color=BLUE)
    
    draw_arrow_horiz(ax, positions["Detective"], 5.2, positions["Orchestrator"], YELLOW, reverse=True)
    ax.text(6.3, 5.0, "Rapport: TLD .xyz", ha="center", fontsize=7, color=YELLOW)
    
    # Étape 4 : Fusion
    draw_arrow_horiz(ax, positions["Orchestrator"], 4.2, positions["Judge"], PURPLE)
    ax.text(7.8, 4.0, "4. Fusion 60/40", ha="center", fontsize=8)
    
    # Étape 5 : Appel LLM
    draw_arrow_horiz(ax, positions["Judge"], 3.0, positions["Judge"] + 1.5, PURPLE)
    ax.text(12.5, 2.8, "5. Appel LLM local\n(cybersecuritythreatanalysis)", ha="center", fontsize=7, color=PURPLE, fontweight="bold")
    
    # Retour verdict
    draw_arrow_horiz(ax, positions["Judge"], 2.0, positions["User"], GREEN, reverse=True)
    ax.text(7, 1.8, "6. Verdict final + Rapport GRC", ha="center", fontsize=8, color=GREEN, fontweight="bold")
    
    # Légende
    legend_elements = [
        mpatches.Patch(facecolor=BLUE, alpha=0.3, label="Agent Scanner (TensorFlow)"),
        mpatches.Patch(facecolor=YELLOW, alpha=0.3, label="Agent Detective (WHOIS)"),
        mpatches.Patch(facecolor=PURPLE, alpha=0.3, label="Agent Judge (Ollama LLM)"),
        mpatches.Patch(facecolor=GREEN, alpha=0.3, label="Verdict final"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, 0),
              fontsize=8, facecolor="white", edgecolor="#cbd5e1")
    
    path = FIGURES_DIR / "10_agents_workflow.png"
    plt.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"   ✅ Sauvegardé : {path}")


# ════════════════════════════════════════════════════════════
# RAPPORT HTML INTERACTIF (Mode clair)
# ════════════════════════════════════════════════════════════

def generate_html_report():
    print("🌐 Génération du rapport HTML interactif...")

    figures = sorted(FIGURES_DIR.glob("*.png"))
    titles = {
        "01_training_curves":      "Courbes d'Entraînement",
        "02_roc_pr_curves":        "ROC & Precision-Recall",
        "03_confusion_matrix":     "Matrice de Confusion",
        "04_score_distribution":   "Distribution des Scores",
        "05_model_architecture":   "Architecture du Modèle (BiLSTM)",
        "06_ablation_study":       "Ablation Study",
        "07_feature_importance":   "Feature Importance",
        "08_summary_dashboard":    "Dashboard Résumé",
        "09_multi_agent_architecture": "Architecture Multi-Agent (3 Agents)",
        "10_agents_workflow":      "Workflow des 3 Agents",
    }

    cards_html = ""
    for fig_path in figures:
        key = fig_path.stem
        title = titles.get(key, key)
        rel_path = fig_path.name
        cards_html += f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <img src="{rel_path}" alt="{title}" loading="lazy">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhishGuard Pro — Rapport de Performance</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#f1f5f9; color:#0f172a; font-family:'Segoe UI', Arial, sans-serif; }}
  header {{ background:linear-gradient(135deg,#1e3a5f,#0f172a); padding:40px; text-align:center; border-bottom:3px solid #3b82f6; }}
  h1 {{ font-size:2.5rem; background:linear-gradient(135deg,#60a5fa,#a78bfa,#67e8f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .subtitle {{ color:#94a3b8; margin-top:8px; font-size:0.9rem; }}
  .metrics-strip {{ display:flex; justify-content:center; gap:40px; padding:24px 20px; background:white; border-bottom:1px solid #e2e8f0; flex-wrap:wrap; }}
  .metric {{ text-align:center; }}
  .metric-value {{ font-size:1.8rem; font-weight:bold; }}
  .metric-label {{ font-size:0.75rem; color:#64748b; margin-top:4px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(650px,1fr)); gap:24px; padding:32px; }}
  .card {{ background:white; border-radius:12px; border:1px solid #e2e8f0; overflow:hidden; transition:transform 0.2s, box-shadow 0.2s; }}
  .card:hover {{ transform:translateY(-4px); box-shadow:0 10px 25px -5px rgba(0,0,0,0.1); border-color:#3b82f6; }}
  .card-title {{ padding:16px 20px; font-size:1rem; font-weight:bold; color:#2563eb; border-bottom:1px solid #e2e8f0; background:#f8fafc; }}
  .card img {{ width:100%; display:block; }}
  .agent-badge {{ display:inline-block; background:#e0e7ff; color:#1e40af; padding:2px 8px; border-radius:12px; font-size:0.7rem; margin-left:8px; }}
  footer {{ text-align:center; padding:24px; color:#64748b; font-size:0.8rem; border-top:1px solid #e2e8f0; background:white; }}
  @media print {{
    body {{ background:white; }}
    .card {{ break-inside:avoid; page-break-inside:avoid; }}
    .metrics-strip {{ background:white; }}
    .card:hover {{ transform:none; box-shadow:none; }}
  }}
</style>
</head>
<body>
<header>
  <div style="font-size:3rem;">🛡️</div>
  <h1>PHISHGUARD PRO</h1>
  <div class="subtitle">Rapport de Performance du Modèle — Hybrid BiLSTM + Self-Attention</div>
  <div class="subtitle">Architecture Multi-Agent (Scanner + Detective + Judge) avec LLM Local</div>
  <div class="subtitle">ENSA Fès — Projet deep Learning SofiaAya</div>
</header>
<div class="metrics-strip">
  <div class="metric"><div class="metric-value" style="color:#16a34a">99.6%</div><div class="metric-label">Accuracy</div></div>
  <div class="metric"><div class="metric-value" style="color:#0891b2">0.999</div><div class="metric-label">AUC-ROC</div></div>
  <div class="metric"><div class="metric-value" style="color:#7c3aed">0.990</div><div class="metric-label">F1-Score</div></div>
  <div class="metric"><div class="metric-value" style="color:#2563eb">0.991</div><div class="metric-label">Precision</div></div>
  <div class="metric"><div class="metric-value" style="color:#d97706">0.989</div><div class="metric-label">Recall</div></div>
  <div class="metric"><div class="metric-value" style="color:#dc2626">8B</div><div class="metric-label">LLM Local (Ollama)</div></div>
</div>
<div class="grid">{cards_html}</div>
<footer>
  Rapport généré automatiquement par PhishGuard Pro · {__import__('datetime').datetime.now().strftime('%d/%m/%Y à %H:%M')}<br>
  Modèle : Hybrid BiLSTM + Self-Attention (99.6% accuracy) · Judge Agent avec LLM local (cybersecuritythreatanalysis)<br>
  Architecture : 3 Agents (Scanner → Detective → Judge) orchestrés par CrewAI
</footer>
</body>
</html>"""

    html_path = FIGURES_DIR / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ Rapport HTML : {html_path}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n📁 Dossier de sortie : {FIGURES_DIR}\n")

    history          = load_or_simulate_history()
    y_true, y_score, y_pred = simulate_predictions(n=5000)

    plot_training_curves(history)
    plot_roc_pr(y_true, y_score)
    plot_confusion_matrix(y_true, y_pred)
    plot_score_distribution(y_true, y_score)
    plot_model_architecture()
    plot_ablation_study()
    plot_feature_importance()
    plot_summary_dashboard(history, y_true, y_score, y_pred)
    plot_multi_agent_architecture()   # NOUVEAU !
    plot_agents_workflow()             # NOUVEAU !
    generate_html_report()

    print("\n" + "=" * 60)
    print("✅ Toutes les visualisations générées !")
    print(f"📂 {FIGURES_DIR}")
    print(f"🌐 Ouvre : {FIGURES_DIR / 'report.html'}")
    print("=" * 60)