# 🎣 Phishing Detection — Hybrid Deep Learning

> BiLSTM + Self-Attention · Features structurelles · PhishTank + Alexa  
> **ENSA de Fès** — Module Deep Learning & NLP

---

##  Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Expected Results](#expected-results)
4. [Installation](#installation)
5. [Datasets](#datasets)
6. [Getting Started](#getting-started)
7. [Project Structure](#project-structure)

---

## Vue d'ensemble

Système hybride de détection de phishing combinant :

- **URLs brutes** → BiLSTM char-level + Self-Attention
- **18 features structurelles** → Dense Neural Network
- **Fusion layer** → décision finale robuste
- **Interface Gradio** → démonstration temps réel

### Pourquoi hybride ?

| Approche | Avantage | Limite |
|----------|----------|--------|
| LSTM seul | Apprend les patterns textuels | Ignore les méta-données |
| Features seules | Rapide, interprétable | Perd l'info séquentielle |
| **Hybride (notre approche)** | **Les deux** | **Aucune** |

---

## Architecture

```
URL brute (texte)
      │
      ▼
Embedding (64 dim)
      │
      ▼
BiLSTM (128 units) ──► Self-Attention ──► GlobalMaxPool ──► Dense(128) ──┐
                                                                           │
                                                                    Concat (Fusion)
                                                                           │
18 features URL ──► Dense(128) ──► BN ──► Dropout ──► Dense(64) ─────────┘
                                                                           │
                                                                    Dense(128) ──► Dense(1, sigmoid)
                                                                           │
                                                                    0 = Légitime
                                                                    1 = Phishing
```

---

## Résultats attendus

| Modèle | Accuracy | AUC-ROC |
|--------|----------|---------|
| Features only | ~93% | ~0.97 |
| LSTM only | ~96% | ~0.99 |
| **Hybrid (notre)** | **~98%** | **~0.997** |

---

## Installation

### 1. Extraire et entrer dans le projet

```bash
unzip phishing-hybrid-dl.zip
cd phishing-hybrid-dl
```

### 2. Créer un environnement virtuel

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Datasets

Le projet utilise **deux sources** téléchargées automatiquement :

| Source | Contenu | Label |
|--------|---------|-------|
| **PhishTank** | URLs phishing vérifiées en temps réel | 1 |
| **Alexa Top 1M** | Sites les plus visités au monde | 0 |

> ⚠️ Si PhishTank ou Alexa sont temporairement inaccessibles,
> le script utilise automatiquement un dataset de fallback intégré.

---

## Lancer le projet

### Pipeline complet en une commande

```bash
make all
```

### Étape par étape

```bash
# 1. Télécharger et préparer les données
make data
# ou :
python src/data/make_dataset.py

# 2. Entraîner le modèle hybride
make train
# ou :
python src/training/train.py --model hybrid

# 3. Entraîner tous les modèles (pour comparaison)
make train-all
# ou :
python src/training/train.py --model all

# 4. Évaluer et générer les figures
make evaluate

# 5. Lancer la démo Gradio
make app
# → Ouvre http://localhost:7860
```

### Modèles disponibles

```bash
python src/training/train.py --model hybrid        # Modèle principal
python src/training/train.py --model lstm_only     # Ablation LSTM
python src/training/train.py --model features_only # Ablation features
python src/training/train.py --model all           # Tous les modèles
```

---

## Structure

```
phishing-hybrid-dl/
│
├── config/
│   └── config.yaml              # Hyperparamètres
│
├── data/
│   ├── raw/                     # PhishTank + Alexa (gitignored)
│   └── processed/               # train/val/test .parquet (gitignored)
│
├── src/
│   ├── data/
│   │   └── make_dataset.py      # Téléchargement + split
│   │
│   ├── features/
│   │   ├── url_features.py      # 18 features structurelles
│   │   ├── char_tokenizer.py    # Tokeniseur char-level
│   │   └── feature_pipeline.py  # Pipeline complet
│   │
│   ├── models/
│   │   └── hybrid_model.py      # BiLSTM + Features + Fusion
│   │
│   ├── training/
│   │   └── train.py             # Entraînement + MLflow
│   │
│   ├── evaluation/
│   │   └── evaluate.py          # Métriques + courbes ROC
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       └── reproducibility.py
│
├── app/
│   └── app.py                   # Interface Gradio
│
├── models/                      # Checkpoints (gitignored)
├── reports/figures/             # Figures générées
├── tests/
├── Makefile
├── requirements.txt
└── README.md
```

---

## Features extraites (18)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Longueur totale de l'URL |
| 2 | `domain_length` | Longueur du domaine |
| 3 | `path_length` | Longueur du path |
| 4 | `subdomain_depth` | Nombre de sous-domaines |
| 5 | `is_https` | Présence de HTTPS |
| 6 | `digit_count` | Nombre de chiffres |
| 7 | `digit_ratio` | Ratio chiffres/longueur |
| 8 | `has_at` | Présence de "@" |
| 9 | `has_double_slash` | Présence de "//" dans le path |
| 10 | `dash_count` | Tirets dans le domaine |
| 11 | `dot_count` | Points dans l'URL |
| 12 | `query_params` | Nombre de paramètres |
| 13 | `domain_entropy` | Entropie de Shannon du domaine |
| 14 | `suspicious_keyword` | Mot-clé suspect (paypal, login...) |
| 15 | `suspicious_tld` | TLD suspect (.xyz, .tk, .ml...) |
| 16 | `longest_token` | Plus long token du path |
| 17 | `is_ip` | IP dans le domaine |
| 18 | `special_char_ratio` | Ratio caractères spéciaux |

---



---


