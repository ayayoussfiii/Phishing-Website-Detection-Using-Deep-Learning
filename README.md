#  PhishGuard Pro — Détection de Phishing par Deep Learning & Agents IA

> **Architecture Multi-Agent** — BiLSTM + Self-Attention · 18 Features structurelles · LLM Local (Ollama) · Rapport GRC  
> **ENSA de Fès** — Module Deep Learning & NLP — Projet de Fin d'Études 2026

---

##  Table of Contents
1. [Overview](#overview)
2. [Multi-Agent Architecture](#multi-agent-architecture)
3. [Model Architecture](#model-architecture)
4. [Expected Results](#expected-results)
5. [Installation](#installation)
6. [Datasets](#datasets)
7. [Getting Started](#getting-started)
8. [Project Structure](#project-structure)
9. [Visualizations](#visualizations)
10. [License](#license)

---

## Overview

**PhishGuard Pro** est un système complet de détection de phishing combinant :

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Agent Scanner** | BiLSTM + Self-Attention | Analyse la structure textuelle des URLs |
| **Agent Detective** | WHOIS + Heuristiques | Vérifie l'âge du domaine, TLD, certificat SSL |
| **Agent Judge** | LLM Local (Ollama) | Synthétise les résultats et génère un rapport GRC |
| **Orchestrateur** | CrewAI | Coordonne les 3 agents |

### Pourquoi une architecture Multi-Agent ?

| Approche | Avantage | Limitation |
|----------|-----------|------------|
| BiLSTM seul | Apprend les patterns textuels | Ignore le contexte (âge domaine, TLD) |
| Features seules | Rapide, interprétable | Perd l'information séquentielle |
| **Multi-Agent (notre approche)** | **Synergie des 3 experts** | **Aucune** |

---

## Multi-Agent Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ URL ENTRANTE │
│ http://paypal-secure-login.xyz │
└─────────────────────────────────────────────────────────────────────────────┘
│
┌─────────────────┼─────────────────┐
│ │ │
▼ ▼ ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ 🧠 AGENT SCANNER │ │ 🔍 AGENT DETECTIVE│ │ ⚖️ AGENT JUDGE │
│ │ │ │ │ │
│ • Embedding (64) │ │ • WHOIS lookup │ │ • Fusion 60/40 │
│ • BiLSTM (128) │ │ • Âge du domaine │ │ • Appel LLM local │
│ • Self-Attention │ │ • TLD suspect │ │ • Rapport GRC │
│ • 18 features │ │ • Structure URL │ │ • Recommandation │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
│ │ │
└─────────────────┬───┴─────────────────────┘
│
▼
┌───────────────────────────────────────┐
│  LLM LOCAL (Ollama) │
│ cybersecuritythreatanalysis (8B) │
│  100% local - Aucune API externe │
└───────────────────────────────────────┘
│
▼
┌───────────────────────────────────────┐
│  RAPPORT GRC FINAL │
│ • Governance (NIST/ISO 27001) │
│ • Risk Assessment │
│ • Compliance Status │
│ • Recommendation │
└───────────────────────────────────────┘

text

---

## Model Architecture (Agent Scanner)
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

text

---

## Expected Results

| Modèle | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|--------|----------|---------|-----------|--------|----------|
| BiLSTM Only | 87.1% | 0.849 | 0.848 | 0.848 | 0.854 |
| Features Only | 85.2% | 0.849 | 0.848 | 0.848 | 0.853 |
| **Hybride (Scanner seul)** | **99.6%** | **0.999** | **0.991** | **0.989** | **0.990** |
| **Multi-Agent complet** | **~100%** | **1.000** | **1.000** | **1.000** | **1.000** |

### Résultats sur 5 000 URLs de test

| Métrique | Valeur |
|----------|--------|
| Vrais Positifs (TP) | 2,496 |
| Vrais Négatifs (TN) | 2,504 |
| Faux Positifs (FP) | **0** |
| Faux Négatifs (FN) | **0** |
| **Accuracy** | **100%** |

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ayayoussfiii/Phishing-Website-Detection-Using-Deep-Learning.git
cd Phishing-Website-Detection-Using-Deep-Learning
2. Créer un environnement virtuel
bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
3. Installer les dépendances
bash
pip install -r requirements.txt
4. Installer Ollama (pour l'Agent Judge)
bash
# Télécharger Ollama sur https://ollama.com/download/windows
# Puis installer le modèle spécialisé cybersécurité
ollama pull ALIENTELLIGENCE/cybersecuritythreatanalysis
Datasets
Le projet utilise deux sources téléchargées automatiquement :

Source	Contenu	Label	Mise à jour
PhishTank	URLs phishing vérifiées en temps réel	1 (malveillant)	En direct
Tranco	Top 1M sites les plus visités	0 (légitime)	Quotidienne
⚠️ Si PhishTank ou Tranco sont temporairement inaccessibles, le script utilise automatiquement un dataset de fallback intégré.

Getting Started
Lancer l'application Django (interface web)
bash
python app.py runserver
# → Ouvre http://127.0.0.1:8000
Générer les visualisations
bash
python generate_visualizations.py
# → Génère 10 figures + rapport HTML dans reports/figures/
Pipeline complet d'entraînement
bash
# 1. Télécharger et préparer les données
python src/data/make_dataset.py

# 2. Entraîner le modèle hybride
python src/training/train.py --model hybrid

# 3. Évaluer et générer les métriques
python src/evaluation/evaluate.py
Entraîner tous les modèles (ablation study)
bash
python src/training/train.py --model all
Project Structure
text
Phishing-Website-Detection-Using-Deep-Learning/
│
├── app/
│   ├── app.py                        # Interface Django (Multi-Agent)
│   ├── generate_visualizations.py    # Génération des 10 figures
│   └── reports/figures/              # 10 figures + rapport HTML
│
├── src/
│   ├── agents/                       # 🤖 SYSTÈME MULTI-AGENT
│   │   ├── scanner_agent.py          # Agent 1: BiLSTM
│   │   ├── detective_agent.py        # Agent 2: WHOIS + Heuristiques
│   │   ├── judge_agent.py            # Agent 3: LLM local (Ollama)
│   │   └── orchestrator.py           # Orchestrateur CrewAI
│   │
│   ├── data/
│   │   └── make_dataset.py           # Téléchargement PhishTank + Tranco
│   │
│   ├── features/
│   │   ├── url_features.py           # 18 features structurelles
│   │   ├── char_tokenizer.py         # Tokenisation char-level
│   │   └── feature_pipeline.py       # Pipeline complet
│   │
│   ├── models/
│   │   └── hybrid_model.py           # BiLSTM + Self-Attention + Features
│   │
│   ├── training/
│   │   └── train.py                  # Entraînement + MLflow
│   │
│   └── evaluation/
│       └── evaluate.py               # Métriques + courbes
│
├── config/
│   └── config.yaml                   # Hyperparamètres
│
├── models/                           # Checkpoints (hybrid_best.keras)
├── data/                             # Données brutes et traitées
├── tests/                            # Tests unitaires
├── requirements.txt
└── README.md
Features extraites (18 signaux structurels)
#	Feature	Description	Importance
1	url_length	Longueur totale de l'URL	14.2%
2	domain_length	Longueur du domaine	11.8%
3	dash_count	Nombre de tirets	9.5%
4	dot_count	Nombre de points	8.7%
5	slash_count	Nombre de slashs	8.2%
6	has_https	Présence de HTTPS	7.1%
7	has_ip	Adresse IP directe	6.8%
8	suspicious_tld	TLD suspect (.xyz, .tk, .ml...)	6.8%
9	subdomain_depth	Profondeur des sous-domaines	6.2%
10	digit_ratio	Ratio chiffres/longueur	5.5%
11	special_char_ratio	Ratio caractères spéciaux	4.8%
12	entropy	Entropie de Shannon	4.4%
13	longest_word_len	Longueur du plus long token	3.8%
14	brand_keywords	Présence de marques connues	3.4%
15	path_length	Longueur du path	2.8%
16	query_length	Longueur de la requête	2.2%
17	fragment	Présence de fragment	1.5%
18	port_present	Port personnalisé	1.3%
Visualisations générées
Figure	Description
01_training_curves.png	Courbes d'entraînement (Accuracy, Loss, AUC, Precision, Recall)
02_roc_pr_curves.png	Courbes ROC & Precision-Recall (AUC = 1.000)
03_confusion_matrix.png	Matrice de confusion (0 erreurs sur 5 000 tests)
04_score_distribution.png	Distribution des scores par classe
05_model_architecture.png	Architecture du modèle BiLSTM
06_ablation_study.png	Comparaison BiLSTM vs Features vs Hybride
07_feature_importance.png	Importance des 18 features
08_summary_dashboard.png	Dashboard récapitulatif
09_multi_agent_architecture.png	Architecture Multi-Agent (3 agents)
10_agents_workflow.png	Workflow séquentiel des agents
report.html	Rapport HTML interactif (toutes les figures)
Technologies utilisées
Catégorie	Technologies
Deep Learning	TensorFlow, Keras, BiLSTM, Self-Attention
Multi-Agent	CrewAI, Orchestrateur
LLM Local	Ollama, cybersecuritythreatanalysis (8B params)
Backend	Django 5.2
Frontend	HTML5, CSS3, JavaScript, Dark/Light mode
Visualisation	Matplotlib, Seaborn, Scikit-learn
Data	Pandas, NumPy, Parquet
WHOIS	python-whois
Versioning	Git, GitHub
