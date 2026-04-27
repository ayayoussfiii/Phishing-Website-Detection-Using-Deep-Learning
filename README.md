# PhishGuard Pro — Détection de Phishing par Deep Learning & Agents IA

> Architecture Multi-Agent — BiLSTM + Self-Attention · 18 Features · LLM Local (Ollama) · Rapport GRC  
> ENSA de Fès — Projet de Fin d'Études 2026

---

## Aperçu

PhishGuard Pro est un système intelligent de détection de phishing basé sur une architecture multi-agent combinant Deep Learning, heuristiques et LLM local.

---

## Architecture

URL → Scanner (BiLSTM) + Detective (WHOIS) → Judge (LLM) → Rapport GRC

---

## Modèle

- Embedding
- BiLSTM
- Self-Attention
- Fusion avec 18 features
- Classification sigmoid

---

## Résultats

- Accuracy: 99.6% (Hybride)
- Accuracy: 100% (Multi-Agent)

---

##  Datasets

- PhishTank (phishing)
- Tranco (légitime)

---

##  Installation

```bash
git clone https://github.com/ayayoussfiii/Phishing-Website-Detection-Using-Deep-Learning.git
cd Phishing-Website-Detection-Using-Deep-Learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

##  Utilisation

```bash
python app.py runserver
```

---

##  Structure

- src/
- app/
- models/
- data/

---

## Features

18 features structurelles (URL length, HTTPS, entropy, etc.)

---

## Rapport GRC

- Governance
- Risk Assessment
- Compliance
- Recommendations

---

##  Technologies

- TensorFlow
- Django
- CrewAI
- Ollama

---


