# PhishGuard Pro — Détection de Phishing par Deep Learning & Agents IA

> Architecture Multi-Agent — BiLSTM + Self-Attention · 18 Features · LLM Local (Ollama) · Rapport GRC  
> ENSA de Fès — Projet Deep Learning SofiaAya2026

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


<img width="896" height="503" alt="image" src="https://github.com/user-attachments/assets/f9816a64-9b75-4b77-84c7-43807a54ee81" />
<img width="898" height="503" alt="image" src="https://github.com/user-attachments/assets/a8625d93-850a-4b8e-a17d-a000e0887a95" />
<img width="881" height="431" alt="image" src="https://github.com/user-attachments/assets/f9e3c8ff-cb6c-48cb-8379-7703165252fa" />
<img width="881" height="469" alt="image" src="https://github.com/user-attachments/assets/12c5e1fb-260f-41e7-afb4-79690728c579" />
<img width="886" height="426" alt="image" src="https://github.com/user-attachments/assets/b7f7a039-5e0c-48a1-81d3-0ce31a41747e" />
<img width="815" height="498" alt="image" src="https://github.com/user-attachments/assets/8561f4d0-fa2c-4980-9045-658b6adb14e5" />
<img width="789" height="304" alt="image" src="https://github.com/user-attachments/assets/b7a74449-dec2-480f-9187-41d81b34b21f" />
<img width="862" height="450" alt="image" src="https://github.com/user-attachments/assets/cb397371-8a65-4dd0-9923-a74fecf3ebba" />
<img width="880" height="482" alt="image" src="https://github.com/user-attachments/assets/3a5a2565-0fe6-4def-b8ae-e8c16f5d8db9" />
<img width="878" height="450" alt="image" src="https://github.com/user-attachments/assets/f1442204-faf2-4c60-a807-c3bb1fd79d94" />
<img width="880" height="191" alt="image" src="https://github.com/user-attachments/assets/6e2b6da8-6cbe-4cd7-8a8d-a0da41f59456" />









