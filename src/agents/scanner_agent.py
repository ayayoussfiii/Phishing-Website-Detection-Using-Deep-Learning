"""
scanner_agent.py
----------------
Agent 1 : Scanner — utilise le modèle BiLSTM hybride pour scorer l'URL.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ScanResult:
    url: str
    prob: float
    pct: int
    level: str          # safe | suspicious | phishing
    features: list      # [{name, value}, ...]
    status: str


class ScannerAgent:
    """
    Agent Scanner : prédit le score de phishing via le modèle BiLSTM.
    """
    name = "Scanner"

    def __init__(self, model, pipeline, feature_names, extract_fn):
        self.model        = model
        self.pipeline     = pipeline
        self.feature_names = feature_names
        self.extract_fn   = extract_fn

    def run(self, url: str) -> ScanResult:
        df = pd.DataFrame({"url": [url], "label": [0]})
        X_seq, X_feat, _ = self.pipeline.transform(df)
        prob = float(self.model.predict([X_seq, X_feat], verbose=0)[0][0])
        pct  = int(prob * 100)

        if prob < 0.3:
            level = "safe"
        elif prob < 0.7:
            level = "suspicious"
        else:
            level = "phishing"

        raw_feats = self.extract_fn(url)
        features  = [
            {"name": n, "value": round(float(v), 4)}
            for n, v in zip(self.feature_names, raw_feats)
        ]

        return ScanResult(
            url=url, prob=prob, pct=pct,
            level=level, features=features,
            status=level.capitalize()
        )