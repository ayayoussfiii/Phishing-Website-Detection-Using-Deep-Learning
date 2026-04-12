"""
feature_pipeline.py
-------------------
Pipeline complet : extraction features + normalisation + tokenisation.
Produit les 2 entrées du modèle hybride.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.features.url_features import extract_features_batch, get_feature_names
from src.features.char_tokenizer import CharTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Pipeline complet pour le modèle hybride.

    Produit :
        - X_seq   : (N, max_len)     → séquences char pour LSTM
        - X_feat  : (N, 18)          → features numériques normalisées
        - y       : (N,)             → labels
    """

    def __init__(self, max_len: int = 200):
        self.max_len = max_len
        self.scaler = StandardScaler()
        self.tokenizer = CharTokenizer()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """Fit sur train + transform. Appeler uniquement sur le train set."""
        logger.info("🔧 fit_transform sur le train set...")

        X_feat_raw = extract_features_batch(df["url"])
        X_feat = self.scaler.fit_transform(X_feat_raw)
        X_seq = self.tokenizer.transform(df["url"], self.max_len)
        y = df["label"].values.astype(np.float32)

        self._fitted = True
        logger.info(f"✅ X_seq={X_seq.shape}, X_feat={X_feat.shape}")
        return X_seq, X_feat.astype(np.float32), y

    def transform(self, df: pd.DataFrame):
        """Transform uniquement (val/test). Ne re-fit pas le scaler."""
        if not self._fitted:
            raise RuntimeError("Pipeline non fitté. Appelle fit_transform d'abord.")

        X_feat_raw = extract_features_batch(df["url"])
        X_feat = self.scaler.transform(X_feat_raw)
        X_seq = self.tokenizer.transform(df["url"], self.max_len)
        y = df["label"].values.astype(np.float32)
        return X_seq, X_feat.astype(np.float32), y

    def save(self, directory: Path):
        """Sauvegarde le pipeline (scaler + tokenizer)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        self.tokenizer.save(directory / "tokenizer.json")
        logger.info(f"💾 Pipeline sauvegardé → {directory}")

    @classmethod
    def load(cls, directory: Path) -> "FeaturePipeline":
        """Charge un pipeline sauvegardé."""
        directory = Path(directory)
        pipeline = cls.__new__(cls)

        with open(directory / "scaler.pkl", "rb") as f:
            pipeline.scaler = pickle.load(f)

        pipeline.tokenizer = CharTokenizer.load(directory / "tokenizer.json")
        pipeline._fitted = True
        pipeline.max_len = 200
        return pipeline
