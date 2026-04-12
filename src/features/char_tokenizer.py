"""
char_tokenizer.py
-----------------
Tokeniseur caractère par caractère pour la branche LSTM du modèle hybride.
Encode les URLs brutes en séquences d'entiers.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ─────────────────────────────────────────────────────────────
# VOCABULAIRE
# ─────────────────────────────────────────────────────────────

# Tous les caractères présents dans des URLs
URL_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".-_~:/?#[]@!$&'()*+,;=%"
)

PAD_TOKEN = 0
UNK_TOKEN = 1


class CharTokenizer:
    """
    Tokeniseur char-level pour URLs.

    Exemple:
        tok = CharTokenizer()
        tok.fit(train_urls)
        X = tok.transform(urls, max_len=200)
    """

    def __init__(self, vocab: Optional[str] = None):
        chars = vocab or URL_CHARS
        # index 0 = padding, 1 = unknown
        self.char2idx = {c: i + 2 for i, c in enumerate(chars)}
        self.idx2char = {i + 2: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 2   # +2 pour PAD et UNK

    def encode(self, url: str) -> List[int]:
        """Encode une URL en liste d'entiers."""
        return [self.char2idx.get(c, UNK_TOKEN) for c in str(url)]

    def transform(self, urls: pd.Series, max_len: int = 200) -> np.ndarray:
        """Transforme une série d'URLs en matrice (N, max_len)."""
        sequences = [self.encode(url) for url in urls]
        padded = pad_sequences(
            sequences,
            maxlen=max_len,
            padding="post",
            truncating="post",
            value=PAD_TOKEN,
        )
        return padded.astype(np.int32)

    def save(self, path: Path):
        """Sauvegarde le tokeniseur en JSON."""
        data = {
            "char2idx": self.char2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """Charge un tokeniseur depuis JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {v: k for k, v in tok.char2idx.items()}
        tok.vocab_size = data["vocab_size"]
        return tok
