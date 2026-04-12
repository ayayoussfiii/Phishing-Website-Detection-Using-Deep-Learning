"""
tests/test_models.py
--------------------
Tests unitaires pour les architectures de modèles.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_model import (
    build_hybrid_model,
    build_lstm_only_model,
    build_features_only_model,
)


VOCAB_SIZE = 100
MAX_LEN = 50
N_FEATURES = 18
BATCH = 8


class TestHybridModel:

    def test_build(self):
        """Le modèle hybride doit se construire sans erreur."""
        model = build_hybrid_model(VOCAB_SIZE, MAX_LEN, N_FEATURES)
        assert model is not None

    def test_output_shape(self):
        """Output doit être (batch, 1)."""
        model = build_hybrid_model(VOCAB_SIZE, MAX_LEN, N_FEATURES)
        X_seq = np.random.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))
        X_feat = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = model.predict([X_seq, X_feat], verbose=0)
        assert y.shape == (BATCH, 1)

    def test_output_range(self):
        """Output doit être entre 0 et 1 (sigmoid)."""
        model = build_hybrid_model(VOCAB_SIZE, MAX_LEN, N_FEATURES)
        X_seq = np.random.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))
        X_feat = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = model.predict([X_seq, X_feat], verbose=0)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_trainable_params(self):
        """Le modèle doit avoir des paramètres entraînables."""
        model = build_hybrid_model(VOCAB_SIZE, MAX_LEN, N_FEATURES)
        assert model.count_params() > 0

    def test_two_inputs(self):
        """Le modèle doit avoir exactement 2 entrées."""
        model = build_hybrid_model(VOCAB_SIZE, MAX_LEN, N_FEATURES)
        assert len(model.inputs) == 2


class TestLSTMOnlyModel:

    def test_build(self):
        model = build_lstm_only_model(VOCAB_SIZE, MAX_LEN)
        assert model is not None

    def test_output_shape(self):
        model = build_lstm_only_model(VOCAB_SIZE, MAX_LEN)
        X_seq = np.random.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))
        y = model.predict(X_seq, verbose=0)
        assert y.shape == (BATCH, 1)

    def test_one_input(self):
        model = build_lstm_only_model(VOCAB_SIZE, MAX_LEN)
        assert len(model.inputs) == 1


class TestFeaturesOnlyModel:

    def test_build(self):
        model = build_features_only_model(N_FEATURES)
        assert model is not None

    def test_output_shape(self):
        model = build_features_only_model(N_FEATURES)
        X_feat = np.random.randn(BATCH, N_FEATURES).astype(np.float32)
        y = model.predict(X_feat, verbose=0)
        assert y.shape == (BATCH, 1)
