"""
tests/test_features.py
----------------------
Tests unitaires pour l'extraction de features URL.
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.url_features import extract_url_features, FEATURE_NAMES
from src.features.char_tokenizer import CharTokenizer
import numpy as np
import pandas as pd


class TestURLFeatures:

    def test_feature_length(self):
        """Vérifie qu'on extrait bien 18 features."""
        features = extract_url_features("http://google.com")
        assert len(features) == 18

    def test_https_detection(self):
        """HTTPS doit être détecté."""
        f_https = extract_url_features("https://google.com")
        f_http = extract_url_features("http://google.com")
        assert f_https[4] == 1   # is_https
        assert f_http[4] == 0

    def test_suspicious_keyword(self):
        """Mot-clé suspect doit être détecté."""
        f_phish = extract_url_features("http://paypal-secure-login.xyz/verify")
        f_legit = extract_url_features("http://google.com")
        assert f_phish[13] == 1   # suspicious_keyword
        assert f_legit[13] == 0

    def test_suspicious_tld(self):
        """TLD suspect doit être détecté."""
        f_phish = extract_url_features("http://example.xyz/page")
        f_legit = extract_url_features("http://example.com/page")
        assert f_phish[14] == 1   # suspicious_tld
        assert f_legit[14] == 0

    def test_at_symbol(self):
        """@ dans l'URL doit être détecté."""
        f = extract_url_features("http://user@paypal.com/login")
        assert f[7] == 1   # has_at

    def test_ip_detection(self):
        """IP dans le domaine doit être détectée."""
        f = extract_url_features("http://192.168.1.1/login")
        assert f[16] == 1   # is_ip

    def test_feature_names_count(self):
        """Nombre de noms de features == 18."""
        assert len(FEATURE_NAMES) == 18

    def test_empty_url(self):
        """URL vide ne doit pas planter."""
        features = extract_url_features("")
        assert len(features) == 18

    def test_all_numeric(self):
        """Toutes les features doivent être numériques."""
        features = extract_url_features("https://www.amazon-login.tk/secure?user=test")
        for f in features:
            assert isinstance(f, (int, float))


class TestCharTokenizer:

    def test_encode_decode(self):
        """Encode une URL simple."""
        tok = CharTokenizer()
        encoded = tok.encode("http://google.com")
        assert len(encoded) > 0
        assert all(isinstance(i, int) for i in encoded)

    def test_transform_shape(self):
        """Transform doit retourner la bonne shape."""
        tok = CharTokenizer()
        urls = pd.Series(["http://google.com", "http://phish.xyz/login"])
        X = tok.transform(urls, max_len=100)
        assert X.shape == (2, 100)

    def test_padding(self):
        """Les séquences courtes doivent être paddées."""
        tok = CharTokenizer()
        urls = pd.Series(["ab"])
        X = tok.transform(urls, max_len=50)
        assert X.shape[1] == 50
        assert X[0, 2] == 0   # padding à droite

    def test_truncation(self):
        """Les séquences longues doivent être tronquées."""
        tok = CharTokenizer()
        urls = pd.Series(["a" * 300])
        X = tok.transform(urls, max_len=100)
        assert X.shape[1] == 100

    def test_vocab_size(self):
        """Vocab size doit être > 2."""
        tok = CharTokenizer()
        assert tok.vocab_size > 2
