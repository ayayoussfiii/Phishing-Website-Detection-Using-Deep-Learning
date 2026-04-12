"""
tests/test_features.py
----------------------
Unit tests for URL feature extraction and character tokenizer.
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.url_features import extract_url_features, FEATURE_NAMES
from src.features.char_tokenizer import CharTokenizer


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def legit_url():
    return "https://www.google.com/search?q=python"

@pytest.fixture
def phishing_url():
    return "http://paypal-secure-login.xyz/verify?user=admin"

@pytest.fixture
def ip_url():
    return "http://192.168.1.1/login?bank=chase&password=reset"

@pytest.fixture
def tokenizer():
    return CharTokenizer()


# ─────────────────────────────────────────────────────────────
# URL FEATURES
# ─────────────────────────────────────────────────────────────

class TestURLFeatures:

    def test_feature_vector_length(self, legit_url):
        """Feature vector must always contain exactly 18 values."""
        features = extract_url_features(legit_url)
        assert len(features) == 18, f"Expected 18 features, got {len(features)}"

    def test_feature_names_count(self):
        """FEATURE_NAMES list must contain exactly 18 entries."""
        assert len(FEATURE_NAMES) == 18, (
            f"Expected 18 feature names, got {len(FEATURE_NAMES)}"
        )

    def test_all_features_numeric(self, phishing_url):
        """All feature values must be int or float (no None, no str)."""
        features = extract_url_features(phishing_url)
        for name, val in zip(FEATURE_NAMES, features):
            assert isinstance(val, (int, float)), (
                f"Feature '{name}' returned non-numeric value: {val!r}"
            )

    def test_https_flag_true(self):
        """is_https must be 1 for HTTPS URLs."""
        f = extract_url_features("https://www.google.com")
        assert f[4] == 1, "Expected is_https=1 for https:// URL"

    def test_https_flag_false(self):
        """is_https must be 0 for HTTP URLs."""
        f = extract_url_features("http://www.google.com")
        assert f[4] == 0, "Expected is_https=0 for http:// URL"

    def test_suspicious_keyword_detected(self, phishing_url):
        """suspicious_keyword must be 1 when login/verify/secure/… is present."""
        f = extract_url_features(phishing_url)
        assert f[13] == 1, (
            f"Expected suspicious_keyword=1 for URL: {phishing_url}"
        )

    def test_suspicious_keyword_absent(self, legit_url):
        """suspicious_keyword must be 0 for clean legitimate URLs."""
        f = extract_url_features("http://www.example.com/about")
        assert f[13] == 0, "Expected suspicious_keyword=0 for clean URL"

    def test_suspicious_tld_detected(self):
        """suspicious_tld must be 1 for .xyz / .tk / .ml / … TLDs."""
        f = extract_url_features("http://example.xyz/page")
        assert f[14] == 1, "Expected suspicious_tld=1 for .xyz TLD"

    def test_suspicious_tld_absent(self):
        """suspicious_tld must be 0 for .com / .org / .edu TLDs."""
        f = extract_url_features("http://example.com/page")
        assert f[14] == 0, "Expected suspicious_tld=0 for .com TLD"

    def test_at_symbol_detected(self):
        """has_at must be 1 when @ appears in the URL."""
        f = extract_url_features("http://user@paypal.com/login")
        assert f[7] == 1, "Expected has_at=1 when URL contains @"

    def test_at_symbol_absent(self, legit_url):
        """has_at must be 0 when @ is not present."""
        f = extract_url_features(legit_url)
        assert f[7] == 0, "Expected has_at=0 for URL without @"

    def test_ip_address_detected(self, ip_url):
        """is_ip must be 1 when domain is a raw IP address."""
        f = extract_url_features(ip_url)
        assert f[16] == 1, f"Expected is_ip=1 for URL: {ip_url}"

    def test_ip_address_absent(self, legit_url):
        """is_ip must be 0 for normal domain names."""
        f = extract_url_features(legit_url)
        assert f[16] == 0, "Expected is_ip=0 for domain-based URL"

    def test_url_length_positive(self, legit_url):
        """url_length feature must be strictly positive."""
        f = extract_url_features(legit_url)
        assert f[0] > 0, "Expected url_length > 0"

    def test_digit_ratio_in_range(self, phishing_url):
        """digit_ratio must be between 0.0 and 1.0."""
        f = extract_url_features(phishing_url)
        assert 0.0 <= f[6] <= 1.0, (
            f"digit_ratio out of range [0,1]: {f[6]}"
        )

    def test_empty_url_does_not_crash(self):
        """Empty string must not raise an exception — return 18 zeros."""
        features = extract_url_features("")
        assert len(features) == 18, "Empty URL must still return 18 features"

    def test_very_long_url(self):
        """Extremely long URLs must not crash the extractor."""
        long_url = "http://malicious.tk/" + "a" * 2000 + "?login=1"
        features = extract_url_features(long_url)
        assert len(features) == 18

    def test_phishing_scores_higher_than_legit(self, phishing_url, legit_url):
        """Phishing URL should have higher raw risk signals than a clean URL."""
        f_phish = extract_url_features(phishing_url)
        f_legit = extract_url_features(legit_url)
        phish_score = sum([f_phish[13], f_phish[14], f_phish[7], f_phish[16]])
        legit_score = sum([f_legit[13], f_legit[14], f_legit[7], f_legit[16]])
        assert phish_score >= legit_score, (
            "Phishing URL should trigger more risk signals than a legit URL"
        )


# ─────────────────────────────────────────────────────────────
# CHAR TOKENIZER
# ─────────────────────────────────────────────────────────────

class TestCharTokenizer:

    def test_encode_returns_integers(self, tokenizer):
        """encode() must return a list of integers."""
        encoded = tokenizer.encode("http://google.com")
        assert len(encoded) > 0
        assert all(isinstance(i, int) for i in encoded), (
            "encode() must return a list of integers"
        )

    def test_transform_output_shape(self, tokenizer):
        """transform() output shape must be (n_urls, max_len)."""
        urls = pd.Series(["http://google.com", "http://phish.xyz/login"])
        X = tokenizer.transform(urls, max_len=100)
        assert X.shape == (2, 100), (
            f"Expected shape (2, 100), got {X.shape}"
        )

    def test_padding_short_sequence(self, tokenizer):
        """Short URLs must be zero-padded to max_len on the right."""
        urls = pd.Series(["ab"])
        X = tokenizer.transform(urls, max_len=50)
        assert X.shape[1] == 50
        assert X[0, -1] == 0, "Last token of a short URL should be padding (0)"

    def test_truncation_long_sequence(self, tokenizer):
        """URLs longer than max_len must be truncated, not crash."""
        urls = pd.Series(["a" * 300])
        X = tokenizer.transform(urls, max_len=100)
        assert X.shape[1] == 100, (
            f"Expected truncated length 100, got {X.shape[1]}"
        )

    def test_vocab_size_reasonable(self, tokenizer):
        """Vocabulary size must be at least 3 (pad + unk + 1 char)."""
        assert tokenizer.vocab_size > 2, (
            f"vocab_size too small: {tokenizer.vocab_size}"
        )

    def test_different_urls_different_encodings(self, tokenizer):
        """Two distinct URLs must not produce identical encodings."""
        urls = pd.Series([
            "https://www.google.com",
            "http://paypal-secure-login.xyz/verify",
        ])
        X = tokenizer.transform(urls, max_len=100)
        assert not np.array_equal(X[0], X[1]), (
            "Different URLs must produce different token sequences"
        )

    def test_same_url_deterministic(self, tokenizer):
        """Same URL must always produce the same encoding."""
        url = "https://www.wikipedia.org/wiki/Deep_learning"
        urls = pd.Series([url, url])
        X = tokenizer.transform(urls, max_len=100)
        assert np.array_equal(X[0], X[1]), (
            "Same URL must produce identical encodings every time"
        )

    def test_batch_transform_many_urls(self, tokenizer):
        """Batch transform must handle 1000+ URLs without error."""
        urls = pd.Series([f"http://site{i}.com/page" for i in range(1000)])
        X = tokenizer.transform(urls, max_len=100)
        assert X.shape == (1000, 100)
