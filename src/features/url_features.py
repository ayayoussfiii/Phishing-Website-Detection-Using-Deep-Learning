"""
url_features.py
---------------
Extraction de 18 features structurelles depuis une URL brute.
Utilisé par la branche "features" du modèle hybride.
"""

import re
import math
import urllib.parse
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# FEATURES INDIVIDUELLES
# ─────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update",
    "confirm", "banking", "password", "credential", "paypal",
    "amazon", "apple", "microsoft", "google", "facebook",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw",
    ".top", ".club", ".online", ".site", ".info", ".biz",
]

LEGIT_TLDS = [".com", ".org", ".edu", ".gov", ".net"]


def _shannon_entropy(s: str) -> float:
    """Entropie de Shannon d'une chaîne (mesure de aléatoire)."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def extract_url_features(url: str) -> List[float]:
    """
    Extrait 18 features numériques depuis une URL brute.

    Returns:
        List[float] de longueur 18
    """
    url = str(url).strip()

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        parsed = None
        domain = ""
        path = ""
        query = ""

    full = url.lower()

    # 1. Longueur totale de l'URL
    f1_url_length = len(url)

    # 2. Longueur du domaine
    f2_domain_length = len(domain)

    # 3. Longueur du path
    f3_path_length = len(path)

    # 4. Nombre de sous-domaines
    parts = [p for p in domain.split(".") if p]
    f4_subdomain_depth = max(0, len(parts) - 2)

    # 5. Présence de HTTPS
    f5_is_https = 1 if url.lower().startswith("https") else 0

    # 6. Nombre de chiffres dans l'URL
    f6_digit_count = sum(c.isdigit() for c in url)

    # 7. Ratio chiffres / longueur URL
    f7_digit_ratio = f6_digit_count / max(len(url), 1)

    # 8. Présence de "@"
    f8_has_at = 1 if "@" in url else 0

    # 9. Présence de "//"  après le schéma
    f9_has_double_slash = 1 if "//" in path else 0

    # 10. Nombre de tirets "-" dans le domaine
    f10_dash_count = domain.count("-")

    # 11. Nombre de points dans l'URL
    f11_dot_count = url.count(".")

    # 12. Nombre de paramètres de query
    f12_query_params = len(urllib.parse.parse_qs(query))

    # 13. Entropie du domaine
    f13_domain_entropy = _shannon_entropy(domain)

    # 14. Mot-clé suspect dans l'URL
    f14_suspicious_keyword = int(
        any(kw in full for kw in SUSPICIOUS_KEYWORDS)
    )

    # 15. TLD suspect
    f15_suspicious_tld = int(
        any(full.endswith(tld) or f".{tld.lstrip('.')}/" in full
            for tld in SUSPICIOUS_TLDS)
    )

    # 16. Longueur du plus long token dans le path
    tokens = re.split(r"[/\-_?=&.]", path)
    f16_longest_token = max((len(t) for t in tokens if t), default=0)

    # 17. Présence d'une IP dans le domaine
    ip_pattern = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
    f17_is_ip = 1 if ip_pattern.search(domain) else 0

    # 18. Ratio de caractères spéciaux dans le path
    special_chars = sum(1 for c in path if not c.isalnum() and c not in "/.")
    f18_special_char_ratio = special_chars / max(len(path), 1)

    return [
        f1_url_length, f2_domain_length, f3_path_length,
        f4_subdomain_depth, f5_is_https, f6_digit_count,
        f7_digit_ratio, f8_has_at, f9_has_double_slash,
        f10_dash_count, f11_dot_count, f12_query_params,
        f13_domain_entropy, f14_suspicious_keyword,
        f15_suspicious_tld, f16_longest_token,
        f17_is_ip, f18_special_char_ratio,
    ]


FEATURE_NAMES = [
    "url_length", "domain_length", "path_length",
    "subdomain_depth", "is_https", "digit_count",
    "digit_ratio", "has_at", "has_double_slash",
    "dash_count", "dot_count", "query_params",
    "domain_entropy", "suspicious_keyword",
    "suspicious_tld", "longest_token",
    "is_ip", "special_char_ratio",
]


# ─────────────────────────────────────────────────────────────
# BATCH EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_features_batch(urls: pd.Series) -> np.ndarray:
    """Extrait les features pour un batch d'URLs."""
    features = []
    for url in tqdm(urls, desc="Extraction features", leave=False):
        features.append(extract_url_features(url))
    return np.array(features, dtype=np.float32)


def get_feature_names() -> List[str]:
    return FEATURE_NAMES
