"""
url_features.py
---------------
Extraction de 18 features structurelles depuis une URL brute.
Utilisé par la branche "features" du modèle hybride.

FIX : normalisation des features numériques pour éviter le biais
      lié à la longueur brute des URLs.
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

# ── Seuils de normalisation (basés sur les stats PhishTank + Alexa) ──────────
# Ces valeurs représentent le 95e percentile observé dans le dataset
# Au-delà → saturé à 1.0, en dessous → ratio proportionnel
_URL_LEN_MAX      = 200   # URLs > 200 chars → score saturé à 1.0
_DOMAIN_LEN_MAX   = 50
_PATH_LEN_MAX     = 150
_SUBDOMAIN_MAX    = 4
_DIGIT_COUNT_MAX  = 20
_DASH_MAX         = 5
_DOT_MAX          = 8
_QUERY_MAX        = 5
_TOKEN_MAX        = 30


def _shannon_entropy(s: str) -> float:
    """Entropie de Shannon d'une chaîne (mesure de aléatoire)."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _norm(value: float, max_val: float) -> float:
    """Normalise une valeur entre 0 et 1 avec saturation."""
    return min(value / max_val, 1.0)


def extract_url_features(url: str) -> List[float]:
    """
    Extrait 18 features numériques depuis une URL brute.
    Toutes les features sont normalisées entre 0.0 et 1.0.

    Returns:
        List[float] de longueur 18
    """
    url = str(url).strip()

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or ""
        path   = parsed.path or ""
        query  = parsed.query or ""
    except Exception:
        parsed = None
        domain = ""
        path   = ""
        query  = ""

    full = url.lower()

    # 1. Longueur totale de l'URL — normalisée [0, 1]
    f1_url_length = _norm(len(url), _URL_LEN_MAX)

    # 2. Longueur du domaine — normalisée [0, 1]
    f2_domain_length = _norm(len(domain), _DOMAIN_LEN_MAX)

    # 3. Longueur du path — normalisée [0, 1]
    f3_path_length = _norm(len(path), _PATH_LEN_MAX)

    # 4. Nombre de sous-domaines — normalisé [0, 1]
    parts = [p for p in domain.split(".") if p]
    f4_subdomain_depth = _norm(max(0, len(parts) - 2), _SUBDOMAIN_MAX)

    # 5. Présence de HTTPS — binaire {0, 1}
    f5_is_https = 1.0 if url.lower().startswith("https") else 0.0

    # 6. Nombre de chiffres — normalisé [0, 1]
    digit_count = sum(c.isdigit() for c in url)
    f6_digit_count = _norm(digit_count, _DIGIT_COUNT_MAX)

    # 7. Ratio chiffres / longueur URL — déjà entre [0, 1]
    f7_digit_ratio = digit_count / max(len(url), 1)

    # 8. Présence de "@" — binaire {0, 1}
    f8_has_at = 1.0 if "@" in url else 0.0

    # 9. Présence de "//" après le schéma — binaire {0, 1}
    f9_has_double_slash = 1.0 if "//" in path else 0.0

    # 10. Nombre de tirets "-" dans le domaine — normalisé [0, 1]
    f10_dash_count = _norm(domain.count("-"), _DASH_MAX)

    # 11. Nombre de points dans l'URL — normalisé [0, 1]
    f11_dot_count = _norm(url.count("."), _DOT_MAX)

    # 12. Nombre de paramètres de query — normalisé [0, 1]
    f12_query_params = _norm(len(urllib.parse.parse_qs(query)), _QUERY_MAX)

    # 13. Entropie du domaine — normalisée [0, 1] (max théorique ≈ 4.7 bits)
    f13_domain_entropy = _norm(_shannon_entropy(domain), 4.7)

    # 14. Mot-clé suspect dans l'URL — binaire {0, 1}
    f14_suspicious_keyword = float(
        any(kw in full for kw in SUSPICIOUS_KEYWORDS)
    )

    # 15. TLD suspect — binaire {0, 1}
    f15_suspicious_tld = float(
        any(full.endswith(tld) or f".{tld.lstrip('.')}/" in full
            for tld in SUSPICIOUS_TLDS)
    )

    # 16. Longueur du plus long token dans le path — normalisée [0, 1]
    tokens = re.split(r"[/\-_?=&.]", path)
    longest = max((len(t) for t in tokens if t), default=0)
    f16_longest_token = _norm(longest, _TOKEN_MAX)

    # 17. Présence d'une IP dans le domaine — binaire {0, 1}
    ip_pattern = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
    f17_is_ip = 1.0 if ip_pattern.search(domain) else 0.0

    # 18. Ratio de caractères spéciaux dans le path — déjà entre [0, 1]
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