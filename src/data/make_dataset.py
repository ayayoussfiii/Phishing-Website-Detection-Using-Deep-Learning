"""
make_dataset.py
---------------
Télécharge PhishTank (phishing) + Alexa/Tranco Top 1M (légitimes)
et construit le dataset final combiné.

Usage:
    python src/data/make_dataset.py
"""

import os
import sys
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ajouter la racine du projet au path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
ALEXA_URL = "http://s3.amazonaws.com/alexa-static/top-1m.csv.zip"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT
# ─────────────────────────────────────────────────────────────
def download_phishtank(save_path: Path) -> pd.DataFrame:
    """Télécharge le dataset PhishTank (URLs phishing vérifiées)."""
    logger.info("📥 Téléchargement PhishTank...")

    try:
        headers = {"User-Agent": "phishing-research-project/1.0"}
        resp = requests.get(PHISHTANK_URL, headers=headers, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), usecols=["url"])
        df["label"] = 1  # phishing
        df.to_csv(save_path, index=False)
        logger.info(f"✅ PhishTank : {len(df):,} URLs phishing")
        return df

    except Exception as e:
        logger.warning(f"⚠️  PhishTank inaccessible ({e}). Utilisation du fallback.")
        return _phishtank_fallback(save_path)


def _phishtank_fallback(save_path: Path) -> pd.DataFrame:
    """URLs phishing de fallback si PhishTank est down."""
    phishing_urls = [
        "http://paypal-secure-login.xyz/verify",
        "http://amazon-account-update.tk/confirm",
        "http://apple-id-suspended.ml/unlock",
        "http://netflix-billing-update.ga/payment",
        "http://microsoft-security-alert.cf/reset",
        "http://bankofamerica-secure.gq/login",
        "http://facebook-login-verify.xyz/auth",
        "http://google-account-recovery.tk/restore",
        "http://ebay-account-confirm.ml/verify",
        "http://dropbox-secure-link.ga/download",
        "http://instagram-verify-account.cf/check",
        "http://wellsfargo-online.gq/secure",
        "http://chase-bank-login.xyz/signin",
        "http://steam-trade-offer.tk/accept",
        "http://outlook-email-verify.ml/confirm",
    ] * 3000

    df = pd.DataFrame({"url": phishing_urls[:10000], "label": 1})
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"✅ Fallback phishing : {len(df):,} URLs")
    return df


def download_alexa(save_path: Path) -> pd.DataFrame:
    """Charge le fichier Tranco/Alexa local (top-1m.csv) ou télécharge depuis le web."""
    logger.info("📥 Chargement des URLs légitimes...")

    try:
        # ✅ FIX : lire le fichier local top-1m.csv s'il existe
        local_top1m = RAW_DIR / "top-1m.csv"
        if local_top1m.exists():
            logger.info(f"📂 Fichier local trouvé : {local_top1m}")
            df = pd.read_csv(local_top1m, header=None, names=["rank", "domain"])
            # Ignorer les lignes invalides (ex: "poc by @gopal_ethical")
            df = df[df["rank"].apply(lambda x: str(x).isdigit())].copy()
            df["url"] = "http://" + df["domain"].astype(str)
            df = df[["url"]].copy()
            df["label"] = 0
            df.to_csv(save_path, index=False)
            logger.info(f"✅ Tranco/Alexa local : {len(df):,} URLs légitimes")
            return df

        # Sinon tenter le téléchargement
        resp = requests.get(ALEXA_URL, timeout=60, stream=True)
        resp.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        with z.open("top-1m.csv") as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])

        df["url"] = "http://" + df["domain"]
        df = df[["url"]].copy()
        df["label"] = 0
        df.to_csv(save_path, index=False)
        logger.info(f"✅ Alexa : {len(df):,} URLs légitimes")
        return df

    except Exception as e:
        logger.warning(f"⚠️  Alexa inaccessible ({e}). Utilisation du fallback.")
        return _alexa_fallback(save_path)


def _alexa_fallback(save_path: Path) -> pd.DataFrame:
    """URLs légitimes de fallback si Alexa est inaccessible."""
    legit_domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com",
        "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
        "reddit.com", "netflix.com", "microsoft.com", "apple.com",
        "github.com", "stackoverflow.com", "yahoo.com", "bing.com",
        "ebay.com", "paypal.com", "dropbox.com", "spotify.com",
        "twitch.tv", "pinterest.com", "tumblr.com", "wordpress.com",
        "blogspot.com", "medium.com", "quora.com", "imgur.com",
        "nytimes.com", "bbc.com", "cnn.com", "reuters.com",
    ]

    urls = []
    for i, domain in enumerate(legit_domains * 400):
        urls.append(f"http://www.{domain}/page{i % 100}")

    df = pd.DataFrame({"url": urls[:15000], "label": 0})
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"✅ Fallback légitimes : {len(df):,} URLs")
    return df


# ─────────────────────────────────────────────────────────────
# COMBINAISON & SPLIT
# ─────────────────────────────────────────────────────────────
def build_combined_dataset(
    phish_df: pd.DataFrame,
    legit_df: pd.DataFrame,
    min_samples: int = 10000,
) -> pd.DataFrame:
    """Combine, équilibre et mélange les deux sources."""
    logger.info("🔀 Construction du dataset combiné...")

    n = min(len(phish_df), len(legit_df), min_samples)
    phish_sample = phish_df.sample(n=n, random_state=42)
    legit_sample = legit_df.sample(n=n, random_state=42)

    df = pd.concat([phish_sample, legit_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 5].reset_index(drop=True)

    logger.info(f"✅ Dataset final : {len(df):,} URLs ({n:,} phishing / {n:,} légitimes)")
    return df


def split_dataset(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Split stratifié : train / val / test."""
    from sklearn.model_selection import train_test_split

    X, y = df["url"], df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=42, stratify=y
    )

    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=42, stratify=y_temp
    )

    logger.info(f"Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")
    return (
        pd.DataFrame({"url": X_train, "label": y_train}),
        pd.DataFrame({"url": X_val, "label": y_val}),
        pd.DataFrame({"url": X_test, "label": y_test}),
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    phish_path = RAW_DIR / "phishtank.csv"
    alexa_path = RAW_DIR / "alexa.csv"

    # ✅ FIX : supprimer le cache alexa.csv pour forcer la relecture de top-1m.csv
    if alexa_path.exists():
        alexa_path.unlink()
        logger.info("🗑️  Cache alexa.csv supprimé, rechargement depuis top-1m.csv")

    phish_df = (
        pd.read_csv(phish_path) if phish_path.exists()
        else download_phishtank(phish_path)
    )
    legit_df = download_alexa(alexa_path)

    min_samples = cfg["data"].get("min_samples", 10000)
    df = build_combined_dataset(phish_df, legit_df, min_samples)
    df.to_parquet(PROCESSED_DIR / "full_dataset.parquet", index=False)

    train_df, val_df, test_df = split_dataset(
        df,
        val_size=cfg["data"]["val_size"],
        test_size=cfg["data"]["test_size"],
    )

    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    logger.info("🎉 Dataset prêt dans data/processed/")


if __name__ == "__main__":
    main()