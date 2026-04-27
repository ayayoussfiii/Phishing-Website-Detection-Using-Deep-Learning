
import sys, io, requests, zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

PHISHTANK_URL  = "http://data.phishtank.com/data/online-valid.csv"
TRANCO_URL     = "https://tranco-list.eu/download/latest/full"
TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"


def download_phishtank(save_path):
    logger.info("Telecharge PhishTank...")
    try:
        resp = requests.get(PHISHTANK_URL, headers={"User-Agent": "research/1.0"}, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), usecols=["url"])
        df["label"] = 1
        df.to_csv(save_path, index=False)
        logger.info(f"PhishTank : {len(df):,} URLs phishing")
        return df
    except Exception as e:
        logger.warning(f"PhishTank inaccessible ({e}). Fallback...")
        return _phishtank_fallback(save_path)


def _phishtank_fallback(save_path):
    patterns = [
        "http://paypal-secure-login.{tld}/verify/{uid}",
        "http://amazon-account-update.{tld}/confirm/{uid}",
        "http://apple-id-suspended.{tld}/unlock/{uid}",
        "http://netflix-billing-update.{tld}/payment/{uid}",
        "http://microsoft-security-alert.{tld}/reset/{uid}",
        "http://bankofamerica-secure.{tld}/login/{uid}",
        "http://facebook-login-verify.{tld}/auth/{uid}",
        "http://google-account-recovery.{tld}/restore/{uid}",
        "http://{uid}.ebay-account-confirm.{tld}/verify",
        "http://secure-{uid}.instagram-verify.{tld}/check",
        "http://192.168.{uid}.1/phishing/login",
        "http://paypal.com.{tld}/webscr?cmd=login&uid={uid}",
    ]
    tlds = ["xyz", "tk", "ml", "ga", "cf", "gq", "top", "club"]
    urls = [patterns[i % len(patterns)].format(tld=tlds[i % len(tlds)], uid=i) for i in range(60000)]
    df = pd.DataFrame({"url": urls, "label": 1}).drop_duplicates().reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Fallback phishing : {len(df):,} URLs")
    return df


def download_tranco(save_path):
    logger.info("Telecharge Tranco Top 1M (URLs legitimes)...")
    for url, is_zip in [(TRANCO_URL, False), (TRANCO_ZIP_URL, True)]:
        try:
            resp = requests.get(url, timeout=90, stream=True)
            resp.raise_for_status()
            if is_zip:
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                with z.open("top-1m.csv") as f:
                    df = pd.read_csv(f, header=None, names=["rank", "domain"])
            else:
                df = pd.read_csv(io.StringIO(resp.text), header=None, names=["rank", "domain"])
            df = df[df["rank"].apply(lambda x: str(x).isdigit())].copy()
            df["url"] = "https://www." + df["domain"].astype(str)
            df = df[["url"]].copy()
            df["label"] = 0
            df.to_csv(save_path, index=False)
            logger.info(f"Tranco : {len(df):,} URLs legitimes")
            return df
        except Exception as e:
            logger.warning(f"Tranco inaccessible ({e})...")
    return _legit_fallback(save_path)


def _legit_fallback(save_path):
    logger.info("Generation fallback legitimes enrichi (50000+ URLs)...")
    domains = [
        "google.com","youtube.com","microsoft.com","apple.com","github.com",
        "stackoverflow.com","mozilla.org","nodejs.org","reactjs.org","docker.com",
        "facebook.com","twitter.com","instagram.com","linkedin.com","reddit.com",
        "discord.com","slack.com","telegram.org","medium.com","quora.com",
        "amazon.com","ebay.com","etsy.com","shopify.com","walmart.com",
        "aliexpress.com","bestbuy.com","target.com","newegg.com","wayfair.com",
        "bbc.com","cnn.com","reuters.com","nytimes.com","theguardian.com",
        "bloomberg.com","forbes.com","techcrunch.com","wired.com","theverge.com",
        "paypal.com","stripe.com","wise.com","coinbase.com","binance.com",
        "nasa.gov","nih.gov","cdc.gov","harvard.edu","mit.edu",
        "netflix.com","spotify.com","twitch.tv","hulu.com","imdb.com",
        "cloudflare.com","digitalocean.com","heroku.com","vercel.com","netlify.com",
        "wikipedia.org","archive.org","coursera.org","edx.org","khanacademy.org",
        "dropbox.com","zoom.us","atlassian.com","notion.so","figma.com",
    ]
    paths = ["/","/about","/contact","/login","/signup","/help","/faq","/blog",
             "/news","/products","/services","/pricing","/docs","/api","/download",
             "/support","/terms","/privacy","/search","/account","/profile",
             "/settings","/dashboard","/explore","/page/1","/page/2","/page/3"]
    subs = ["www","blog","help","support","docs","api","shop","news","dev"]
    urls = set()
    for d in domains:
        for p in paths:
            urls.add(f"https://www.{d}{p}")
        for s in subs:
            urls.add(f"https://{s}.{d}/")
    i = 0
    while len(urls) < 60000:
        urls.add(f"https://www.{domains[i % len(domains)]}/item/{i}")
        i += 1
    df = pd.DataFrame({"url": list(urls), "label": 0}).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Fallback legitimes : {len(df):,} URLs")
    return df


def build_combined_dataset(phish_df, legit_df, min_samples=50000):
    logger.info("Construction du dataset combine...")
    phish_df = phish_df[phish_df["url"].astype(str).str.len() > 5].copy()
    legit_df = legit_df[legit_df["url"].astype(str).str.len() > 5].copy()
    n = min(len(phish_df), len(legit_df), min_samples)
    logger.info(f"Phishing : {len(phish_df):,} | Legitimes : {len(legit_df):,} | Utilises : {n:,} chacun")
    df = pd.concat([phish_df.sample(n=n, random_state=42),
                    legit_df.sample(n=n, random_state=42)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["url"] = df["url"].astype(str).str.strip()
    logger.info(f"Dataset final : {len(df):,} URLs equilibrees")
    return df


def split_dataset(df, val_size=0.15, test_size=0.15):
    X, y = df["url"], df["label"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=val_size+test_size, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=test_size/(val_size+test_size), random_state=42, stratify=y_tmp)
    logger.info(f"Train : {len(X_tr):,} | Val : {len(X_val):,} | Test : {len(X_te):,}")
    return (pd.DataFrame({"url": X_tr,  "label": y_tr}),
            pd.DataFrame({"url": X_val, "label": y_val}),
            pd.DataFrame({"url": X_te,  "label": y_te}))


def main():
    cfg = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    phish_path = RAW_DIR / "phishtank.csv"
    legit_path = RAW_DIR / "tranco.csv"
    phish_df = pd.read_csv(phish_path) if phish_path.exists() else download_phishtank(phish_path)
    legit_df = download_tranco(legit_path)
    df = build_combined_dataset(phish_df, legit_df, cfg["data"].get("min_samples", 50000))
    df.to_parquet(PROCESSED_DIR / "full_dataset.parquet", index=False)
    train_df, val_df, test_df = split_dataset(df, cfg["data"]["val_size"], cfg["data"]["test_size"])
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet",   index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet",  index=False)
    logger.info("Dataset pret dans data/processed/")

if __name__ == "__main__":
    main()