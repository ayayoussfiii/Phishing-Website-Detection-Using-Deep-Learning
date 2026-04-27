"""
detective_agent.py
------------------
Agent 2 : Detective — analyse WHOIS, structure de l'URL, réputation.
"""
import re
import socket
from datetime import datetime, timezone
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False


@dataclass
class DetectiveResult:
    domain: str
    ip_address: str
    domain_age_days: int        # -1 = inconnu
    registrar: str
    country: str
    is_ip_url: bool
    has_suspicious_tld: bool
    subdomain_depth: int
    url_length: int
    has_at_symbol: bool
    has_double_slash: bool
    redirect_count: int
    risk_signals: list = field(default_factory=list)
    risk_score: int = 0         # 0–100


SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "top", "club",
    "online", "site", "info", "biz", "work", "loan",
    "click", "link", "zip", "download",
}


class DetectiveAgent:
    """
    Agent Detective : collecte les signaux de réputation via WHOIS + heuristiques.
    """
    name = "Detective"

    def run(self, url: str) -> DetectiveResult:
        parsed   = urlparse(url if url.startswith("http") else "http://" + url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""

        # ── IP url ? ────────────────────────────────────────────
        is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))

        # ── TLD suspect ? ───────────────────────────────────────
        tld = hostname.rsplit(".", 1)[-1].lower() if "." in hostname else ""
        bad_tld = tld in SUSPICIOUS_TLDS

        # ── Profondeur sous-domaines ─────────────────────────────
        parts          = hostname.split(".")
        subdomain_depth = max(0, len(parts) - 2)

        # ── Symboles suspects ────────────────────────────────────
        has_at     = "@" in url
        has_dslash = "//" in path

        # ── IP résolution ────────────────────────────────────────
        ip_addr = ""
        try:
            ip_addr = socket.gethostbyname(hostname)
        except Exception:
            pass

        # ── WHOIS ────────────────────────────────────────────────
        domain_age = -1
        registrar  = "Unknown"
        country    = "Unknown"

        if WHOIS_AVAILABLE and not is_ip and hostname:
            try:
                domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
                w      = whois.whois(domain)
                registrar = str(w.registrar or "Unknown")[:80]
                country   = str(w.country   or "Unknown")[:10]
                created   = w.creation_date
                if isinstance(created, list):
                    created = created[0]
                if created:
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    domain_age = (datetime.now(timezone.utc) - created).days
            except Exception:
                pass

        # ── Calcul score de risque ────────────────────────────────
        signals = []
        score   = 0

        if is_ip:
            signals.append("⚠️  URL pointe vers une IP directe (pas de domaine)")
            score += 25
        if bad_tld:
            signals.append(f"⚠️  TLD suspect : .{tld}")
            score += 20
        if subdomain_depth >= 3:
            signals.append(f"⚠️  Trop de sous-domaines ({subdomain_depth})")
            score += 15
        if has_at:
            signals.append("⚠️  Symbole @ détecté dans l'URL")
            score += 20
        if has_dslash:
            signals.append("⚠️  Double slash dans le path")
            score += 10
        if 0 < domain_age < 30:
            signals.append(f"⚠️  Domaine très récent ({domain_age} jours)")
            score += 25
        elif domain_age == -1 and not is_ip:
            signals.append("ℹ️  Age du domaine inconnu (WHOIS indisponible)")
        if len(url) > 100:
            signals.append(f"ℹ️  URL longue ({len(url)} caractères)")
            score += 5

        if not signals:
            signals.append("✅ Aucun signal de risque détecté")

        return DetectiveResult(
            domain=hostname,
            ip_address=ip_addr,
            domain_age_days=domain_age,
            registrar=registrar,
            country=country,
            is_ip_url=is_ip,
            has_suspicious_tld=bad_tld,
            subdomain_depth=subdomain_depth,
            url_length=len(url),
            has_at_symbol=has_at,
            has_double_slash=has_dslash,
            redirect_count=0,
            risk_signals=signals,
            risk_score=min(score, 100),
        )
