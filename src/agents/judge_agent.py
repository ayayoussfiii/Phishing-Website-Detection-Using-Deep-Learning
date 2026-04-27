"""
judge_agent.py
--------------
Agent 3 : Judge — LLM actif avec Chain-of-Thought + outils appelables.
Le LLM prend la décision finale basée sur des données brutes enrichies.
Fallback automatique vers règles locales si Ollama est indisponible.
"""

import json
import ssl
import socket
import urllib.request
import urllib.error
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

OLLAMA_MODEL = "ALIENTELLIGENCE/cybersecuritythreatanalysisv2"

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ─────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────

@dataclass
class RawContext:
    """Données brutes collectées par le Judge avant d'appeler le LLM."""
    url:             str   = ""
    http_status:     int   = -1
    redirect_chain:  list  = field(default_factory=list)
    page_title:      str   = ""
    html_snippet:    str   = ""          # premiers 800 chars du body
    keywords_found:  list  = field(default_factory=list)
    ssl_valid:       bool  = False
    ssl_issuer:      str   = ""
    ssl_expires_in:  int   = -1          # jours restants
    ssl_domain_match:bool  = False
    resolved_ip:     str   = ""
    fetch_error:     str   = ""


@dataclass
class JudgeVerdict:
    final_level:    str
    final_score:    int
    confidence:     str
    verdict_text:   str
    recommendation: str
    grc_category:   str
    reasoning:         str  = ""   # Chain-of-Thought exposé
    knowledge_signals: list = field(default_factory=list)  # Ce que le LLM a reconnu
    summary_points:    list = field(default_factory=list)


# ─────────────────────────────────────────────
#  Outils appelables par le Judge
# ─────────────────────────────────────────────

PHISHING_KEYWORDS = [
    "verify your account", "vérifiez votre compte",
    "enter your password", "entrez votre mot de passe",
    "confirm your identity", "confirmez votre identité",
    "your account has been suspended", "votre compte a été suspendu",
    "update your billing", "mettez à jour votre facturation",
    "click here immediately", "cliquez ici immédiatement",
    "urgent action required", "action urgente requise",
    "limited time offer", "offre limitée",
    "you have been selected", "vous avez été sélectionné",
    "free gift", "cadeau gratuit",
    "login", "connexion", "sign in", "se connecter",
    "paypal", "amazon", "apple", "microsoft", "google",
    "bank", "banque", "credit card", "carte de crédit",
]


def fetch_page_content(url: str, timeout: int = 6) -> RawContext:
    """
    Outil 1 : Récupère le contenu HTTP brut de l'URL.
    Retourne statut, redirections, titre, extrait HTML, mots-clés suspects.
    """
    ctx = RawContext(url=url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (PhishingDetector/2.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctx.http_status = resp.status
            # Chaîne de redirections
            ctx.redirect_chain = [resp.url] if resp.url != url else []
            raw_html = resp.read(4096).decode("utf-8", errors="ignore")

        # Titre
        title_start = raw_html.lower().find("<title>")
        title_end   = raw_html.lower().find("</title>")
        if title_start != -1 and title_end != -1:
            ctx.page_title = raw_html[title_start + 7 : title_end].strip()

        # Extrait lisible (strip tags basique)
        import re
        text_only = re.sub(r"<[^>]+>", " ", raw_html)
        text_only = re.sub(r"\s+", " ", text_only).strip()
        ctx.html_snippet = text_only[:800]

        # Mots-clés suspects
        lower_text = text_only.lower()
        ctx.keywords_found = [kw for kw in PHISHING_KEYWORDS if kw in lower_text]

    except urllib.error.HTTPError as e:
        ctx.http_status = e.code
        ctx.fetch_error = f"HTTP {e.code}"
    except Exception as e:
        ctx.fetch_error = str(e)

    return ctx


def check_ssl(hostname: str, port: int = 443, timeout: int = 5) -> dict:
    """
    Outil 2 : Vérifie le certificat SSL du domaine.
    Retourne validité, émetteur, expiration, correspondance domaine.
    """
    result = {
        "valid": False,
        "issuer": "",
        "expires_in_days": -1,
        "domain_match": False,
        "error": "",
    }
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        # Émetteur
        issuer_dict = dict(x[0] for x in cert.get("issuer", []))
        result["issuer"] = issuer_dict.get("organizationName", "Unknown")

        # Expiration
        not_after = datetime.strptime(
            cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=timezone.utc)
        result["expires_in_days"] = (not_after - datetime.now(timezone.utc)).days

        # Correspondance domaine
        sans = cert.get("subjectAltName", [])
        san_hosts = [v for k, v in sans if k == "DNS"]
        result["domain_match"] = any(
            hostname.endswith(s.lstrip("*")) for s in san_hosts
        )
        result["valid"] = True

    except ssl.SSLCertVerificationError as e:
        result["error"] = f"SSL invalide : {e}"
    except Exception as e:
        result["error"] = str(e)

    return result


def resolve_dns(hostname: str) -> dict:
    """
    Outil 3 : Résout le DNS et retourne l'IP + détection IP directe.
    """
    result = {"ip": "", "is_direct_ip": False, "error": ""}
    try:
        # Vérifie si c'est déjà une IP
        socket.inet_aton(hostname)
        result["ip"] = hostname
        result["is_direct_ip"] = True
    except socket.error:
        pass

    if not result["is_direct_ip"]:
        try:
            result["ip"] = socket.gethostbyname(hostname)
        except Exception as e:
            result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
#  Agent Judge
# ─────────────────────────────────────────────

class JudgeAgent:
    name  = "Judge"
    # Poids suggérés — le LLM peut les ignorer selon le contexte
    W_AI  = 0.60
    W_DET = 0.40

    def run(self, scan, detective) -> JudgeVerdict:
        # 1. Score fusionné suggéré (hint pour le LLM, pas décision finale)
        fused_hint = int(scan.pct * self.W_AI + detective.risk_score * self.W_DET)
        fused_hint = max(0, min(100, fused_hint))

        if fused_hint < 30:   level_hint = "safe"
        elif fused_hint < 65: level_hint = "suspicious"
        else:                  level_hint = "phishing"

        # 2. Collecte de données brutes (outils)
        raw = self._collect_raw_context(detective.domain, scan.url if hasattr(scan, "url") else "")

        if OLLAMA_AVAILABLE:
            try:
                return self._llm_verdict(scan, detective, fused_hint, level_hint, raw)
            except Exception as e:
                print(f"[Judge] Ollama indisponible ({e}), fallback règles.")

        return self._rule_based_verdict(scan, detective, fused_hint, level_hint, raw)

    # ── Collecte des données brutes ──────────────────────────────
    def _collect_raw_context(self, domain: str, url: str) -> RawContext:
        """Appelle les trois outils et fusionne les résultats dans RawContext."""
        target_url = url if url.startswith("http") else f"https://{domain}"
        raw = fetch_page_content(target_url)

        ssl_info = check_ssl(domain)
        raw.ssl_valid        = ssl_info["valid"]
        raw.ssl_issuer       = ssl_info["issuer"]
        raw.ssl_expires_in   = ssl_info["expires_in_days"]
        raw.ssl_domain_match = ssl_info["domain_match"]

        dns_info = resolve_dns(domain)
        raw.resolved_ip = dns_info["ip"]

        return raw

    # ── LLM actif avec Chain-of-Thought ─────────────────────────
    def _llm_verdict(self, scan, detective, fused_hint, level_hint, raw: RawContext) -> JudgeVerdict:
        prompt = f"""Tu es un expert en cybersécurité et threat intelligence avec une connaissance approfondie de :
- Les techniques de phishing, typosquatting, homoglyphes, et ingénierie sociale
- Les TLDs et hébergeurs bulletproof connus pour le phishing (.tk, .ml, .gq, .cf, .ga, etc.)
- Les patterns d'URLs malveillantes (sous-domaines trompeurs, longs chemins d'obfuscation, etc.)
- Les campagnes de phishing connues (PayPal, Amazon, Microsoft, banques, etc.)
- Les indicateurs SSL frauduleux (Let's Encrypt abusif, cert récent sur domaine jeune, etc.)
- Les bases de réputation (patterns connus de fast-flux DNS, IP de datacenter douteux, etc.)

════════════════════════════════════════
ÉTAPE 1 — ANALYSE PAR TES PROPRES CONNAISSANCES
════════════════════════════════════════
Domaine à analyser : {detective.domain}
Extrait du contenu de la page :
{raw.html_snippet[:500] if raw.html_snippet else "Non disponible"}
Titre de la page : {raw.page_title or "Non disponible"}

Sans regarder les scores automatiques, réponds à ces questions avec tes connaissances :
- Ce domaine ressemble-t-il à un domaine légitime connu ? (typosquatting, homoglyphe ?)
- Le TLD est-il associé à des campagnes de phishing dans ta base de connaissances ?
- Le contenu de la page correspond-il à une technique d'ingénierie sociale connue ?
- Y a-t-il des patterns d'URL suspects que tu reconnais ?

════════════════════════════════════════
ÉTAPE 2 — DONNÉES TECHNIQUES OBSERVÉES
════════════════════════════════════════
- Statut HTTP        : {raw.http_status}
- Redirections       : {raw.redirect_chain if raw.redirect_chain else "Aucune"}
- Mots-clés suspects : {raw.keywords_found if raw.keywords_found else "Aucun"}
- SSL valide         : {raw.ssl_valid}
- SSL émetteur       : {raw.ssl_issuer or "Inconnu"}
- SSL expire dans    : {raw.ssl_expires_in} jours
- SSL domaine match  : {raw.ssl_domain_match}
- IP résolue         : {raw.resolved_ip or "Non résolue"}
- URL → IP directe   : {detective.is_ip_url}
- Âge domaine        : {detective.domain_age_days} jours
- TLD suspect        : {detective.has_suspicious_tld}
- Signaux WHOIS      : {detective.risk_signals}

════════════════════════════════════════
ÉTAPE 3 — SCORES AUTOMATIQUES (confirmation uniquement)
════════════════════════════════════════
Ces scores sont fournis APRÈS ton analyse pour ne pas biaiser ton jugement.
Utilise-les uniquement pour confirmer ou nuancer ta conclusion.
- Score BiLSTM   : {scan.pct}/100  (niveau : {scan.level})
- Score WHOIS    : {detective.risk_score}/100
- Score fusionné : {fused_hint}/100  (niveau suggéré : {level_hint})

Si ton analyse experte diverge du score suggéré, fais confiance à ton expertise.

════════════════════════════════════════
ÉTAPE 4 — VERDICT FINAL
════════════════════════════════════════
Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après.

{{
  "reasoning": "Raisonnement en 4-6 phrases : ce que TES CONNAISSANCES t'ont révélé sur ce domaine/contenu, puis comment les données techniques ont confirmé ou infirmé ton analyse.",
  "knowledge_signals": ["Ce que tu as reconnu grâce à tes connaissances : pattern X, campagne Y, technique Z..."],
  "final_level": "safe | suspicious | phishing",
  "final_score": <entier 0-100, basé sur TON expertise d'abord>,
  "score_adjusted": <true si tu diverges du score suggéré {fused_hint}>,
  "adjustment_reason": "Pourquoi ton expertise te donne un score différent (ou confirme le score automatique).",
  "confidence": "High | Medium | Low",
  "verdict_text": "Verdict clair en français (2-3 phrases) basé sur ton expertise.",
  "recommendation": "Action concrète en français.",
  "grc_category": "NIST CSF fonction + contrôle ISO 27001 applicable",
  "summary_points": ["Signal 1", "Signal 2", "Signal 3"]
}}"""

        resp = _ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.15, "num_predict": 900},
        )
        raw_text = resp["message"]["content"].strip()
        raw_text = self._extract_json(raw_text)
        data = json.loads(raw_text)

        # Validation du level
        level = data.get("final_level", level_hint)
        if level not in ("safe", "suspicious", "phishing"):
            level = level_hint

        # Score final : on respecte la décision du LLM
        final_score = int(data.get("final_score", fused_hint))
        final_score = max(0, min(100, final_score))

        return JudgeVerdict(
            final_level       = level,
            final_score       = final_score,
            confidence        = data.get("confidence", "Medium"),
            verdict_text      = data.get("verdict_text", ""),
            recommendation    = data.get("recommendation", ""),
            grc_category      = data.get("grc_category", "NIST CSF"),
            reasoning         = data.get("reasoning", ""),
            knowledge_signals = data.get("knowledge_signals", []),
            summary_points    = self._build_summary(data, raw, fused_hint, final_score),
        )

    def _build_summary(self, data: dict, raw: RawContext, fused_hint: int, final_score: int) -> list:
        """Construit les summary_points enrichis avec les données brutes."""
        points = data.get("summary_points", [])

        # Ajouts automatiques basés sur les données brutes
        if raw.keywords_found:
            points.append(f"🔴 {len(raw.keywords_found)} mot(s)-clé(s) phishing détecté(s) dans la page")
        if not raw.ssl_valid:
            points.append("🔴 Certificat SSL invalide ou absent")
        elif raw.ssl_expires_in < 15:
            points.append(f"🟠 SSL expire dans {raw.ssl_expires_in} jours")
        if raw.redirect_chain:
            points.append(f"🟠 {len(raw.redirect_chain)} redirection(s) détectée(s)")
        if raw.http_status not in (-1, 200, 301, 302):
            points.append(f"🟠 Statut HTTP anormal : {raw.http_status}")
        if data.get("score_adjusted"):
            points.append(
                f"⚖️  Score ajusté par le LLM : {fused_hint} → {final_score} "
                f"({data.get('adjustment_reason', '')})"
            )
        return points

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end]
        return text

    # ── Fallback règles locales ──────────────────────────────────
    def _rule_based_verdict(self, scan, detective, fused, level, raw: RawContext) -> JudgeVerdict:
        # Ajustement du score selon les données brutes
        bonus = 0
        if raw.keywords_found:
            bonus += min(len(raw.keywords_found) * 4, 20)
        if not raw.ssl_valid and not raw.fetch_error:
            bonus += 15
        if raw.redirect_chain:
            bonus += 8
        if raw.http_status not in (-1, 200, 201, 301, 302):
            bonus += 5

        adjusted = max(0, min(100, fused + bonus))

        if adjusted < 30:   level = "safe"
        elif adjusted < 65: level = "suspicious"
        else:               level = "phishing"

        verdicts = {
            "safe":       "Cette URL ne présente aucun indicateur de menace. L'analyse BiLSTM, WHOIS et les données brutes confirment une ressource légitime.",
            "suspicious": "Des anomalies ont été détectées. Certains indicateurs sont ambigus — procédez avec prudence.",
            "phishing":   "ALERTE : Cette URL présente de multiples indicateurs de phishing confirmé. Ne cliquez pas.",
        }
        recommendations = {
            "safe":       "✅ URL sécurisée, vous pouvez la visiter.",
            "suspicious": "⚠️  Vérifiez manuellement avant de visiter.",
            "phishing":   "🚫 Bloquer immédiatement. Signaler à l'équipe sécurité.",
        }
        grc_categories = {
            "safe":       "NIST CSF : IDENTIFY — Aucune menace détectée",
            "suspicious": "NIST CSF : DETECT — Surveillance requise (ISO 27001 A.12.6)",
            "phishing":   "NIST CSF : RESPOND — Incident actif (ISO 27001 A.12.6.1 / A.16.1)",
        }

        summary = [
            f"Score BiLSTM      : {scan.pct}/100",
            f"Score WHOIS       : {detective.risk_score}/100",
            f"Score fusionné    : {fused}/100",
        ]
        if bonus > 0:
            summary.append(f"⚠️  Bonus risque données brutes : +{bonus} → score ajusté {adjusted}/100")
        if detective.domain_age_days >= 0:
            summary.append(f"Âge domaine       : {detective.domain_age_days} jours")
        if detective.is_ip_url:
            summary.append("🔴 URL pointe vers une IP directe")
        if detective.has_suspicious_tld:
            summary.append("🔴 TLD suspect détecté")
        if raw.keywords_found:
            summary.append(f"🔴 Mots-clés phishing : {', '.join(raw.keywords_found[:5])}")
        if not raw.ssl_valid and not raw.fetch_error:
            summary.append("🔴 SSL invalide ou absent")
        if raw.redirect_chain:
            summary.append(f"🟠 Redirections : {raw.redirect_chain}")
        summary += detective.risk_signals

        confidence = "High" if scan.level == level and adjusted not in range(25, 70) else "Medium"

        return JudgeVerdict(
            final_level       = level,
            final_score       = adjusted,
            confidence        = confidence,
            verdict_text      = verdicts[level],
            recommendation    = recommendations[level],
            grc_category      = grc_categories[level],
            reasoning         = "Mode fallback — règles locales avec enrichissement données brutes.",
            knowledge_signals = [],
            summary_points    = summary,
        )
