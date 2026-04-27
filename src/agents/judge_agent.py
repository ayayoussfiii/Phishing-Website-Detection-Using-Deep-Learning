"""
judge_agent.py
--------------
Agent 3 : Judge — synthétise les résultats Scanner + Detective
et rédige un verdict structuré (sans LLM externe).
"""
from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    final_level: str        # safe | suspicious | phishing
    final_score: int        # 0–100
    confidence: str         # High | Medium | Low
    verdict_text: str
    recommendation: str
    grc_category: str       # ISO 27001 / NIST category
    summary_points: list    # bullet points du rapport


class JudgeAgent:
    """
    Agent Judge : fusionne les scores et génère un verdict GRC.
    Score final = 60% modèle IA + 40% analyse Detective.
    """
    name = "Judge"

    # Seuils de décision
    THRESHOLD_SAFE       = 30
    THRESHOLD_SUSPICIOUS = 65

    def run(self, scan, detective) -> JudgeVerdict:
        # ── Score fusionné ─────────────────────────────────────
        ai_score  = scan.pct
        det_score = detective.risk_score
        final     = int(ai_score * 0.60 + det_score * 0.40)
        final     = max(0, min(100, final))

        # ── Niveau final ────────────────────────────────────────
        if final < self.THRESHOLD_SAFE:
            level = "safe"
        elif final < self.THRESHOLD_SUSPICIOUS:
            level = "suspicious"
        else:
            level = "phishing"

        # ── Confiance ───────────────────────────────────────────
        agreement = (scan.level == detective_level(det_score))
        confidence = "High" if agreement and final not in range(25, 70) else "Medium"
        if not agreement and 30 <= final <= 65:
            confidence = "Low"

        # ── Texte du verdict ────────────────────────────────────
        verdicts = {
            "safe": (
                "Cette URL ne présente aucun indicateur de menace connu. "
                "L'analyse BiLSTM et les signaux de réputation confirment "
                "qu'il s'agit d'une ressource légitime."
            ),
            "suspicious": (
                "Des anomalies ont été détectées dans la structure de cette URL. "
                "Certains indicateurs sont ambigus — il est conseillé de procéder "
                "avec prudence avant de visiter ce lien."
            ),
            "phishing": (
                "ALERTE CRITIQUE : Cette URL présente de multiples indicateurs "
                "caractéristiques d'une attaque de phishing confirmée. "
                "Ne cliquez pas sur ce lien et ne saisissez aucune information personnelle."
            ),
        }

        recommendations = {
            "safe":       "✅ Vous pouvez visiter cette URL en toute sécurité.",
            "suspicious": "⚠️  Vérifiez l'URL manuellement avant de la visiter. Évitez de saisir des identifiants.",
            "phishing":   "🚫 Bloquer immédiatement. Signaler à l'équipe sécurité. Ne pas partager ce lien.",
        }

        grc_categories = {
            "safe":       "NIST CSF : IDENTIFY — Aucune menace détectée",
            "suspicious": "NIST CSF : DETECT — Surveillance recommandée (ISO 27001 A.12.6)",
            "phishing":   "NIST CSF : RESPOND — Incident de sécurité (ISO 27001 A.12.6.1 / A.13.2.1)",
        }

        # ── Points de synthèse ──────────────────────────────────
        summary = []
        summary.append(f"Score IA (BiLSTM) : {ai_score}/100")
        summary.append(f"Score réputation (WHOIS) : {det_score}/100")
        summary.append(f"Score fusionné final : {final}/100")
        if detective.domain_age_days > 0:
            summary.append(f"Âge du domaine : {detective.domain_age_days} jours")
        elif detective.domain_age_days == -1:
            summary.append("Âge du domaine : inconnu")
        if detective.is_ip_url:
            summary.append("URL pointe directement vers une IP")
        if detective.has_suspicious_tld:
            summary.append(f"TLD suspect détecté : .{detective.domain.rsplit('.', 1)[-1]}")
        summary += detective.risk_signals

        return JudgeVerdict(
            final_level=level,
            final_score=final,
            confidence=confidence,
            verdict_text=verdicts[level],
            recommendation=recommendations[level],
            grc_category=grc_categories[level],
            summary_points=summary,
        )


def detective_level(score: int) -> str:
    if score < 30:
        return "safe"
    elif score < 65:
        return "suspicious"
    return "phishing"
