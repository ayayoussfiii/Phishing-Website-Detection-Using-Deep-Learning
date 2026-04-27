"""
orchestrator.py
---------------
Orchestrateur : coordonne les 3 agents (Scanner → Detective → Judge)
et produit le rapport final complet.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnalysisReport:
    url: str
    timestamp: str
    duration_ms: int
    scan: dict
    detective: dict
    verdict: dict
    export_json: str = ""


class Orchestrator:
    """
    Orchestre les 3 agents dans l'ordre :
    ScannerAgent → DetectiveAgent → JudgeAgent
    """
    def __init__(self, scanner, detective, judge):
        self.scanner   = scanner
        self.detective = detective
        self.judge     = judge

    def analyze(self, url: str) -> AnalysisReport:
        t0 = time.time()
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # ── Étape 1 : Scanner IA ────────────────────────────────
        scan_result = self.scanner.run(url)

        # ── Étape 2 : Detective WHOIS ───────────────────────────
        det_result  = self.detective.run(url)

        # ── Étape 3 : Judge synthèse ────────────────────────────
        verdict     = self.judge.run(scan_result, det_result)

        duration_ms = int((time.time() - t0) * 1000)

        # ── Sérialisation ────────────────────────────────────────
        scan_dict = {
            "url":      scan_result.url,
            "prob":     round(scan_result.prob, 4),
            "pct":      scan_result.pct,
            "level":    scan_result.level,
            "features": scan_result.features,
        }

        det_dict = {
            "domain":             det_result.domain,
            "ip_address":         det_result.ip_address,
            "domain_age_days":    det_result.domain_age_days,
            "registrar":          det_result.registrar,
            "country":            det_result.country,
            "is_ip_url":          det_result.is_ip_url,
            "has_suspicious_tld": det_result.has_suspicious_tld,
            "subdomain_depth":    det_result.subdomain_depth,
            "url_length":         det_result.url_length,
            "risk_signals":       det_result.risk_signals,
            "risk_score":         det_result.risk_score,
        }

        verdict_dict = {
            "final_level":       verdict.final_level,
            "final_score":       verdict.final_score,
            "confidence":        verdict.confidence,
            "verdict_text":      verdict.verdict_text,
            "recommendation":    verdict.recommendation,
            "grc_category":      verdict.grc_category,
            "summary_points":    verdict.summary_points,
            "reasoning":         verdict.reasoning,           # ← corrigé
            "knowledge_signals": verdict.knowledge_signals,   # ← corrigé
        }

        export = {
            "url":         url,
            "timestamp":   ts,
            "duration_ms": duration_ms,
            "scan":        scan_dict,
            "detective":   det_dict,
            "verdict":     verdict_dict,
        }

        report = AnalysisReport(
            url=url,
            timestamp=ts,
            duration_ms=duration_ms,
            scan=scan_dict,
            detective=det_dict,
            verdict=verdict_dict,
            export_json=json.dumps(export, ensure_ascii=False, indent=2),
        )

        return report
