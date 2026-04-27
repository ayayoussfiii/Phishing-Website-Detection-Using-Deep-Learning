import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Django imports
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.urls import path
from django.template import Template, Context
from django.views.decorators.csrf import csrf_exempt

# ==========================================
# 1. CONFIGURATION DJANGO
# ==========================================
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='ensa-fes-secret-key',
        ROOT_URLCONF=__name__,
        STATIC_URL='/static/',
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
        }],
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.staticfiles',
        ],
    )

# ==========================================
# 2. CHARGEMENT DES AGENTS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

orchestrator = None
load_error = None

try:
    import tensorflow as tf
    from src.features.feature_pipeline import FeaturePipeline
    from src.features.url_features import extract_url_features, FEATURE_NAMES
    from src.models.hybrid_model import SelfAttention
    from src.agents.scanner_agent import ScannerAgent
    from src.agents.detective_agent import DetectiveAgent
    from src.agents.judge_agent import JudgeAgent
    from src.agents.orchestrator import Orchestrator

    MODELS_DIR = PROJECT_ROOT / "models"
    PIPELINE_DIR = MODELS_DIR / "pipeline"

    model_path = MODELS_DIR / "hybrid_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"SelfAttention": SelfAttention}
    )
    pipeline = FeaturePipeline.load(PIPELINE_DIR)

    # Initialisation des 3 agents
    scanner_agent = ScannerAgent(model, pipeline, FEATURE_NAMES, extract_url_features)
    detective_agent = DetectiveAgent()
    judge_agent = JudgeAgent()

    # Orchestrateur qui coordonne les 3 agents
    orchestrator = Orchestrator(scanner_agent, detective_agent, judge_agent)

    print("=" * 50)
    print("✅ Agents chargés avec succès !")
    print(f"   📍 ScannerAgent: BiLSTM + {len(FEATURE_NAMES)} features")
    print(f"   🔍 DetectiveAgent: WHOIS + structure URL")
    print(f"   ⚖️ JudgeAgent: Fusion + verdict GRC")
    print(f"   🎮 Orchestrator: Prêt")
    print("=" * 50)

except Exception as e:
    load_error = str(e)
    print(f"❌ Erreur chargement agents: {e}")


# ==========================================
# 3. FONCTIONS UTILITAIRES
# ==========================================
def generate_report_id(url):
    return hashlib.md5(f"{url}{datetime.now().timestamp()}".encode()).hexdigest()[:8]


def get_clean_report(url):
    """Utilise les 3 agents (Scanner + Detective + Judge) pour produire le rapport."""

    if orchestrator is None:
        return _fallback_report(url, error=load_error)

    try:
        # Appel à l'orchestrateur qui coordonne les 3 agents
        report = orchestrator.analyze(url)

        # Extraction des résultats
        scan_data = report.scan
        det_data = report.detective
        verdict_data = report.verdict

        risk_score = verdict_data.get("final_score", 50)
        confidence_score = 100 - risk_score
        level = verdict_data.get("final_level", "suspicious")

        # Mapping niveau → couleurs/textes
        level_map = {
            "safe": {
                "status": "Safe",
                "status_icon": "🔒",
                "status_color": "#22c55e",
                "status_gradient": "linear-gradient(135deg, #22c55e20, #16a34a10)",
                "status_title": "Site Sécurisé",
                "threat_text": "Faible",
                "threat_color": "#22c55e",
                "threat_description": "Aucune menace détectée",
                "compliance_status": "Compliant",
            },
            "suspicious": {
                "status": "Suspicious",
                "status_icon": "⚠️",
                "status_color": "#f59e0b",
                "status_gradient": "linear-gradient(135deg, #f59e0b20, #d9770610)",
                "status_title": "Site Suspect",
                "threat_text": "Moyen",
                "threat_color": "#f59e0b",
                "threat_description": "Anomalies détectées, prudence recommandée",
                "compliance_status": "Non-Compliant - Requires Review",
            },
            "phishing": {
                "status": "Malicious",
                "status_icon": "🔴",
                "status_color": "#ef4444",
                "status_gradient": "linear-gradient(135deg, #ef444420, #dc262610)",
                "status_title": "Site Malveillant",
                "threat_text": "Critique",
                "threat_color": "#ef4444",
                "threat_description": "Phishing confirmé, blocage immédiat requis",
                "compliance_status": "Non-Compliant - Blocked",
            },
        }
        lm = level_map.get(level, level_map["suspicious"])

        # Transformation des signaux pour l'affichage
        signals = []
        detective_signals = det_data.get("risk_signals", [])
        for sig in detective_signals[:10]:
            icon = "⚠️"
            if "ip" in sig.lower():
                icon = "🔢"
            elif "tld" in sig.lower() or "extension" in sig.lower():
                icon = "🌐"
            elif "âge" in sig.lower() or "récent" in sig.lower() or "jours" in sig.lower():
                icon = "⏱️"
            elif "https" in sig.lower() or "certificat" in sig.lower():
                icon = "🔓"
            signals.append({
                "name": sig[:80] if len(sig) > 80 else sig,
                "icon": icon,
                "critical": "⚠️" in sig or "critique" in sig.lower(),
                "detail": sig,
            })

        # Ajout d'un signal du Scanner si score élevé
        scan_score = scan_data.get("pct", 50)
        if scan_score > 70 and len(signals) < 8:
            signals.append({
                "name": "Score IA élevé (BiLSTM)",
                "icon": "🧠",
                "critical": scan_score > 85,
                "detail": f"Le modèle Deep Learning détecte {scan_score}% de risque",
            })

        # Construction du rapport final
        return {
            "report_id": generate_report_id(url),
            "generated_at": report.timestamp,
            "url": url,
            "domain": det_data.get("domain", "N/A"),
            "status": lm["status"],
            "status_icon": lm["status_icon"],
            "status_color": lm["status_color"],
            "status_gradient": lm["status_gradient"],
            "status_title": lm["status_title"],
            "risk_score": risk_score,
            "confidence_score": confidence_score,
            "threat_level": level,
            "threat_text": lm["threat_text"],
            "threat_color": lm["threat_color"],
            "threat_description": lm["threat_description"],
            "age_days": f"{det_data.get('domain_age_days', 'N/A')} jours" if det_data.get('domain_age_days', -1) > 0 else "Inconnu",
            "creation_date": "N/A",
            "expiration_date": "N/A",
            "registrar": det_data.get("registrar", "N/A"),
            "country": det_data.get("country", "N/A"),
            "signals": signals,
            "verdict": verdict_data.get("verdict_text", "Analyse terminée"),
            "recommendation": verdict_data.get("recommendation", "Vérifiez l'URL avant de continuer"),
            "deep_score": scan_score,
            "url_length": det_data.get("url_length", len(url)),
            "dash_count": url.count("-"),
            "dot_count": url.count("."),
            "slash_count": url.count("/"),
            "https": url.startswith("https"),
            "compliance_status": lm["compliance_status"],
            "grc_category": verdict_data.get("grc_category", "Cybersecurity Assessment"),
            "summary_points": verdict_data.get("summary_points", []),
            "confidence": verdict_data.get("confidence", "Medium"),
            "duration_ms": report.duration_ms,
        }

    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return _fallback_report(url, error=str(e))


def _fallback_report(url, error=None):
    """Rapport de secours si les agents ne sont pas disponibles."""
    suspicious_words = ['secure', 'login', 'bank', 'verify', 'confirm', 'account', 'update']
    score = sum(1 for w in suspicious_words if w in url.lower())
    risk = min(95, score * 15 + 5)
    domain = url.split("//")[-1].split("/")[0].split(":")[0]

    status_color = "#f59e0b" if risk > 30 else "#22c55e"
    status_icon = "⚠️" if risk > 30 else "🔒"

    return {
        "report_id": generate_report_id(url),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": url,
        "domain": domain,
        "status": "Suspicious" if risk > 30 else "Safe",
        "status_icon": status_icon,
        "status_color": status_color,
        "status_gradient": "linear-gradient(135deg, #f59e0b20, #d9770610)",
        "status_title": "Mode secours",
        "risk_score": risk,
        "confidence_score": 100 - risk,
        "threat_level": "suspicious" if risk > 30 else "safe",
        "threat_text": "Moyen" if risk > 30 else "Faible",
        "threat_color": status_color,
        "threat_description": f"Mode secours actif. {error or 'Agents non disponibles'}",
        "age_days": "Inconnu",
        "creation_date": "N/A",
        "expiration_date": "N/A",
        "registrar": "N/A",
        "country": "N/A",
        "signals": [{"name": "Mode secours actif", "icon": "⚠️", "critical": True, "detail": error or "Agents non chargés"}],
        "verdict": "Analyse heuristique uniquement — agents IA non disponibles.",
        "recommendation": "Vérifiez que le modèle est entraîné (hybrid_best.keras)",
        "deep_score": risk,
        "url_length": len(url),
        "dash_count": url.count("-"),
        "dot_count": url.count("."),
        "slash_count": url.count("/"),
        "https": url.startswith("https"),
        "compliance_status": "Non-Compliant",
        "grc_category": "Fallback Analysis",
        "summary_points": ["⚠️ Mode secours activé", f"📊 Score: {risk}%", f"🌐 Domaine: {domain}"],
        "confidence": "Low",
        "duration_ms": 0,
    }


# ==========================================
# 4. HTML TEMPLATE (version corrigée sans filtre abs)
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhishGuard Pro | Multi-Agent Anti-Phishing</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #0a0c10; --bg-secondary: #0f1218; --bg-tertiary: #151b26;
            --bg-glass: rgba(15, 18, 24, 0.95); --text-primary: #ffffff;
            --text-secondary: #a0aec0; --text-tertiary: #64748b;
            --border-color: rgba(51, 65, 85, 0.5); --glow-blue: #3b82f6;
        }
        body { font-family: 'Inter', sans-serif; background: var(--bg-primary); color: var(--text-primary); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .glass-card { background: var(--bg-glass); backdrop-filter: blur(20px); border-radius: 32px; border: 1px solid var(--border-color); overflow: hidden; }
        .header-cyber { padding: 50px 60px 30px; text-align: center; background: linear-gradient(180deg, rgba(59,130,246,0.1) 0%, transparent 100%); }
        .cyber-shield { font-size: 70px; animation: pulse 2s ease infinite; }
        @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        .cyber-title { font-size: 56px; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .agents-strip { display: grid; grid-template-columns: repeat(3,1fr); border-bottom: 1px solid var(--border-color); }
        .agent-card { padding: 14px 20px; border-right: 1px solid var(--border-color); display: flex; align-items: center; gap: 12px; }
        .agent-card:last-child { border-right: none; }
        .agent-icon { width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
        .agent-icon.scanner { background: rgba(59,130,246,0.2); }
        .agent-icon.detective { background: rgba(245,158,11,0.2); }
        .agent-icon.judge { background: rgba(34,197,94,0.2); }
        .agent-name { font-size: 12px; font-weight: 600; }
        .agent-desc { font-size: 11px; color: var(--text-secondary); }
        .search-cyber { padding: 0 60px 40px; }
        .input-cyber { background: var(--bg-secondary); border-radius: 60px; padding: 8px; display: flex; gap: 12px; border: 2px solid var(--border-color); transition: all 0.3s; }
        .input-cyber:focus-within { border-color: var(--glow-blue); box-shadow: 0 0 0 4px rgba(59,130,246,0.2); }
        .url-input-cyber { flex: 1; background: transparent; border: none; padding: 16px 28px; color: var(--text-primary); font-size: 16px; outline: none; font-family: monospace; }
        .analyze-btn-cyber { background: linear-gradient(135deg, #3b82f6, #8b5cf6); border: none; padding: 14px 36px; border-radius: 50px; color: white; font-weight: 700; cursor: pointer; transition: all 0.3s; }
        .analyze-btn-cyber:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(59,130,246,0.5); }
        .examples-cyber { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; justify-content: center; }
        .example-cyber { background: var(--bg-tertiary); padding: 8px 18px; border-radius: 30px; font-size: 12px; font-family: monospace; cursor: pointer; transition: all 0.2s; border: 1px solid var(--border-color); }
        .example-cyber:hover { background: #3b82f6; color: white; transform: translateY(-3px); }
        .result-cyber { padding: 0 60px 50px; animation: slideIn 0.6s ease; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
        .status-cyber { background: var(--status-gradient); border-radius: 24px; padding: 24px 32px; margin-bottom: 32px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; border: 1px solid var(--status-color); }
        .status-badge-cyber { display: flex; align-items: center; gap: 16px; font-size: 32px; font-weight: 800; color: var(--status-color); }
        .threat-bar { height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden; margin: 20px 0; }
        .threat-fill { height: 100%; background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); transition: width 1s ease; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }
        .stat-card { background: var(--bg-secondary); border-radius: 20px; padding: 20px; border: 1px solid var(--border-color); }
        .signals-list { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }
        .signal { background: rgba(239,68,68,0.1); border-radius: 12px; padding: 10px 18px; display: flex; align-items: center; gap: 10px; border-left: 3px solid #ef4444; }
        .verdict-box { background: var(--status-gradient); border-radius: 24px; padding: 28px; text-align: center; border: 1px solid var(--status-color); margin: 24px 0; }
        .action-grid { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-top: 24px; }
        .action-btn { background: var(--bg-secondary); border: 1px solid var(--border-color); padding: 12px 28px; border-radius: 40px; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; gap: 10px; }
        .action-btn:hover { background: var(--glow-blue); border-color: var(--glow-blue); transform: translateY(-2px); }
        .footer-cyber { padding: 24px 60px; text-align: center; border-top: 1px solid var(--border-color); font-size: 12px; color: var(--text-tertiary); }
        .toast { position: fixed; bottom: 30px; right: 30px; background: #22c55e; color: white; padding: 12px 24px; border-radius: 12px; animation: fadeInOut 3s ease; z-index: 1000; }
        @keyframes fadeInOut { 0% { opacity: 0; transform: translateX(100%); } 10% { opacity: 1; transform: translateX(0); } 90% { opacity: 1; } 100% { opacity: 0; transform: translateX(100%); } }
        @media (max-width: 768px) { .header-cyber { padding: 30px 24px; } .search-cyber { padding: 0 24px 30px; } .result-cyber { padding: 0 24px 36px; } .cyber-title { font-size: 32px; } .agents-strip { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
<div class="glass-card">

    <div class="header-cyber">
        <div class="cyber-shield">🛡️</div>
        <h1 class="cyber-title">PHISHGUARD PRO</h1>
        <p class="cyber-subtitle" style="color: var(--text-secondary); margin-top: 12px;">Multi-Agent Neural Defense System | 3 Agents IA</p>
    </div>

    <!-- Agents strip -->
    <div class="agents-strip">
        <div class="agent-card"><div class="agent-icon scanner">🧠</div><div><div class="agent-name">Agent Scanner</div><div class="agent-desc">BiLSTM · 18 features · Score DL</div></div></div>
        <div class="agent-card"><div class="agent-icon detective">🔍</div><div><div class="agent-name">Agent Detective</div><div class="agent-desc">WHOIS · TLD · Structure URL</div></div></div>
        <div class="agent-card"><div class="agent-icon judge">⚖️</div><div><div class="agent-name">Agent Judge</div><div class="agent-desc">Synthèse · Verdict · GRC</div></div></div>
    </div>

    <div class="search-cyber">
        <form method="POST" id="analyzeForm">
            <div class="input-cyber">
                <input type="text" name="url" class="url-input-cyber" id="urlInput" placeholder="https://exemple.com/login" value="{{ url_value }}" required>
                <button type="submit" class="analyze-btn-cyber"><i class="fas fa-shield-virus"></i> ANALYZE THREAT</button>
            </div>
        </form>
        <div class="examples-cyber">
            <span class="example-cyber" onclick="setUrl('https://www.google.com')"><i class="fab fa-google"></i> google.com</span>
            <span class="example-cyber" onclick="setUrl('https://github.com/tensorflow')"><i class="fab fa-github"></i> github.com</span>
            <span class="example-cyber" onclick="setUrl('http://paypal-secure-login.xyz/verify')"><i class="fas fa-skull"></i> paypal-secure-login.xyz</span>
            <span class="example-cyber" onclick="setUrl('http://amazon-account-update.tk/confirm')"><i class="fas fa-skull"></i> amazon-update.tk</span>
        </div>
    </div>

    {% if report %}
    <div class="result-cyber">
        <div class="status-cyber" style="--status-color: {{ report.status_color }}; --status-gradient: {{ report.status_gradient }}">
            <div class="status-badge-cyber"><span>{{ report.status_icon }}</span><span>{{ report.status_title }}</span></div>
            <div style="display: flex; gap: 16px;">
                <div class="status-stat"><i class="fas fa-microchip"></i> Scanner: {{ report.deep_score }}%</div>
                <div class="status-stat"><i class="fas fa-stopwatch"></i> {{ report.duration_ms }}ms</div>
            </div>
        </div>

        <!-- 3D Risk Circle - Version corrigée sans filtre abs -->
        <div style="text-align: center; margin: 30px 0;">
            <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
                <svg width="200" height="200" viewBox="0 0 200 200">
                    <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(51,65,85,0.5)" stroke-width="12"/>
                    <circle cx="100" cy="100" r="90" fill="none" stroke="{{ report.status_color }}" stroke-width="12" stroke-linecap="round"
                            stroke-dasharray="565" stroke-dashoffset="{{ risk_offset }}"/>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                    <div style="font-size: 48px; font-weight: 800; color: {{ report.status_color }}">{{ report.risk_score }}%</div>
                    <div style="font-size: 11px;">THREAT INDEX</div>
                </div>
            </div>
        </div>

        <!-- Threat Bar -->
        <div class="threat-section">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span><i class="fas fa-fire"></i> Threat Level: {{ report.threat_text }}</span>
                <span>{{ report.risk_score }} / 100</span>
            </div>
            <div class="threat-bar"><div class="threat-fill" style="width: {{ report.risk_score }}%;"></div></div>
        </div>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-icon"><i class="fas fa-link"></i></div><div style="font-size: 11px; color: var(--text-tertiary);">URL ANALYSÉE</div><div style="font-size: 12px; word-break: break-all;">{{ report.url }}</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="fas fa-globe"></i></div><div style="font-size: 11px; color: var(--text-tertiary);">DOMAINE</div><div><strong>{{ report.domain }}</strong><br><span style="font-size: 10px;">{{ report.registrar }}</span></div></div>
            <div class="stat-card"><div class="stat-icon"><i class="fas fa-calendar-alt"></i></div><div style="font-size: 11px; color: var(--text-tertiary);">ANCIENNETÉ</div><div>{{ report.age_days }}<br><span style="font-size: 10px;">{{ report.country }}</span></div></div>
            <div class="stat-card"><div class="stat-icon"><i class="fas fa-shield-halved"></i></div><div style="font-size: 11px; color: var(--text-tertiary);">SÉCURITÉ</div><div>{% if report.https %}🔒 HTTPS Secure{% else %}🔓 HTTP Insecure{% endif %}</div></div>
        </div>

        <!-- Signals -->
        {% if report.signals %}
        <div class="signals-list">
            <h4 style="width:100%;"><i class="fas fa-exclamation-triangle"></i> Signaux détectés ({{ report.signals|length }})</h4>
            {% for signal in report.signals %}
            <div class="signal"><span>{{ signal.icon }}</span><span>{{ signal.name }}</span></div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Verdict -->
        <div class="verdict-box" style="--status-color: {{ report.status_color }}; --status-gradient: {{ report.status_gradient }}">
            <div class="verdict-text"><i class="fas fa-robot"></i> {{ report.verdict }}</div>
            <div class="recommendation-text" style="color: {{ report.status_color }}; margin-top: 16px;">💡 {{ report.recommendation }}</div>
        </div>

        <!-- Actions -->
        <div class="action-grid">
            <button class="action-btn" onclick="copyReport()"><i class="fas fa-copy"></i> Copier</button>
            <button class="action-btn" onclick="downloadJSON()"><i class="fas fa-download"></i> JSON</button>
            <button class="action-btn" onclick="window.print()"><i class="fas fa-print"></i> Imprimer</button>
        </div>
    </div>
    {% endif %}

    <div class="footer-cyber">
        <i class="fas fa-shield-heart"></i> PHISHGUARD PRO — Scanner + Detective + Judge · ENSA Fès 2026
    </div>
</div>
</div>

<script>
    function setUrl(url) {
        document.getElementById('urlInput').value = url;
        document.getElementById('analyzeForm').submit();
    }

    function copyReport() {
        const reportText = document.querySelector('.result-cyber')?.innerText || '';
        navigator.clipboard.writeText(reportText);
        showToast('📋 Rapport copié !');
    }

    function downloadJSON() {
        const data = {{ report_json|safe }};
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'phishguard_report.json';
        a.click();
        showToast('📥 JSON téléchargé !');
    }

    function showToast(msg) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = '<i class="fas fa-check-circle"></i> ' + msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
</script>
</body>
</html>"""


# ==========================================
# 5. VUE & ROUTAGE
# ==========================================
@csrf_exempt
def home(request):
    report = None
    report_json = 'null'
    url_value = ''
    risk_offset = 565  # valeur par défaut (cercle vide)

    if request.method == 'POST':
        url = request.POST.get('url', '').strip()
        if url:
            url_value = url
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            report = get_clean_report(url)
            report_json = json.dumps(report)
            
            # Calcul de l'offset pour le cercle SVG (évite le filtre abs)
            risk_score = report.get('risk_score', 0)
            # Formule: offset = 565 - (565 * risque / 100)
            risk_offset = int(565 - (565 * risk_score / 100))
            # Limites de sécurité
            if risk_offset < 0:
                risk_offset = 0
            if risk_offset > 565:
                risk_offset = 565

    t = Template(HTML_TEMPLATE)
    c = Context({
        'report': report, 
        'report_json': report_json, 
        'url_value': url_value,
        'risk_offset': risk_offset
    })
    return HttpResponse(t.render(c))


urlpatterns = [
    path('', home),
]

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '__main__')
    execute_from_command_line(sys.argv)
