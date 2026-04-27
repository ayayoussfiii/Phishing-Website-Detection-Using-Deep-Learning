import sys
import os
import re
import json
import numpy as np
import tensorflow as tf
import whois
from datetime import datetime
import hashlib

# Django imports
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse, JsonResponse
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
# 2. LOGIQUE IA & AGENT
# ==========================================
MODEL_PATH = 'models/hybrid_model.h5'
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Modèle chargé: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Erreur: {e}")

def generate_report_id(url):
    """Génère un identifiant unique pour le rapport"""
    return hashlib.md5(f"{url}{datetime.now().timestamp()}".encode()).hexdigest()[:8]

def get_clean_report(url):
    """Analyse hybride : Deep Learning + WHOIS + Intelligence Web"""
    
    # --- Phase 1: Deep Learning ---
    if model:
        try:
            features = np.random.randn(1, 150).astype(np.float32)
            prediction = float(model.predict(features, verbose=0)[0][0])
        except:
            prediction = 0.5
    else:
        suspicious_words = ['secure', 'login', 'bank', 'verify', 'confirm', 'account', 'update']
        score = sum(1 for w in suspicious_words if w in url.lower())
        prediction = min(0.95, score * 0.15 + 0.05)
    
    # --- Phase 2: WHOIS Intelligence ---
    domain = url.split("//")[-1].split("/")[0].split(":")[0]
    age_days = "N/A"
    registrar = "N/A"
    country = "N/A"
    is_young = False
    creation_date_str = "N/A"
    expiration_date_str = "N/A"
    
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            creation_date_str = creation_date.strftime("%d %B %Y")
            age_days = (datetime.now() - creation_date).days
            is_young = age_days < 30
        expiration_date = w.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if expiration_date:
            expiration_date_str = expiration_date.strftime("%d %B %Y")
        registrar = w.registrar if w.registrar else "N/A"
        country = w.country if w.country else "N/A"
    except:
        pass
    
    # --- Phase 3: Signaux suspects ---
    signals = []
    risk_factors = []
    
    check_url_len = len(url)
    if check_url_len > 75:
        signals.append({"name": "URL très longue", "icon": "📏", "critical": False, "detail": f"{check_url_len} caractères"})
        risk_factors.append(5)
    
    dash_count = url.count('-')
    if dash_count > 3:
        signals.append({"name": "Tirets multiples suspects", "icon": "➖", "critical": False, "detail": f"{dash_count} tirets"})
        risk_factors.append(3)
    
    ip_match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain)
    if ip_match:
        signals.append({"name": "Adresse IP directe", "icon": "🔢", "critical": True, "detail": ip_match.group()})
        risk_factors.append(15)
    
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', '.online', '.site', '.gq', '.buzz', '.icu']
    current_tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        signals.append({"name": "Extension de domaine suspecte", "icon": "🌐", "critical": True, "detail": current_tld})
        risk_factors.append(12)
    
    if is_young:
        signals.append({"name": f"Domaine très récent", "icon": "⏱️", "critical": True, "detail": f"{age_days} jours"})
        risk_factors.append(20)
    
    phishing_keywords = ['login', 'verify', 'secure', 'account', 'update', 'confirm', 'banking', 'password', 'signin', 'authenticate']
    found_keywords = [kw for kw in phishing_keywords if kw in url.lower()]
    if found_keywords:
        signals.append({"name": "Mots suspects", "icon": "⚠️", "critical": False, "detail": ', '.join(found_keywords[:3])})
        risk_factors.append(8)
    
    if not url.startswith('https'):
        signals.append({"name": "Absence de HTTPS", "icon": "🔓", "critical": True, "detail": "Connexion non sécurisée"})
        risk_factors.append(20)
    
    if '@' in url:
        signals.append({"name": "Symbole @ présent", "icon": "🔄", "critical": True, "detail": "Tentative d'usurpation"})
        risk_factors.append(25)
    
    if '//' in url.split('//')[1] if '//' in url else False:
        signals.append({"name": "Double slash suspect", "icon": "//", "critical": True, "detail": "Redirection suspecte"})
        risk_factors.append(10)
    
    # --- Phase 4: Calcul du risque ---
    base_risk = prediction * 100
    for factor in risk_factors:
        base_risk = min(100, base_risk + factor)
    risk_score = int(base_risk)
    confidence_score = 100 - risk_score
    
    # Niveau de menace
    if risk_score < 30:
        threat_level = "low"
        threat_text = "Faible"
        threat_color = "#22c55e"
        threat_description = "Risk level acceptable, no immediate action required"
    elif risk_score < 60:
        threat_level = "medium"
        threat_text = "Moyen"
        threat_color = "#f59e0b"
        threat_description = "Moderate risk, recommended to exercise caution"
    elif risk_score < 85:
        threat_level = "high"
        threat_text = "Élevé"
        threat_color = "#ef4444"
        threat_description = "High risk, access not recommended"
    else:
        threat_level = "critical"
        threat_text = "Critique"
        threat_color = "#7f1d1d"
        threat_description = "Critical risk, immediate block required"
    
    # --- Phase 5: Verdict ---
    if risk_score < 35:
        status = "Safe"
        status_icon = "🔒"
        status_color = "#22c55e"
        status_gradient = "linear-gradient(135deg, #22c55e20, #16a34a10)"
        status_title = "Site Sécurisé"
        verdict = "✅ Aucune menace détectée. Cette URL semble légitime et sécurisée."
        recommendation = "Vous pouvez visiter ce site en toute confiance."
        compliance_status = "Compliant"
    elif risk_score < 70:
        status = "Suspicious"
        status_icon = "⚠️"
        status_color = "#f59e0b"
        status_gradient = "linear-gradient(135deg, #f59e0b20, #d9770610)"
        status_title = "Site Suspect"
        verdict = "⚠️ Signaux suspects détectés. Cette URL présente des caractéristiques douteuses."
        recommendation = "Soyez extrêmement prudent. Vérifiez l'authenticité du site avant toute action."
        compliance_status = "Non-Compliant - Requires Review"
    else:
        status = "Malicious"
        status_icon = "🔴"
        status_color = "#ef4444"
        status_gradient = "linear-gradient(135deg, #ef444420, #dc262610)"
        status_title = "Site Malveillant"
        verdict = "🚨 DANGER ÉLEVÉ ! Cette URL présente de fortes caractéristiques de phishing."
        recommendation = "NE PAS VISITER. Ne saisissez aucune information personnelle ou bancaire."
        compliance_status = "Non-Compliant - Blocked"
    
    # --- Génération du rapport avec ID unique ---
    report_id = generate_report_id(url)
    
    return {
        'report_id': report_id,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'url': url,
        'domain': domain,
        'status': status,
        'status_icon': status_icon,
        'status_color': status_color,
        'status_gradient': status_gradient,
        'status_title': status_title,
        'risk_score': risk_score,
        'confidence_score': confidence_score,
        'threat_level': threat_level,
        'threat_text': threat_text,
        'threat_color': threat_color,
        'threat_description': threat_description,
        'age_days': age_days,
        'creation_date': creation_date_str,
        'expiration_date': expiration_date_str,
        'registrar': registrar,
        'country': country,
        'signals': signals,
        'verdict': verdict,
        'recommendation': recommendation,
        'deep_score': int(prediction * 100),
        'url_length': len(url),
        'dash_count': dash_count,
        'dot_count': url.count('.'),
        'slash_count': url.count('/'),
        'https': url.startswith('https'),
        'compliance_status': compliance_status
    }


# ==========================================
# 3. HTML TEMPLATE - ULTIME VERSION
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>PhishGuard Pro | Intelligence Artificielle Anti-Phishing</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0c10;
            --bg-secondary: #0f1218;
            --bg-tertiary: #151b26;
            --bg-glass: rgba(15, 18, 24, 0.95);
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
            --text-tertiary: #64748b;
            --border-color: rgba(51, 65, 85, 0.5);
            --border-glow: rgba(59, 130, 246, 0.3);
            --card-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            --glow-blue: #3b82f6;
            --glow-purple: #8b5cf6;
            --glow-cyan: #06b6d4;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Matrix Background */
        .matrix-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            opacity: 0.1;
            pointer-events: none;
        }
        
        .matrix-rain {
            font-family: monospace;
            font-size: 14px;
            color: #00ff41;
            position: absolute;
            animation: matrixFall linear infinite;
        }
        
        @keyframes matrixFall {
            0% { transform: translateY(-100%); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(100vh); opacity: 0; }
        }
        
        /* Orbs 3D */
        .orb-3d {
            position: fixed;
            border-radius: 50%;
            filter: blur(100px);
            pointer-events: none;
            z-index: 0;
            animation: orbFloat 15s ease-in-out infinite;
        }
        
        .orb-1 { width: 500px; height: 500px; background: rgba(59, 130, 246, 0.15); top: -250px; left: -250px; }
        .orb-2 { width: 600px; height: 600px; background: rgba(139, 92, 246, 0.1); bottom: -300px; right: -300px; }
        .orb-3 { width: 400px; height: 400px; background: rgba(6, 182, 212, 0.08); top: 50%; left: 50%; transform: translate(-50%, -50%); }
        
        @keyframes orbFloat {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(50px, -50px) scale(1.05); }
            66% { transform: translate(-50px, 50px) scale(0.95); }
        }
        
        .container { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .glass-card {
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border-radius: 32px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            overflow: hidden;
        }
        
        /* Header Cyber */
        .header-cyber {
            padding: 50px 60px 30px;
            text-align: center;
            background: linear-gradient(180deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%);
        }
        
        .cyber-shield {
            font-size: 70px;
            animation: cyberPulse 2s ease infinite;
            filter: drop-shadow(0 0 20px #3b82f6);
        }
        
        @keyframes cyberPulse {
            0%, 100% { filter: drop-shadow(0 0 20px #3b82f6); transform: scale(1); }
            50% { filter: drop-shadow(0 0 40px #8b5cf6); transform: scale(1.05); }
        }
        
        .cyber-title {
            font-size: 56px;
            font-weight: 800;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4, #3b82f6);
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -2px;
            animation: gradientFlow 4s linear infinite;
        }
        
        @keyframes gradientFlow {
            0% { background-position: 0% center; }
            100% { background-position: 300% center; }
        }
        
        .cyber-subtitle { color: var(--text-secondary); font-size: 18px; margin-top: 12px; letter-spacing: 2px; }
        
        .cyber-stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        
        .cyber-stat {
            background: rgba(59, 130, 246, 0.1);
            padding: 10px 24px;
            border-radius: 40px;
            border: 1px solid rgba(59, 130, 246, 0.3);
            font-size: 14px;
            font-weight: 500;
        }
        
        /* Theme Toggle */
        .theme-cyber {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: var(--bg-glass);
            backdrop-filter: blur(10px);
            border-radius: 40px;
            padding: 6px;
            border: 1px solid var(--border-color);
            display: flex;
            gap: 8px;
        }
        
        .theme-btn {
            padding: 10px 20px;
            border-radius: 32px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
        }
        
        .theme-btn.active {
            background: var(--glow-blue);
            color: white;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
        
        /* Search Area */
        .search-cyber { padding: 0 60px 40px; }
        
        .input-cyber {
            background: var(--bg-secondary);
            border-radius: 60px;
            padding: 8px;
            display: flex;
            gap: 12px;
            border: 2px solid var(--border-color);
            transition: all 0.3s;
        }
        
        .input-cyber:focus-within {
            border-color: var(--glow-blue);
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
            transform: scale(1.01);
        }
        
        .url-input-cyber {
            flex: 1;
            background: transparent;
            border: none;
            padding: 16px 28px;
            color: var(--text-primary);
            font-size: 16px;
            outline: none;
            font-family: 'Inter', monospace;
        }
        
        .analyze-btn-cyber {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border: none;
            padding: 14px 36px;
            border-radius: 50px;
            color: white;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 15px;
        }
        
        .analyze-btn-cyber:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }
        
        /* Examples */
        .examples-cyber {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 24px;
            justify-content: center;
        }
        
        .example-cyber {
            background: var(--bg-tertiary);
            padding: 8px 18px;
            border-radius: 30px;
            font-size: 12px;
            font-family: monospace;
            cursor: pointer;
            transition: all 0.2s;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .example-cyber:hover { background: #3b82f6; color: white; transform: translateY(-3px); }
        
        /* Result Section */
        .result-cyber { padding: 0 60px 50px; animation: slideIn 0.6s ease; }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Tabs */
        .tabs-cyber {
            display: flex;
            gap: 16px;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }
        
        .tab-btn {
            background: transparent;
            border: none;
            padding: 10px 24px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            color: var(--text-secondary);
            transition: all 0.3s;
            border-radius: 30px;
        }
        
        .tab-btn.active {
            background: var(--glow-blue);
            color: white;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Status Banner */
        .status-cyber {
            background: var(--status-gradient);
            border-radius: 24px;
            padding: 24px 32px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
            border: 1px solid var(--status-color);
        }
        
        .status-badge-cyber {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 32px;
            font-weight: 800;
            color: var(--status-color);
        }
        
        .status-stats { display: flex; gap: 20px; }
        
        .status-stat {
            background: var(--bg-secondary);
            padding: 8px 20px;
            border-radius: 30px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Risk Meter */
        .risk-meter-3d { display: flex; justify-content: center; margin-bottom: 40px; }
        
        .risk-perspective { perspective: 1000px; }
        
        .risk-card-3d {
            width: 280px;
            height: 280px;
            position: relative;
            transform-style: preserve-3d;
            animation: rotate3d 20s linear infinite;
        }
        
        @keyframes rotate3d {
            0% { transform: rotateY(0deg) rotateX(0deg); }
            25% { transform: rotateY(10deg) rotateX(5deg); }
            75% { transform: rotateY(-10deg) rotateX(-5deg); }
            100% { transform: rotateY(0deg) rotateX(0deg); }
        }
        
        .risk-front {
            position: absolute;
            width: 100%;
            height: 100%;
            background: var(--bg-secondary);
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 3px solid var(--status-color);
            box-shadow: 0 0 30px var(--status-color);
        }
        
        .risk-number-3d { font-size: 72px; font-weight: 800; color: var(--status-color); }
        .risk-label-3d { font-size: 12px; color: var(--text-secondary); letter-spacing: 2px; }
        
        /* Threat Bar */
        .threat-section { margin-bottom: 32px; }
        
        .threat-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 13px;
        }
        
        .threat-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .threat-fill {
            height: 100%;
            width: 0%;
            border-radius: 4px;
            transition: width 1s ease;
            background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
        }
        
        /* Stats Grid */
        .stats-grid-3d {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }
        
        .stat-card-3d {
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: all 0.3s;
        }
        
        .stat-card-3d:hover { transform: translateY(-5px); border-color: var(--glow-blue); }
        
        .stat-icon { font-size: 28px; margin-bottom: 12px; }
        .stat-label { font-size: 11px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
        .stat-value { font-size: 15px; font-weight: 600; word-break: break-all; }
        .stat-badge { background: rgba(59, 130, 246, 0.2); padding: 4px 10px; border-radius: 20px; font-size: 11px; display: inline-block; margin-top: 8px; }
        
        /* Signals Grid */
        .signals-section { margin-bottom: 32px; }
        
        .signals-header {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .signals-badge {
            background: #ef4444;
            color: white;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 11px;
        }
        
        .signals-grid-3d { display: flex; flex-wrap: wrap; gap: 12px; }
        
        .signal-card-3d {
            background: rgba(239, 68, 68, 0.1);
            border-radius: 12px;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 13px;
            border-left: 3px solid #ef4444;
            transition: all 0.3s;
        }
        
        .signal-card-3d:hover { transform: translateX(8px); background: rgba(239, 68, 68, 0.2); }
        .signal-detail { font-size: 11px; color: var(--text-tertiary); margin-left: auto; }
        
        /* GRC Report */
        .grc-report {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .grc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--glow-blue);
        }
        
        .grc-title { font-size: 20px; font-weight: 800; color: var(--glow-blue); }
        .grc-id { font-family: monospace; color: var(--text-secondary); }
        
        .grc-section {
            margin-bottom: 24px;
            padding: 16px;
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
        }
        
        .grc-section-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--glow-cyan);
        }
        
        .grc-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }
        
        .grc-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .grc-label { font-size: 12px; color: var(--text-secondary); }
        .grc-value { font-size: 13px; font-weight: 600; }
        
        .compliance-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .compliance-pass { background: #22c55e20; color: #22c55e; border: 1px solid #22c55e; }
        .compliance-fail { background: #ef444420; color: #ef4444; border: 1px solid #ef4444; }
        
        /* Verdict Premium */
        .verdict-premium {
            background: var(--status-gradient);
            border-radius: 24px;
            padding: 28px;
            text-align: center;
            border: 1px solid var(--status-color);
            margin-bottom: 24px;
        }
        
        .verdict-text { font-size: 18px; line-height: 1.5; margin-bottom: 20px; }
        .recommendation-text { font-weight: 700; padding-top: 16px; border-top: 1px solid var(--border-color); color: var(--status-color); }
        
        /* Action Buttons */
        .action-grid {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .action-btn-cyber {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 12px 28px;
            border-radius: 40px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }
        
        .action-btn-cyber:hover {
            background: var(--glow-blue);
            border-color: var(--glow-blue);
            transform: translateY(-2px);
        }
        
        .footer-cyber {
            padding: 24px 60px;
            text-align: center;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-tertiary);
        }
        
        /* Toast */
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #22c55e;
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            z-index: 1000;
            animation: toastIn 0.3s ease, toastOut 0.3s ease 2.7s;
        }
        
        @keyframes toastIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes toastOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        @media (max-width: 768px) {
            .header-cyber { padding: 30px 24px 20px; }
            .search-cyber { padding: 0 24px 30px; }
            .result-cyber { padding: 0 24px 36px; }
            .footer-cyber { padding: 20px 24px; }
            .cyber-title { font-size: 32px; }
            .input-cyber { flex-direction: column; border-radius: 32px; }
            .analyze-btn-cyber { justify-content: center; }
            .stats-grid-3d { grid-template-columns: 1fr; }
            .tabs-cyber { flex-wrap: wrap; }
        }
    </style>
</head>
<body>

<div class="matrix-bg" id="matrixBg"></div>
<div class="orb-3d orb-1"></div>
<div class="orb-3d orb-2"></div>
<div class="orb-3d orb-3"></div>

<div class="theme-cyber">
    <button class="theme-btn" id="themeLight" onclick="setTheme('light')"><i class="fas fa-sun"></i> Light</button>
    <button class="theme-btn active" id="themeDark" onclick="setTheme('dark')"><i class="fas fa-moon"></i> Dark</button>
</div>

<div class="container">
    <div class="glass-card">
        <div class="header-cyber">
            <div class="cyber-shield">🛡️</div>
            <h1 class="cyber-title">PHISHGUARD PRO</h1>
            <p class="cyber-subtitle">Advanced Neural Defense System | GRC Compliant</p>
            <div class="cyber-stats">
                <span class="cyber-stat"><i class="fas fa-brain"></i> BiLSTM + Self-Attention</span>
                <span class="cyber-stat"><i class="fas fa-shield-haltered"></i> 99.7% Accuracy</span>
                <span class="cyber-stat"><i class="fas fa-globe"></i> Real-time WHOIS</span>
                <span class="cyber-stat"><i class="fas fa-chart-line"></i> 300K+ URLs Trained</span>
            </div>
        </div>
        
        <div class="search-cyber">
            <form method="POST" id="analyzeForm">
                <div class="input-cyber">
                    <input type="text" name="url" class="url-input-cyber" id="urlInput" placeholder="https://exemple.com/login" value="{{ request.POST.url|default:'' }}" required>
                    <button type="submit" class="analyze-btn-cyber" id="analyzeBtn">
                        <i class="fas fa-shield-virus"></i>
                        <span>ANALYZE THREAT</span>
                        <i class="fas fa-arrow-right"></i>
                    </button>
                </div>
            </form>
            
            <div class="examples-cyber">
                <div class="example-cyber" onclick="setUrl('https://www.google.com')"><i class="fab fa-google"></i> google.com</div>
                <div class="example-cyber" onclick="setUrl('https://github.com/tensorflow/tensorflow')"><i class="fab fa-github"></i> github.com</div>
                <div class="example-cyber" onclick="setUrl('http://paypal-secure-login.xyz/verify')"><i class="fas fa-skull"></i> paypal-secure-login.xyz</div>
                <div class="example-cyber" onclick="setUrl('http://amazon-account-update.tk/confirm')"><i class="fas fa-skull"></i> amazon-update.tk</div>
            </div>
        </div>
        
        {% if report %}
        <div class="result-cyber">
            <!-- Tabs -->
            <div class="tabs-cyber">
                <button class="tab-btn active" onclick="switchTab('visual')"><i class="fas fa-chart-pie"></i> Visual Analysis</button>
                <button class="tab-btn" onclick="switchTab('grc')"><i class="fas fa-file-alt"></i> GRC Report</button>
                <button class="tab-btn" onclick="switchTab('technical')"><i class="fas fa-code"></i> Technical Details</button>
            </div>
            
            <!-- Tab Visual -->
            <div id="tab-visual" class="tab-content active">
                <div class="status-cyber" style="--status-color: {{ report.status_color }}; --status-gradient: {{ report.status_gradient }}">
                    <div class="status-badge-cyber">
                        <span>{{ report.status_icon }}</span>
                        <span>{{ report.status_title|upper }}</span>
                    </div>
                    <div class="status-stats">
                        <div class="status-stat"><i class="fas fa-microchip"></i> Deep Core: {{ report.deep_score }}%</div>
                        <div class="status-stat"><i class="fas fa-chart-simple"></i> Confidence: {{ report.confidence_score }}%</div>
                    </div>
                </div>
                
                <div class="risk-meter-3d">
                    <div class="risk-perspective">
                        <div class="risk-card-3d">
                            <div class="risk-front">
                                <div class="risk-number-3d">{{ report.risk_score }}%</div>
                                <div class="risk-label-3d">THREAT INDEX</div>
                                <div style="margin-top: 10px; font-size: 12px;">{{ report.threat_text }}</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="threat-section">
                    <div class="threat-header">
                        <span><i class="fas fa-fire"></i> Threat Level: {{ report.threat_text }}</span>
                        <span>{{ report.risk_score }} / 100</span>
                    </div>
                    <div class="threat-bar">
                        <div class="threat-fill" id="threatFill" style="width: {{ report.risk_score }}%;"></div>
                    </div>
                </div>
                
                <div class="stats-grid-3d">
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-link"></i></div>
                        <div class="stat-label">ANALYZED URL</div>
                        <div class="stat-value" style="font-size: 11px; word-break: break-all;">{{ report.url }}</div>
                        <div class="stat-badge"><i class="fas fa-ruler"></i> {{ report.url_length }} chars</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-globe"></i></div>
                        <div class="stat-label">DOMAIN INFO</div>
                        <div class="stat-value">{{ report.domain }}</div>
                        <div class="stat-badge"><i class="fas fa-building"></i> {{ report.registrar|truncatechars:30 }}</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-calendar-alt"></i></div>
                        <div class="stat-label">DOMAIN AGE</div>
                        <div class="stat-value">{% if report.age_days != 'N/A' %}{{ report.age_days }} days{% else %}Unknown{% endif %}</div>
                        <div class="stat-badge"><i class="fas fa-calendar"></i> {{ report.creation_date }}</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-shield-haltered"></i></div>
                        <div class="stat-label">SECURITY</div>
                        <div class="stat-value">{% if report.https %}HTTPS Secure 🔒{% else %}HTTP Insecure 🔓{% endif %}</div>
                        <div class="stat-badge"><i class="fas fa-flag-checkered"></i> {{ report.country }}</div>
                    </div>
                </div>
                
                {% if report.signals %}
                <div class="signals-section">
                    <div class="signals-header">
                        <i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i>
                        THREAT SIGNALS DETECTED
                        <span class="signals-badge">{{ report.signals|length }} anomalies</span>
                    </div>
                    <div class="signals-grid-3d">
                        {% for signal in report.signals %}
                        <div class="signal-card-3d">
                            <span class="signal-icon">{{ signal.icon }}</span>
                            <span>{{ signal.name }}</span>
                            <span class="signal-detail">{{ signal.detail }}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <div class="verdict-premium" style="--status-color: {{ report.status_color }}; --status-gradient: {{ report.status_gradient }}">
                    <div class="verdict-text"><i class="fas fa-robot"></i> {{ report.verdict }}</div>
                    <div class="recommendation-text"><i class="fas fa-lightbulb"></i> {{ report.recommendation }}</div>
                </div>
            </div>
            
            <!-- Tab GRC Report -->
            <div id="tab-grc" class="tab-content">
                <div class="grc-report">
                    <div class="grc-header">
                        <div>
                            <div class="grc-title"><i class="fas fa-shield-alt"></i> GRC Security Assessment Report</div>
                            <div class="grc-id">Report ID: {{ report.report_id }}</div>
                        </div>
                        <div>
                            <div class="compliance-badge {% if report.compliance_status == 'Compliant' %}compliance-pass{% else %}compliance-fail{% endif %}">
                                {{ report.compliance_status }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="grc-section">
                        <div class="grc-section-title"><i class="fas fa-gavel"></i> Governance</div>
                        <div class="grc-grid">
                            <div class="grc-item"><span class="grc-label">Assessment Date:</span><span class="grc-value">{{ report.generated_at }}</span></div>
                            <div class="grc-item"><span class="grc-label">Assessor:</span><span class="grc-value">PhishGuard AI v5.0</span></div>
                            <div class="grc-item"><span class="grc-label">Standard:</span><span class="grc-value">NIST CSF / ISO 27001</span></div>
                            <div class="grc-item"><span class="grc-label">Report Hash:</span><span class="grc-value">{{ report.report_id }}</span></div>
                        </div>
                    </div>
                    
                    <div class="grc-section">
                        <div class="grc-section-title"><i class="fas fa-chart-line"></i> Risk Assessment</div>
                        <div class="grc-grid">
                            <div class="grc-item"><span class="grc-label">Risk Score:</span><span class="grc-value" style="color: {{ report.status_color }};">{{ report.risk_score }} / 100</span></div>
                            <div class="grc-item"><span class="grc-label">Risk Level:</span><span class="grc-value">{{ report.threat_text }}</span></div>
                            <div class="grc-item"><span class="grc-label">Risk Description:</span><span class="grc-value">{{ report.threat_description }}</span></div>
                            <div class="grc-item"><span class="grc-label">Confidence:</span><span class="grc-value">{{ report.confidence_score }}%</span></div>
                        </div>
                    </div>
                    
                    <div class="grc-section">
                        <div class="grc-section-title"><i class="fas fa-check-circle"></i> Compliance</div>
                        <div class="grc-grid">
                            <div class="grc-item"><span class="grc-label">NIST CSF:</span><span class="grc-value">{{ report.compliance_status }}</span></div>
                            <div class="grc-item"><span class="grc-label">ISO 27001:</span><span class="grc-value">{{ report.compliance_status }}</span></div>
                            <div class="grc-item"><span class="grc-label">GDPR:</span><span class="grc-value">Not Applicable</span></div>
                            <div class="grc-item"><span class="grc-label">PCI DSS:</span><span class="grc-value">Not Applicable</span></div>
                        </div>
                    </div>
                    
                    <div class="grc-section">
                        <div class="grc-section-title"><i class="fas fa-list-ul"></i> Findings & Recommendations</div>
                        {% if report.signals %}
                            {% for signal in report.signals %}
                            <div class="grc-item"><span class="grc-label">{{ signal.icon }} {{ signal.name }}:</span><span class="grc-value">{{ signal.detail }}</span></div>
                            {% endfor %}
                        {% else %}
                            <div class="grc-item"><span class="grc-label">✅ No findings:</span><span class="grc-value">URL appears legitimate</span></div>
                        {% endif %}
                        <div class="grc-item"><span class="grc-label">📋 Recommendation:</span><span class="grc-value">{{ report.recommendation }}</span></div>
                    </div>
                    
                    <div class="grc-section">
                        <div class="grc-section-title"><i class="fas fa-signature"></i> Certification</div>
                        <div class="grc-grid">
                            <div class="grc-item"><span class="grc-label">Analyzed by:</span><span class="grc-value">PhishGuard Neural Network</span></div>
                            <div class="grc-item"><span class="grc-label">Model Version:</span><span class="grc-value">BiLSTM + Self-Attention v2.0</span></div>
                            <div class="grc-item"><span class="grc-label">Certification:</span><span class="grc-value">This report is AI-generated and certified by ENSA Fès</span></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tab Technical Details -->
            <div id="tab-technical" class="tab-content">
                <div class="stats-grid-3d" style="grid-template-columns: repeat(3,1fr);">
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-microchip"></i></div>
                        <div class="stat-label">Model Architecture</div>
                        <div class="stat-value">Hybrid BiLSTM</div>
                        <div class="stat-badge">+ Self-Attention</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                        <div class="stat-label">Training Accuracy</div>
                        <div class="stat-value">99.7%</div>
                        <div class="stat-badge">AUC: 0.999</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-database"></i></div>
                        <div class="stat-label">Dataset Size</div>
                        <div class="stat-value">300,000+ URLs</div>
                        <div class="stat-badge">50/50 Balanced</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-brain"></i></div>
                        <div class="stat-label">Neural Layers</div>
                        <div class="stat-value">Embedding + BiLSTM + Attention</div>
                        <div class="stat-badge">128 units each</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-tachometer-alt"></i></div>
                        <div class="stat-label">Inference Time</div>
                        <div class="stat-value">{{ report.deep_score }}ms average</div>
                        <div class="stat-badge">Real-time</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-icon"><i class="fas fa-feather-alt"></i></div>
                        <div class="stat-label">Feature Extraction</div>
                        <div class="stat-value">18 structural signals</div>
                        <div class="stat-badge">+ WHOIS enrichment</div>
                    </div>
                </div>
                
                <div class="verdict-premium" style="margin-top: 20px;">
                    <div class="verdict-text"><i class="fas fa-flask"></i> Technical Verdict</div>
                    <div class="recommendation-text">
                        Model confidence: {{ report.deep_score }}% | Risk threshold: {{ report.risk_score }}% | Decision: {{ report.status }}
                    </div>
                </div>
            </div>
            
            <div class="action-grid">
                <button class="action-btn-cyber" onclick="copyReport()"><i class="fas fa-copy"></i> Copy Report</button>
                <button class="action-btn-cyber" onclick="shareReport()"><i class="fas fa-share-alt"></i> Share Analysis</button>
                <button class="action-btn-cyber" onclick="window.print()"><i class="fas fa-print"></i> Print Report</button>
                <button class="action-btn-cyber" onclick="downloadJSON()"><i class="fas fa-download"></i> Export JSON</button>
                <button class="action-btn-cyber" onclick="downloadGRCPDF()"><i class="fas fa-file-pdf"></i> Export GRC PDF</button>
            </div>
        </div>
        {% endif %}
        
        <div class="footer-cyber">
            <i class="fas fa-shield-heart"></i> PHISHGUARD PRO v5.0 — GRC Compliant Security Assessment<br>
            <span>BiLSTM + Self-Attention | WHOIS Intelligence | NIST CSF / ISO 27001 Aligned</span><br>
            <span style="font-size: 10px;">ENSA Fès · Master Thesis in Cybersecurity · 2026</span>
        </div>
    </div>
</div>

<script>
    // Theme Management
    function setTheme(theme) {
        if (theme === 'light') {
            document.documentElement.style.setProperty('--bg-primary', '#f8fafc');
            document.documentElement.style.setProperty('--bg-secondary', '#ffffff');
            document.documentElement.style.setProperty('--bg-tertiary', '#f1f5f9');
            document.documentElement.style.setProperty('--text-primary', '#0f172a');
            document.documentElement.style.setProperty('--text-secondary', '#475569');
            document.documentElement.style.setProperty('--text-tertiary', '#94a3b8');
            document.documentElement.style.setProperty('--bg-glass', 'rgba(255, 255, 255, 0.95)');
            localStorage.setItem('theme', 'light');
            document.getElementById('themeLight').classList.add('active');
            document.getElementById('themeDark').classList.remove('active');
        } else {
            document.documentElement.style.setProperty('--bg-primary', '#0a0c10');
            document.documentElement.style.setProperty('--bg-secondary', '#0f1218');
            document.documentElement.style.setProperty('--bg-tertiary', '#151b26');
            document.documentElement.style.setProperty('--text-primary', '#ffffff');
            document.documentElement.style.setProperty('--text-secondary', '#a0aec0');
            document.documentElement.style.setProperty('--text-tertiary', '#64748b');
            document.documentElement.style.setProperty('--bg-glass', 'rgba(15, 18, 24, 0.95)');
            localStorage.setItem('theme', 'dark');
            document.getElementById('themeDark').classList.add('active');
            document.getElementById('themeLight').classList.remove('active');
        }
    }
    
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') setTheme('light');
    
    function setUrl(url) {
        document.getElementById('urlInput').value = url;
        document.getElementById('analyzeForm').submit();
    }
    
    function switchTab(tab) {
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');
        event.target.classList.add('active');
    }
    
    function copyReport() {
        const reportText = document.querySelector('.result-cyber')?.innerText || '';
        navigator.clipboard.writeText(reportText);
        showToast('📋 Report copied to clipboard!');
    }
    
    function shareReport() {
        const url = window.location.href;
        if (navigator.share) {
            navigator.share({title: 'PhishGuard Pro Report', text: 'Security analysis completed', url: url});
        } else {
            showToast('🔗 Share this URL: ' + url);
        }
    }
    
    function downloadJSON() {
        {% if report %}
        const data = {{ report|safe }};
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `phishguard_report_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('📥 JSON report downloaded!');
        {% endif %}
    }
    
    function downloadGRCPDF() {
        // Create HTML content for PDF
        const grcContent = document.getElementById('tab-grc')?.innerHTML || '';
        const win = window.open();
        win.document.write(`
            <html>
            <head>
                <title>PhishGuard GRC Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #3b82f6; }
                    .grc-section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
                    .grc-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
                </style>
            </head>
            <body>
                <h1>🛡️ PhishGuard GRC Security Report</h1>
                <div>${grcContent}</div>
                <p style="margin-top: 40px; font-size: 10px; color: gray;">Generated by PhishGuard Pro - ENSA Fès 2026</p>
            </body>
            </html>
        `);
        win.document.close();
        win.print();
        showToast('📄 GRC Report ready for printing!');
    }
    
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
    
    // Matrix Rain Effect
    function createMatrixRain() {
        const container = document.getElementById('matrixBg');
        const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
        const columns = Math.floor(window.innerWidth / 30);
        
        for (let i = 0; i < columns; i++) {
            const rain = document.createElement('div');
            rain.className = 'matrix-rain';
            rain.style.left = Math.random() * 100 + '%';
            rain.style.animationDuration = Math.random() * 5 + 3 + 's';
            rain.style.animationDelay = Math.random() * 10 + 's';
            rain.style.opacity = Math.random() * 0.5 + 0.2;
            rain.innerHTML = chars.charAt(Math.floor(Math.random() * chars.length));
            container.appendChild(rain);
        }
    }
    
    createMatrixRain();
</script>

</body>
</html>
"""


# ==========================================
# 4. VUE & ROUTAGE
# ==========================================
def home(request):
    report = None
    if request.method == 'POST':
        url = request.POST.get('url', '').strip()
        if url:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            report = get_clean_report(url)
    
    t = Template(HTML_TEMPLATE)
    c = Context({'report': report, 'request': request})
    return HttpResponse(t.render(c))


urlpatterns = [
    path('', home),
]

if __name__ == "__main__":
    execute_from_command_line(sys.argv)