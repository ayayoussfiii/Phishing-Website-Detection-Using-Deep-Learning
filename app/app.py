"""
app.py
------
Flask — Phishing URL Detector (single file, no templates folder)
Usage: python app/app.py  →  http://localhost:5000
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ✅ CORRECTION : remonter de app/ vers la racine phishing-hybrid-dl/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify
import tensorflow as tf
from src.features.feature_pipeline import FeaturePipeline
from src.features.url_features import extract_url_features, FEATURE_NAMES

app = Flask(__name__)

MODELS_DIR   = PROJECT_ROOT / "models"
PIPELINE_DIR = MODELS_DIR / "pipeline"

# ── Load model once at startup ─────────────────────────────────────────────────
model, pipeline, load_error = None, None, None
try:
    model_path = MODELS_DIR / "hybrid_best.keras"
    if not model_path.exists():
        raise FileNotFoundError("hybrid_best.keras not found — run: python src/training/train.py")
    model    = tf.keras.models.load_model(str(model_path))
    pipeline = FeaturePipeline.load(PIPELINE_DIR)
except Exception as e:
    load_error = str(e)


# ── HTML (inline, no templates folder needed) ─────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PhishGuard — URL Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --navy:      #0f2545;
      --navy-mid:  #1a3a6e;
      --bg:        #f0f2f5;
      --white:     #ffffff;
      --border:    #e4e8ef;
      --border-lt: #f0f3f8;
      --text:      #0d1b2e;
      --muted:     #7a8898;
      --hint:      #aab3c0;
      --surface:   #fafbfd;
      --input-bg:  #f7f9fc;
    }

    body { font-family: 'DM Sans', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

    /* NAV */
    nav { background: var(--navy); height: 54px; padding: 0 36px; display: flex; align-items: center; gap: 14px; }
    .nav-icon { width: 30px; height: 30px; background: rgba(255,255,255,.1); border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .nav-name { font-size: 14px; font-weight: 600; color: #fff; letter-spacing: -.1px; }
    .nav-tag { font-size: 10px; background: rgba(255,255,255,.1); color: rgba(255,255,255,.5); border-radius: 3px; padding: 2px 8px; font-family: 'DM Mono', monospace; letter-spacing: .06em; }
    .nav-right { margin-left: auto; font-size: 11px; color: rgba(255,255,255,.3); font-family: 'DM Mono', monospace; letter-spacing: .04em; }

    /* HERO */
    .hero { background: var(--white); border-bottom: 1px solid var(--border); padding: 36px 36px 30px; display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; }
    .hero-eyebrow { font-size: 10px; font-weight: 600; color: var(--navy); letter-spacing: .16em; text-transform: uppercase; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
    .hero-line { width: 22px; height: 2px; background: var(--navy); border-radius: 1px; }
    .hero h1 { font-size: 26px; font-weight: 700; color: var(--text); letter-spacing: -.4px; line-height: 1.2; margin-bottom: 8px; }
    .hero p { font-size: 13px; color: var(--muted); line-height: 1.75; max-width: 500px; }
    .hero-stats { display: flex; gap: 32px; align-items: center; flex-shrink: 0; }
    .hero-stat { text-align: right; }
    .hero-stat-val { font-size: 26px; font-weight: 700; color: var(--navy); font-family: 'DM Mono', monospace; letter-spacing: -.3px; }
    .hero-stat-lbl { font-size: 10px; color: var(--hint); text-transform: uppercase; letter-spacing: .1em; margin-top: 2px; }
    .hero-divider { width: 1px; height: 40px; background: var(--border); }

    /* METRICS */
    .metrics { display: grid; grid-template-columns: repeat(4,1fr); background: var(--white); border-bottom: 1px solid var(--border); }
    .metric { padding: 14px 24px; border-right: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
    .metric:last-child { border-right: none; }
    .metric-icon { width: 34px; height: 34px; background: #edf1f7; border-radius: 7px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .metric-val { font-size: 17px; font-weight: 700; color: var(--text); font-family: 'DM Mono', monospace; letter-spacing: -.2px; }
    .metric-lbl { font-size: 10px; color: var(--hint); text-transform: uppercase; letter-spacing: .08em; margin-top: 1px; }

    /* GRID */
    .main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 24px 36px; }

    /* CARD */
    .card { background: var(--white); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .card-head { padding: 13px 18px; border-bottom: 1px solid var(--border-lt); background: var(--surface); display: flex; align-items: center; gap: 8px; font-size: 10px; font-weight: 600; color: #8a97a8; text-transform: uppercase; letter-spacing: .12em; }
    .card-body { padding: 18px; }

    /* INPUT */
    .input-wrap { display: flex; align-items: center; background: var(--input-bg); border: 1.5px solid var(--border); border-radius: 7px; overflow: hidden; margin-bottom: 12px; transition: border-color .15s, box-shadow .15s; }
    .input-wrap:focus-within { border-color: var(--navy); box-shadow: 0 0 0 3px rgba(15,37,69,.08); background: var(--white); }
    .input-wrap input { flex: 1; border: none; outline: none; background: transparent; font-family: 'DM Mono', monospace; font-size: 13px; color: var(--text); padding: 11px 14px; }
    .input-wrap input::placeholder { color: #bbc4cf; }
    .input-clear { padding: 0 12px; cursor: pointer; color: #bbc4cf; display: none; align-items: center; background: none; border: none; font-size: 20px; line-height: 1; }
    .input-clear:hover { color: var(--muted); }

    /* BUTTON */
    .btn-analyze { width: 100%; background: var(--navy); color: #fff; border: none; border-radius: 7px; padding: 12px 20px; font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 500; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: background .15s, transform .1s, box-shadow .15s; letter-spacing: .01em; }
    .btn-analyze:hover { background: var(--navy-mid); box-shadow: 0 4px 16px rgba(15,37,69,.25); transform: translateY(-1px); }
    .btn-analyze:active { transform: translateY(0); }
    .btn-analyze.loading { opacity: .7; pointer-events: none; }
    .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.3); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading .spinner { display: block; }
    .loading .btn-text { display: none; }

    /* EXAMPLES */
    .ex-label { font-size: 10px; font-weight: 600; color: #bbc4cf; text-transform: uppercase; letter-spacing: .1em; margin: 14px 0 7px; display: block; }
    .ex-chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .ex-chip { font-family: 'DM Mono', monospace; font-size: 10px; background: #f4f6fa; border: 1px solid var(--border); border-radius: 20px; padding: 4px 10px; color: var(--muted); cursor: pointer; transition: all .12s; }
    .ex-chip:hover { background: #edf1f7; border-color: var(--navy); color: var(--navy); }

    /* PLACEHOLDER */
    .placeholder { border: 1.5px dashed var(--border); border-radius: 8px; padding: 52px 24px; text-align: center; color: #cdd4df; font-family: 'DM Mono', monospace; font-size: 12px; letter-spacing: .06em; display: flex; flex-direction: column; align-items: center; gap: 10px; }

    /* RESULT */
    .result-box { display: none; }
    .result-box.visible { display: block; }
    .result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
    .result-badge { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 600; padding: 5px 13px; border-radius: 20px; }
    .result-score { font-size: 12px; color: var(--muted); font-family: 'DM Mono', monospace; }
    .bar-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .bar-label span:first-child { font-size: 10px; font-weight: 600; color: #8a97a8; text-transform: uppercase; letter-spacing: .1em; }
    .bar-label span:last-child { font-size: 11px; font-weight: 600; font-family: 'DM Mono', monospace; }
    .bar-track { background: #eef1f6; border-radius: 99px; height: 5px; overflow: hidden; margin-bottom: 18px; }
    .bar-fill { height: 100%; border-radius: 99px; transition: width .5s cubic-bezier(.4,0,.2,1); }
    .url-display { display: flex; align-items: flex-start; gap: 9px; background: var(--input-bg); border: 1px solid var(--border); border-radius: 7px; padding: 10px 13px; margin-bottom: 14px; }
    .url-display span { font-family: 'DM Mono', monospace; font-size: 11px; color: #4a5568; word-break: break-all; line-height: 1.6; }
    .verdict-box { background: var(--surface); border: 1px solid var(--border); border-radius: 0 7px 7px 0; padding: 11px 14px; }
    .verdict-box p { font-size: 13px; color: var(--muted); line-height: 1.65; }

    /* ERROR */
    .error-box { display: none; align-items: center; gap: 10px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #991b1b; }

    /* FEATURES */
    .feats-section { grid-column: 1 / -1; background: var(--white); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .feats-toggle { width: 100%; background: var(--surface); border: none; border-bottom: 1px solid var(--border-lt); padding: 13px 18px; display: flex; align-items: center; justify-content: space-between; cursor: pointer; font-family: 'DM Sans', sans-serif; }
    .feats-toggle-left { display: flex; align-items: center; gap: 8px; font-size: 10px; font-weight: 600; color: #8a97a8; text-transform: uppercase; letter-spacing: .12em; }
    .feats-arrow { color: var(--hint); font-size: 12px; transition: transform .2s; }
    .feats-toggle.open .feats-arrow { transform: rotate(180deg); }
    .feats-body { display: none; padding: 14px 18px; }
    .feats-body.open { display: grid; grid-template-columns: repeat(3,1fr); gap: 7px; }
    .feat-row { display: flex; justify-content: space-between; align-items: center; padding: 7px 11px; background: var(--input-bg); border-radius: 6px; border: 1px solid #eef1f6; }
    .feat-name { font-size: 11px; color: #8a97a8; font-family: 'DM Mono', monospace; }
    .feat-val  { font-size: 11px; font-weight: 500; color: var(--text); font-family: 'DM Mono', monospace; }

    /* FOOTER */
    footer { text-align: center; padding: 18px 36px 28px; font-size: 10px; color: #bbc4cf; letter-spacing: .08em; font-family: 'DM Mono', monospace; border-top: 1px solid var(--border); background: var(--white); margin-top: 0; }
  </style>
</head>
<body>

  <nav>
    <div class="nav-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,.9)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
    </div>
    <span class="nav-name">PhishGuard</span>
    <span class="nav-tag">v2.0 · ENSA Fès</span>
    <span class="nav-right">BiLSTM + Self-Attention + Structural Features · Master Thesis 2026</span>
  </nav>

  <div class="hero">
    <div>
      <div class="hero-eyebrow"><div class="hero-line"></div>Deep Learning · NLP · Cybersecurity</div>
      <h1>Phishing URL Detection<br>powered by neural networks</h1>
      <p>A hybrid architecture combining character-level BiLSTM with structural feature engineering to classify malicious URLs with near-perfect precision.</p>
    </div>
    <div class="hero-stats">
      <div class="hero-stat"><div class="hero-stat-val">99.6%</div><div class="hero-stat-lbl">Accuracy</div></div>
      <div class="hero-divider"></div>
      <div class="hero-stat"><div class="hero-stat-val">0.999</div><div class="hero-stat-lbl">AUC-ROC</div></div>
    </div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="metric-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
      <div><div class="metric-val">99.6%</div><div class="metric-lbl">Accuracy</div></div>
    </div>
    <div class="metric">
      <div class="metric-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 118 2.83"/><path d="M22 12A10 10 0 0012 2v10z"/></svg></div>
      <div><div class="metric-val">0.999</div><div class="metric-lbl">AUC-ROC</div></div>
    </div>
    <div class="metric">
      <div class="metric-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg></div>
      <div><div class="metric-val">100k</div><div class="metric-lbl">URLs trained</div></div>
    </div>
    <div class="metric">
      <div class="metric-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg></div>
      <div><div class="metric-val">18</div><div class="metric-lbl">Features</div></div>
    </div>
  </div>

  <div class="main">

    <!-- Input -->
    <div class="card">
      <div class="card-head">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        Target URL
      </div>
      <div class="card-body">
        <div class="input-wrap" id="input-wrap">
          <input type="text" id="url-input" placeholder="https://example.com/path?query=value" autocomplete="off" spellcheck="false"/>
          <button class="input-clear" id="clear-btn">×</button>
        </div>
        <button class="btn-analyze" id="analyze-btn" onclick="analyze()">
          <div class="spinner"></div>
          <span class="btn-text">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:5px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Analyze URL
          </span>
        </button>
        <span class="ex-label">Quick examples</span>
        <div class="ex-chips">
          <span class="ex-chip" onclick="setUrl('https://www.google.com')">google.com</span>
          <span class="ex-chip" onclick="setUrl('https://github.com/tensorflow/tensorflow')">github.com</span>
          <span class="ex-chip" onclick="setUrl('https://www.wikipedia.org/wiki/Phishing')">wikipedia.org</span>
          <span class="ex-chip" onclick="setUrl('http://paypal-secure-login.xyz/verify?user=admin')">paypal-login.xyz</span>
          <span class="ex-chip" onclick="setUrl('http://amazon-account-update.tk/confirm&session=abc123')">amazon-update.tk</span>
          <span class="ex-chip" onclick="setUrl('http://192.168.1.1/login?bank=chase&password=reset')">192.168.1.1/login</span>
        </div>
      </div>
    </div>

    <!-- Result -->
    <div class="card">
      <div class="card-head">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        Detection result
      </div>
      <div class="card-body">
        <div class="placeholder" id="placeholder">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dde3ec" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          <span>Enter a URL above and click Analyze</span>
        </div>
        <div class="error-box" id="error-box">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          <span id="error-msg"></span>
        </div>
        <div class="result-box" id="result-box">
          <div class="result-header">
            <div class="result-badge" id="result-badge"></div>
            <div class="result-score" id="result-score"></div>
          </div>
          <div class="bar-label">
            <span>Risk score</span>
            <span id="bar-pct"></span>
          </div>
          <div class="bar-track"><div class="bar-fill" id="bar-fill" style="width:0%"></div></div>
          <div class="url-display">
            <svg style="flex-shrink:0;margin-top:1px" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#8a97a8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>
            <span id="result-url"></span>
          </div>
          <div class="verdict-box" id="verdict-box"><p id="verdict-text"></p></div>
        </div>
      </div>
    </div>

    <!-- Features -->
    <div class="feats-section">
      <button class="feats-toggle" id="feats-toggle" onclick="toggleFeats()">
        <div class="feats-toggle-left">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#0f2545" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
          18 structural signals extracted from URL
        </div>
        <span class="feats-arrow">▾</span>
      </button>
      <div class="feats-body" id="feats-body"></div>
    </div>

  </div>

  <footer>PhishTank · Tranco Top 1M &nbsp;·&nbsp; Hybrid BiLSTM + Dense Fusion &nbsp;·&nbsp; ENSA Fès 2026</footer>

  <script>
    const LEVELS = {
      safe:       { color:'#166534', bg:'#dcfce7', bar:'#22c55e', icon:'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>' },
      suspicious: { color:'#92400e', bg:'#fef3c7', bar:'#f59e0b', icon:'<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>' },
      phishing:   { color:'#991b1b', bg:'#fee2e2', bar:'#ef4444', icon:'<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>' },
    };

    const urlInput  = document.getElementById('url-input');
    const clearBtn  = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');

    urlInput.addEventListener('input', () => {
      clearBtn.style.display = urlInput.value ? 'flex' : 'none';
    });
    clearBtn.addEventListener('click', () => {
      urlInput.value = '';
      clearBtn.style.display = 'none';
      urlInput.focus();
    });
    urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') analyze(); });

    function setUrl(u) {
      urlInput.value = u;
      clearBtn.style.display = 'flex';
      urlInput.focus();
    }

    function toggleFeats() {
      document.getElementById('feats-body').classList.toggle('open');
      document.getElementById('feats-toggle').classList.toggle('open');
    }

    async function analyze() {
      const url = urlInput.value.trim();
      if (!url) return;

      analyzeBtn.classList.add('loading');
      document.getElementById('placeholder').style.display  = 'none';
      document.getElementById('error-box').style.display    = 'none';
      document.getElementById('result-box').className       = 'result-box';

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (!res.ok || data.error) { showError(data.error || 'Server error.'); }
        else { showResult(data); }
      } catch (e) {
        showError('Network error — make sure the server is running.');
      } finally {
        analyzeBtn.classList.remove('loading');
      }
    }

    function showError(msg) {
      document.getElementById('error-msg').textContent  = msg;
      document.getElementById('error-box').style.display = 'flex';
    }

    function showResult(d) {
      const lv = LEVELS[d.level];
      const badge = document.getElementById('result-badge');
      badge.style.background = lv.bg;
      badge.style.color      = lv.color;
      badge.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${lv.color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">${lv.icon}</svg>${d.status}`;

      document.getElementById('result-score').textContent = `${d.pct} / 100`;
      document.getElementById('bar-pct').textContent      = `${d.pct}%`;
      document.getElementById('bar-pct').style.color      = lv.color;

      const fill = document.getElementById('bar-fill');
      fill.style.background = lv.bar;
      fill.style.width = '0%';
      setTimeout(() => { fill.style.width = d.pct + '%'; }, 30);

      document.getElementById('result-url').textContent    = d.url;
      document.getElementById('verdict-text').textContent  = d.verdict;
      document.getElementById('verdict-box').style.borderLeft = `3px solid ${lv.bar}`;
      document.getElementById('result-box').className = 'result-box visible';

      const body = document.getElementById('feats-body');
      body.innerHTML = d.features.map(f =>
        `<div class="feat-row"><span class="feat-name">${f.name}</span><span class="feat-val">${f.value}</span></div>`
      ).join('');
    }
  </script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/predict", methods=["POST"])
def predict():
    if load_error:
        return jsonify({"error": load_error}), 500

    data = request.get_json()
    url  = (data.get("url") or "").strip()

    if not url or len(url) < 5:
        return jsonify({"error": "Please enter a valid URL."}), 400

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        df = pd.DataFrame({"url": [url], "label": [0]})
        X_seq, X_feat, _ = pipeline.transform(df)
        prob = float(model.predict([X_seq, X_feat], verbose=0)[0][0])
        pct  = int(prob * 100)

        if prob < 0.3:
            status, verdict, level = "Safe",       "No threat detected. This URL appears legitimate and trustworthy.",          "safe"
        elif prob < 0.7:
            status, verdict, level = "Suspicious", "Anomalous patterns detected. Proceed with caution before visiting.",        "suspicious"
        else:
            status, verdict, level = "Phishing",   "High risk confirmed. This URL is likely malicious — do not visit.",         "phishing"

        features_raw = extract_url_features(url)
        features     = [
            {"name": name, "value": round(float(val), 4)}
            for name, val in zip(FEATURE_NAMES, features_raw)
        ]

        return jsonify({
            "url": url, "prob": prob, "pct": pct,
            "status": status, "verdict": verdict, "level": level,
            "features": features,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
