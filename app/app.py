"""
app.py
------
Interface Gradio — Phishing URL Detector
Usage: python app/app.py  →  http://localhost:7860
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import tensorflow as tf
from src.features.feature_pipeline import FeaturePipeline
from src.features.url_features import extract_url_features, FEATURE_NAMES

MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_DIR = MODELS_DIR / "pipeline"


def load_model_and_pipeline():
    model_path = MODELS_DIR / "hybrid_best.keras"
    if not model_path.exists():
        return None, None, "Model not found. Run: python src/training/train.py"
    model = tf.keras.models.load_model(str(model_path))
    pipeline = FeaturePipeline.load(PIPELINE_DIR)
    return model, pipeline, None


model, pipeline, load_error = load_model_and_pipeline()


def predict_url(url: str):
    if load_error:
        return _error_html(load_error), None

    if not url or len(url.strip()) < 5:
        return _error_html("Please enter a valid URL."), None

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    df = pd.DataFrame({"url": [url], "label": [0]})
    X_seq, X_feat, _ = pipeline.transform(df)
    prob = float(model.predict([X_seq, X_feat], verbose=0)[0][0])
    pct = int(prob * 100)

    if prob < 0.3:
        status    = "Safe"
        verdict   = "No threat detected. This URL appears legitimate."
        tag_color = "#166534"
        tag_bg    = "#dcfce7"
        bar_color = "#22c55e"
        icon      = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>"""
    elif prob < 0.7:
        status    = "Suspicious"
        verdict   = "Unusual patterns detected. Proceed with caution."
        tag_color = "#92400e"
        tag_bg    = "#fef3c7"
        bar_color = "#f59e0b"
        icon      = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#92400e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
    else:
        status    = "Phishing"
        verdict   = "High risk — this URL is likely malicious. Do not visit."
        tag_color = "#991b1b"
        tag_bg    = "#fee2e2"
        bar_color = "#ef4444"
        icon      = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""

    result_html = f"""
<div style="font-family:'Geist','Inter',system-ui,sans-serif;">

  <!-- Status badge row -->
  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
    <div style="display:flex; align-items:center; gap:8px;">
      <span style="
        display:inline-flex; align-items:center; gap:6px;
        background:{tag_bg}; color:{tag_color};
        font-size:12px; font-weight:600;
        padding:5px 12px; border-radius:6px;
        letter-spacing:0.01em;
      ">{icon} {status}</span>
    </div>
    <span style="font-size:12px; color:#94a3b8; font-weight:500;">{pct}% risk score</span>
  </div>

  <!-- Risk bar -->
  <div style="margin-bottom:20px;">
    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:11px; font-weight:500; color:#64748b; letter-spacing:0.05em; text-transform:uppercase;">Risk level</span>
      <span style="font-size:11px; font-weight:600; color:#1e293b;">{pct} / 100</span>
    </div>
    <div style="background:#f1f5f9; border-radius:99px; height:6px; overflow:hidden;">
      <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:99px; transition:width 0.4s ease;"></div>
    </div>
  </div>

  <!-- URL row -->
  <div style="
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:8px;
    padding:10px 14px;
    margin-bottom:16px;
    display:flex; align-items:center; gap:10px;
  ">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>
    <span style="font-family:'JetBrains Mono','Fira Code',monospace; font-size:11px; color:#475569; word-break:break-all;">{url}</span>
  </div>

  <!-- Verdict -->
  <div style="
    display:flex; align-items:flex-start; gap:10px;
    background:#f8fafc; border-radius:8px; padding:12px 14px;
    border-left:3px solid {bar_color};
  ">
    <span style="font-size:12px; color:#64748b; line-height:1.6;">{verdict}</span>
  </div>

</div>
"""

    features = extract_url_features(url)
    feat_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Value":   [round(v, 4) for v in features],
    })

    return result_html, feat_df


def _error_html(msg: str) -> str:
    return f"""
<div style="
  font-family:'Geist','Inter',system-ui,sans-serif;
  display:flex; align-items:center; gap:10px;
  background:#fef2f2; border:1px solid #fecaca;
  border-radius:8px; padding:12px 16px;
  font-size:13px; color:#991b1b;
"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
{msg}</div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #f8fafc !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #0f172a !important;
    min-height: 100vh;
}

footer { display: none !important; }
.gr-block.gr-box { box-shadow: none !important; }

/* ── Topbar ── */
.ph-topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 32px;
    display: flex;
    align-items: center;
    height: 56px;
    gap: 12px;
    margin-bottom: 28px;
}
.ph-logo-dot {
    width: 28px; height: 28px;
    background: #6366f1;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
}
.ph-logo-name {
    font-size: 15px;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.2px;
}
.ph-logo-version {
    font-size: 11px;
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 2px 7px;
    margin-left: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.ph-topbar-right {
    margin-left: auto;
    font-size: 11px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Page title ── */
.ph-page-title {
    padding: 0 32px 20px;
}
.ph-page-title h2 {
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 4px;
    letter-spacing: -0.3px;
}
.ph-page-title p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}

/* ── Stat cards ── */
.ph-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 0 32px 24px;
}
.ph-stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 18px;
}
.ph-stat-label {
    font-size: 11px;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}
.ph-stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.5px;
    line-height: 1;
}
.ph-stat-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
}

/* ── Main content area ── */
.ph-content {
    padding: 0 32px;
}

/* ── Cards ── */
.ph-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    box-shadow: none !important;
}

/* ── Card headers ── */
.ph-card-header {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Input ── */
#url-box textarea, #url-box input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    padding: 11px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
#url-box textarea:focus, #url-box input:focus {
    border-color: #6366f1 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}
#url-box textarea::placeholder, #url-box input::placeholder {
    color: #cbd5e1 !important;
}
#url-box label span {
    display: none !important;
}

/* ── Button ── */
#analyze-btn {
    background: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 11px 0 !important;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
}
#analyze-btn:hover {
    background: #4f46e5 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transform: translateY(-1px) !important;
}
#analyze-btn:active {
    transform: translateY(0) !important;
}

/* ── Examples ── */
.examples-holder .label-wrap,
.gr-samples-table .label-wrap { display: none !important; }

.examples-holder table,
.gr-samples-table table {
    border-collapse: separate !important;
    border-spacing: 4px !important;
}
.examples-holder table td button,
.gr-samples-table td button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    color: #475569 !important;
    padding: 5px 10px !important;
    transition: all 0.12s !important;
    white-space: nowrap;
}
.examples-holder table td button:hover,
.gr-samples-table td button:hover {
    background: #ede9fe !important;
    border-color: #a5b4fc !important;
    color: #4f46e5 !important;
}

/* ── Result placeholder ── */
.result-placeholder {
    border: 1.5px dashed #e2e8f0;
    border-radius: 10px;
    padding: 52px 24px;
    text-align: center;
    color: #cbd5e1;
    font-size: 12px;
    letter-spacing: 0.04em;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

/* ── Accordion ── */
.gradio-accordion {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-top: 0 !important;
}
.gradio-accordion > .label-wrap button {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    letter-spacing: 0.04em !important;
    padding: 14px 20px !important;
    text-transform: uppercase !important;
}

/* ── Dataframe ── */
.gr-dataframe { border: none !important; }
.gr-dataframe table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    width: 100% !important;
    border-collapse: collapse !important;
}
.gr-dataframe th {
    background: #f8fafc !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 9px 18px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    text-align: left !important;
}
.gr-dataframe td {
    padding: 7px 18px !important;
    color: #334155 !important;
    border-bottom: 1px solid #f1f5f9 !important;
}
.gr-dataframe tr:last-child td { border-bottom: none !important; }
.gr-dataframe tr:hover td { background: #f8fafc !important; }

/* ── Section wrapper padding ── */
.section-wrap {
    padding: 0 32px;
    margin-top: 16px;
}

/* ── Footer ── */
.ph-footer {
    text-align: center;
    padding: 24px 0 36px;
    font-size: 11px;
    color: #cbd5e1;
    letter-spacing: 0.05em;
    border-top: 1px solid #f1f5f9;
    margin-top: 28px;
}

/* ── Divider ── */
.ph-divider {
    height: 1px;
    background: #f1f5f9;
    margin: 0 32px 24px;
}

/* ── Row padding override ── */
.gradio-row { padding: 0 32px !important; gap: 16px !important; }
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────

TOPBAR = """
<div class="ph-topbar">
    <div class="ph-logo-dot">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
    </div>
    <span class="ph-logo-name">PhishGuard</span>
    <span class="ph-logo-version">v2.0</span>
    <div class="ph-topbar-right">BiLSTM · Self-Attention · ENSA Fès 2026</div>
</div>
"""

PAGE_TITLE = """
<div class="ph-page-title">
    <h2>URL Scanner</h2>
    <p>Analyze any URL for phishing threats using deep learning.</p>
</div>
"""

STATS = """
<div class="ph-stats">
    <div class="ph-stat-card">
        <div class="ph-stat-label">Accuracy</div>
        <div class="ph-stat-value">99.6%</div>
        <div class="ph-stat-sub">On test set</div>
    </div>
    <div class="ph-stat-card">
        <div class="ph-stat-label">AUC-ROC</div>
        <div class="ph-stat-value">0.999</div>
        <div class="ph-stat-sub">Near-perfect</div>
    </div>
    <div class="ph-stat-card">
        <div class="ph-stat-label">Training data</div>
        <div class="ph-stat-value">100k</div>
        <div class="ph-stat-sub">URLs labeled</div>
    </div>
    <div class="ph-stat-card">
        <div class="ph-stat-label">Features</div>
        <div class="ph-stat-value">18</div>
        <div class="ph-stat-sub">Structural signals</div>
    </div>
</div>
"""

INPUT_HEADER = """
<div class="ph-card-header">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
    Scan target
</div>
"""

RESULT_HEADER = """
<div class="ph-card-header">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    Detection result
</div>
"""

PLACEHOLDER = """
<div class="result-placeholder">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#e2e8f0" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
    <span>Enter a URL above and click Analyze</span>
</div>
"""

EXAMPLES_LABEL = """
<div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; margin:14px 0 8px;">Quick examples</div>
"""

FOOTER = """
<div class="ph-footer">
    PhishTank · Tranco Top 1M &nbsp;·&nbsp; Hybrid BiLSTM + Dense Fusion &nbsp;·&nbsp; ENSA Fès 2026
</div>
"""

EXAMPLES = [
    ["https://www.google.com"],
    ["https://github.com/tensorflow/tensorflow"],
    ["https://www.wikipedia.org/wiki/Phishing"],
    ["http://paypal-secure-login.xyz/verify?user=admin"],
    ["http://amazon-account-update.tk/confirm&session=abc123"],
    ["http://192.168.1.1/login?bank=chase&password=reset"],
]

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="PhishGuard", css=CSS) as demo:

    gr.HTML(TOPBAR)
    gr.HTML(PAGE_TITLE)
    gr.HTML(STATS)
    gr.HTML('<div class="ph-divider"></div>')

    with gr.Row(equal_height=False):

        # Left — input card
        with gr.Column(scale=5, min_width=320):
            with gr.Group(elem_classes="ph-card"):
                gr.HTML(INPUT_HEADER)
                url_input = gr.Textbox(
                    label="",
                    placeholder="https://example.com/path?query=value",
                    elem_id="url-box",
                    lines=1,
                    show_label=False,
                )
                analyze_btn = gr.Button(
                    "Analyze URL",
                    variant="primary",
                    elem_id="analyze-btn",
                    size="lg",
                )
                gr.HTML(EXAMPLES_LABEL)
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=url_input,
                    label="",
                    elem_id="examples-holder",
                )

        # Right — result card
        with gr.Column(scale=5, min_width=320):
            with gr.Group(elem_classes="ph-card"):
                gr.HTML(RESULT_HEADER)
                result_out = gr.HTML(value=PLACEHOLDER)

    # Features accordion
    with gr.Row():
        with gr.Column():
            with gr.Accordion("18 structural signals extracted from the URL", open=False):
                feat_table = gr.Dataframe(
                    headers=["Feature", "Value"],
                    label="",
                    wrap=False,
                )

    gr.HTML(FOOTER)

    analyze_btn.click(
        fn=predict_url,
        inputs=url_input,
        outputs=[result_out, feat_table],
    )
    url_input.submit(
        fn=predict_url,
        inputs=url_input,
        outputs=[result_out, feat_table],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
