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
        status   = "SAFE"
        verdict  = "This URL appears legitimate and trustworthy."
        hue      = "#16a34a"
        bg       = "#f0fdf4"
        ring     = "#bbf7d0"
        bar_bg   = "#dcfce7"
        icon_svg = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>"""
    elif prob < 0.7:
        status   = "SUSPICIOUS"
        verdict  = "Proceed with caution — this URL shows unusual patterns."
        hue      = "#d97706"
        bg       = "#fffbeb"
        ring     = "#fde68a"
        bar_bg   = "#fef3c7"
        icon_svg = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
    else:
        status   = "PHISHING"
        verdict  = "High risk — this URL is likely malicious. Do not visit."
        hue      = "#dc2626"
        bg       = "#fef2f2"
        ring     = "#fecaca"
        bar_bg   = "#fee2e2"
        icon_svg = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""

    result_html = f"""
<div style="
    background:{bg};
    border:1.5px solid {ring};
    border-radius:14px;
    padding:28px 28px 22px;
    font-family:'DM Sans',sans-serif;
">
    <!-- Status row -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        {icon_svg}
        <span style="font-size:1.35rem;font-weight:700;color:{hue};letter-spacing:.02em;">{status}</span>
        <span style="margin-left:auto;font-size:.8rem;font-weight:600;background:{bar_bg};color:{hue};padding:3px 10px;border-radius:999px;">{pct}% risk</span>
    </div>

    <!-- Progress bar -->
    <div style="background:{bar_bg};border-radius:999px;height:6px;margin-bottom:18px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{hue};border-radius:999px;"></div>
    </div>

    <!-- URL display -->
    <div style="
        background:rgba(255,255,255,.7);
        border:1px solid {ring};
        border-radius:8px;
        padding:10px 14px;
        font-family:'JetBrains Mono','Fira Code',monospace;
        font-size:.78rem;
        color:#374151;
        word-break:break-all;
        margin-bottom:14px;
    ">{url}</div>

    <!-- Verdict -->
    <p style="margin:0;font-size:.88rem;color:#6b7280;line-height:1.55;">{verdict}</p>
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
    background:#fef2f2;border:1.5px solid #fecaca;border-radius:14px;
    padding:20px 24px;font-family:'DM Sans',sans-serif;color:#991b1b;font-size:.9rem;
">{msg}</div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #f8f9fb !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Header ── */
.ph-header {
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    padding: 28px 0 24px;
    text-align: center;
    margin-bottom: 32px;
}
.ph-badge {
    display: inline-block;
    background: #111827;
    color: #f9fafb;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .18em;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
    text-transform: uppercase;
}
.ph-title {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px;
    letter-spacing: -.02em;
    line-height: 1.2;
}
.ph-sub {
    font-size: .83rem;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    margin: 0;
    letter-spacing: .04em;
}

/* ── Cards ── */
.ph-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
}

/* ── Input ── */
#url-box textarea, #url-box input {
    font-family: 'DM Mono', monospace !important;
    font-size: .9rem !important;
    background: #f9fafb !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    transition: border-color .18s, box-shadow .18s;
    padding: 12px 14px !important;
}
#url-box textarea:focus, #url-box input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
    background: #ffffff !important;
    outline: none !important;
}
label[for="url-box"] span,
#url-box label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
#analyze-btn {
    background: #111827 !important;
    color: #f9fafb !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
    letter-spacing: .03em !important;
    border: none !important;
    padding: 13px 0 !important;
    transition: background .18s, transform .1s, box-shadow .18s !important;
    box-shadow: 0 2px 6px rgba(0,0,0,.12) !important;
}
#analyze-btn:hover {
    background: #1f2937 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.18) !important;
}
#analyze-btn:active {
    transform: translateY(0) !important;
}

/* ── Examples ── */
.examples-holder table td button,
.gr-samples-table td button {
    font-family: 'DM Mono', monospace !important;
    font-size: .75rem !important;
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    color: #374151 !important;
    padding: 5px 10px !important;
    transition: all .15s !important;
}
.examples-holder table td button:hover,
.gr-samples-table td button:hover {
    background: #ede9fe !important;
    border-color: #c4b5fd !important;
    color: #4f46e5 !important;
}

/* ── Accordion ── */
.gradio-accordion {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
.gradio-accordion > .label-wrap button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .83rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: .03em !important;
    padding: 14px 18px !important;
}

/* ── Dataframe ── */
.gr-dataframe table {
    font-family: 'DM Mono', monospace !important;
    font-size: .78rem !important;
}
.gr-dataframe th {
    background: #f3f4f6 !important;
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: .72rem !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    padding: 8px 14px !important;
}
.gr-dataframe td {
    padding: 7px 14px !important;
    color: #1f2937 !important;
    border-bottom: 1px solid #f3f4f6 !important;
}

/* ── Footer ── */
.ph-footer {
    text-align: center;
    padding: 24px 0 32px;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: #d1d5db;
    letter-spacing: .06em;
}

/* ── Stats bar ── */
.ph-stats {
    display: flex;
    justify-content: center;
    gap: 32px;
    padding: 16px 0;
    margin-bottom: 8px;
}
.ph-stat {
    text-align: center;
}
.ph-stat-val {
    display: block;
    font-size: 1.3rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.ph-stat-lbl {
    display: block;
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    color: #9ca3af;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-top: 2px;
}
.ph-divider {
    width: 1px;
    background: #e5e7eb;
    align-self: stretch;
}

/* ── Section labels ── */
.section-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 10px;
    display: block;
}

/* ── Result placeholder ── */
.result-placeholder {
    border: 1.5px dashed #e5e7eb;
    border-radius: 14px;
    padding: 40px 24px;
    text-align: center;
    color: #d1d5db;
    font-family: 'DM Mono', monospace;
    font-size: .8rem;
    letter-spacing: .06em;
}
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────

HEADER = """
<div class="ph-header">
    <div class="ph-badge">Deep Learning · NLP · ENSA Fès</div>
    <h1 class="ph-title">Phishing URL Detector</h1>
    <p class="ph-sub">BiLSTM + Self-Attention + Structural Features</p>
</div>
"""

STATS = """
<div class="ph-stats">
    <div class="ph-stat">
        <span class="ph-stat-val">99.6%</span>
        <span class="ph-stat-lbl">Accuracy</span>
    </div>
    <div class="ph-divider"></div>
    <div class="ph-stat">
        <span class="ph-stat-val">0.999</span>
        <span class="ph-stat-lbl">AUC-ROC</span>
    </div>
    <div class="ph-divider"></div>
    <div class="ph-stat">
        <span class="ph-stat-val">100k</span>
        <span class="ph-stat-lbl">URLs trained</span>
    </div>
    <div class="ph-divider"></div>
    <div class="ph-stat">
        <span class="ph-stat-val">18</span>
        <span class="ph-stat-lbl">Features</span>
    </div>
</div>
"""

PLACEHOLDER = """
<div class="result-placeholder">
    ↑ &nbsp; Enter a URL above and click Analyze
</div>
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

with gr.Blocks(title="Phishing Detector", css=CSS) as demo:

    gr.HTML(HEADER)
    gr.HTML(STATS)

    with gr.Row(equal_height=False):

        # ── Left column — input ──────────────────────────────────────────────
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<span class="section-label">URL to analyze</span>')
            url_input = gr.Textbox(
                label="",
                placeholder="https://example.com/path?query=value",
                elem_id="url-box",
                lines=1,
                show_label=False,
            )
            analyze_btn = gr.Button(
                "Analyze URL →",
                variant="primary",
                elem_id="analyze-btn",
                size="lg",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=url_input,
                label="Quick examples",
                elem_id="examples-holder",
            )

        # ── Right column — result ────────────────────────────────────────────
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<span class="section-label">Detection result</span>')
            result_out = gr.HTML(value=PLACEHOLDER)

    # ── Feature accordion ────────────────────────────────────────────────────
    with gr.Accordion("18 structural signals extracted from the URL", open=False):
        feat_table = gr.Dataframe(
            headers=["Feature", "Value"],
            label="",
            wrap=False,
        )

    gr.HTML(FOOTER)

    # ── Events ───────────────────────────────────────────────────────────────
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
