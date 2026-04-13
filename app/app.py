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
        label    = "Safe"
        verdict  = "This URL appears legitimate and trustworthy."
        accent   = "#16a34a"
        bg       = "#f0fdf4"
        border   = "#bbf7d0"
        icon_svg = (
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
            'stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            '<polyline points="9 12 11 14 15 10"/></svg>'
        )
    elif prob < 0.7:
        label    = "Suspicious"
        verdict  = "Proceed with caution — this URL shows unusual patterns."
        accent   = "#b45309"
        bg       = "#fffbeb"
        border   = "#fde68a"
        icon_svg = (
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
            'stroke="#b45309" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/>'
            '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        )
    else:
        label    = "Phishing"
        verdict  = "High risk — this URL is likely malicious. Do not visit."
        accent   = "#dc2626"
        bg       = "#fef2f2"
        border   = "#fecaca"
        icon_svg = (
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
            'stroke="#dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="15" y1="9" x2="9" y2="15"/>'
            '<line x1="9" y1="9" x2="15" y2="15"/></svg>'
        )

    result_html = f"""
<div style="font-family:'Inter',system-ui,sans-serif; padding:2px;">

  <div style="
    display:flex; align-items:center; gap:10px;
    background:{bg}; border:1px solid {border};
    border-radius:10px; padding:14px 18px; margin-bottom:14px;
  ">
    {icon_svg}
    <span style="font-size:15px; font-weight:600; color:{accent};">{label}</span>
    <span style="
      margin-left:auto; font-size:12px; font-weight:600;
      color:{accent}; background:white; border:1px solid {border};
      border-radius:20px; padding:3px 10px;
    ">{pct}%</span>
  </div>

  <div style="
    background:#f8f9fa; border:1px solid #e9ecef; border-radius:8px;
    padding:9px 13px; margin-bottom:12px;
    font-family:'JetBrains Mono','Fira Code',monospace;
    font-size:12px; color:#495057; word-break:break-all;
  ">{url}</div>

  <div style="background:#e9ecef; border-radius:99px; height:5px; overflow:hidden; margin-bottom:12px;">
    <div style="
      width:{pct}%; height:100%; background:{accent};
      border-radius:99px;
    "></div>
  </div>

  <p style="margin:0; font-size:13px; color:#6c757d; line-height:1.6;">{verdict}</p>

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
  font-family:'Inter',system-ui,sans-serif;
  background:#fef2f2; border:1px solid #fecaca;
  border-radius:10px; padding:14px 18px;
  font-size:13px; color:#dc2626;
">{msg}</div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #ffffff !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #1a1a1a !important;
}

/* Remove default Gradio chrome */
footer { display: none !important; }
.gr-block.gr-box, .gr-form { box-shadow: none !important; }

/* ── Header ── */
.ph-header {
    padding: 36px 0 28px;
    text-align: center;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 28px;
}
.ph-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #6c757d;
    background: #f5f5f5;
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 14px;
    text-transform: uppercase;
}
.ph-title {
    font-size: 26px;
    font-weight: 600;
    color: #111;
    margin: 0 0 6px;
    letter-spacing: -0.4px;
}
.ph-sub {
    font-size: 13px;
    color: #999;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Stats ── */
.ph-stats {
    display: flex;
    justify-content: center;
    gap: 0;
    border-top: 1px solid #f0f0f0;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 32px;
    background: #fafafa;
}
.ph-stat {
    padding: 16px 36px;
    text-align: center;
    border-right: 1px solid #f0f0f0;
}
.ph-stat:last-child { border-right: none; }
.ph-stat-val {
    display: block;
    font-size: 20px;
    font-weight: 600;
    color: #111;
    line-height: 1.2;
}
.ph-stat-lbl {
    display: block;
    font-size: 11px;
    color: #aaa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Section labels ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 10px;
    display: block;
}

/* ── Input ── */
#url-box textarea, #url-box input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #fff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    color: #111 !important;
    padding: 11px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
#url-box textarea:focus, #url-box input:focus {
    border-color: #111 !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.06) !important;
    outline: none !important;
}
#url-box label span {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #aaa !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
#analyze-btn {
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 0 !important;
    transition: background 0.15s, transform 0.1s !important;
    letter-spacing: 0.01em !important;
}
#analyze-btn:hover {
    background: #333 !important;
    transform: translateY(-1px) !important;
}
#analyze-btn:active {
    transform: translateY(0) !important;
}

/* ── Examples ── */
.examples-holder table td button,
.gr-samples-table td button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    background: #fff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    color: #555 !important;
    padding: 4px 10px !important;
    transition: all 0.12s !important;
}
.examples-holder table td button:hover,
.gr-samples-table td button:hover {
    background: #f5f5f5 !important;
    border-color: #ccc !important;
    color: #111 !important;
}

/* ── Result placeholder ── */
.result-placeholder {
    border: 1px dashed #e0e0e0;
    border-radius: 10px;
    padding: 48px 24px;
    text-align: center;
    color: #ccc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* ── Accordion ── */
.gradio-accordion {
    background: #fff !important;
    border: 1px solid #efefef !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.gradio-accordion > .label-wrap button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #555 !important;
    padding: 13px 18px !important;
}

/* ── Dataframe ── */
.gr-dataframe table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
.gr-dataframe th {
    background: #fafafa !important;
    color: #999 !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 8px 16px !important;
    border-bottom: 1px solid #f0f0f0 !important;
}
.gr-dataframe td {
    padding: 7px 16px !important;
    color: #333 !important;
    border-bottom: 1px solid #f9f9f9 !important;
}

/* ── Footer ── */
.ph-footer {
    text-align: center;
    padding: 28px 0 36px;
    font-size: 11px;
    color: #ccc;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
}
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────

HEADER = """
<div class="ph-header">
    <div class="ph-tag">Deep Learning · NLP · ENSA Fès</div>
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
    <div class="ph-stat">
        <span class="ph-stat-val">0.999</span>
        <span class="ph-stat-lbl">AUC-ROC</span>
    </div>
    <div class="ph-stat">
        <span class="ph-stat-val">100k</span>
        <span class="ph-stat-lbl">URLs trained</span>
    </div>
    <div class="ph-stat">
        <span class="ph-stat-val">18</span>
        <span class="ph-stat-lbl">Features</span>
    </div>
</div>
"""

PLACEHOLDER = """
<div class="result-placeholder">
    Enter a URL and click Analyze
</div>
"""

FOOTER = """
<div class="ph-footer">
    PhishTank · Tranco Top 1M · Hybrid BiLSTM + Dense Fusion · ENSA Fès 2026
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
                "Analyze →",
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

        with gr.Column(scale=5, min_width=320):
            gr.HTML('<span class="section-label">Detection result</span>')
            result_out = gr.HTML(value=PLACEHOLDER)

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
