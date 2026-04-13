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
        return _error_html("ERR :: please enter a valid URL."), None

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    df = pd.DataFrame({"url": [url], "label": [0]})
    X_seq, X_feat, _ = pipeline.transform(df)
    prob = float(model.predict([X_seq, X_feat], verbose=0)[0][0])
    pct = int(prob * 100)

    if prob < 0.3:
        status  = "SAFE"
        verdict = "Target URL is clean. No threat detected."
        color   = "#00ff41"
        prefix  = "[OK]"
    elif prob < 0.7:
        status  = "SUSPICIOUS"
        verdict = "Anomalous patterns detected. Proceed with caution."
        color   = "#ffb700"
        prefix  = "[!!]"
    else:
        status  = "PHISHING"
        verdict = "THREAT CONFIRMED. Do not visit this URL."
        color   = "#ff3131"
        prefix  = "[XX]"

    bar_filled = int(pct / 5)
    bar_empty  = 20 - bar_filled
    bar        = "█" * bar_filled + "░" * bar_empty

    result_html = f"""
<div style="
  font-family:'Courier New',Courier,monospace;
  background:#0a0a0a;
  border:1px solid #1a1a1a;
  border-left:3px solid {color};
  border-radius:4px;
  padding:20px 22px;
  color:#00ff41;
  font-size:13px;
  line-height:1.8;
">
  <div style="color:#333; margin-bottom:14px; font-size:11px; letter-spacing:0.08em;">
    ┌─ PHISHGUARD SCAN RESULT ────────────────────────────────
  </div>
  <div style="margin-bottom:10px;">
    <span style="color:#2a5c2a;">STATUS  </span>
    <span style="color:{color}; letter-spacing:0.12em;">{prefix} {status}</span>
  </div>
  <div style="margin-bottom:10px;">
    <span style="color:#2a5c2a;">RISK    </span>
    <span style="color:{color};">{bar}  {pct}%</span>
  </div>
  <div style="margin-bottom:14px; word-break:break-all;">
    <span style="color:#2a5c2a;">TARGET  </span>
    <span style="color:#555;">{url}</span>
  </div>
  <div style="border-top:1px solid #1a1a1a; padding-top:12px; color:#3a7a3a; font-size:11px;">
    &gt; {verdict}
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
  font-family:'Courier New',Courier,monospace;
  background:#0a0a0a;
  border:1px solid #1a1a1a;
  border-left:3px solid #ff3131;
  border-radius:4px;
  padding:16px 22px;
  color:#ff3131;
  font-size:13px;
">[ERR] {msg}</div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #080808 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #00ff41 !important;
}

footer { display: none !important; }

/* ── Header ── */
.ph-header {
    padding: 36px 0 30px;
    text-align: center;
    border-bottom: 1px solid #111;
    margin-bottom: 0;
}
.ph-prompt {
    font-size: 11px;
    color: #2a5c2a;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
}
.ph-title {
    font-size: 28px;
    font-weight: 400;
    color: #00ff41;
    margin: 0 0 8px;
    letter-spacing: 0.08em;
    text-shadow: 0 0 20px rgba(0,255,65,0.2);
}
.ph-sub {
    font-size: 11px;
    color: #1e4d1e;
    letter-spacing: 0.12em;
    margin: 0;
}

/* ── Stats ── */
.ph-stats {
    display: flex;
    justify-content: center;
    border-bottom: 1px solid #111;
    background: #080808;
    margin-bottom: 32px;
}
.ph-stat {
    padding: 14px 40px;
    text-align: center;
    border-right: 1px solid #111;
}
.ph-stat:last-child { border-right: none; }
.ph-stat-val {
    display: block;
    font-size: 18px;
    color: #00ff41;
    text-shadow: 0 0 10px rgba(0,255,65,0.25);
}
.ph-stat-lbl {
    display: block;
    font-size: 10px;
    color: #1e4d1e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Section labels ── */
.section-label {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1e4d1e;
    margin-bottom: 8px;
    display: block;
}

/* ── Input ── */
#url-box textarea, #url-box input {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 13px !important;
    background: #0d0d0d !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 4px !important;
    color: #00ff41 !important;
    padding: 12px 14px !important;
    caret-color: #00ff41 !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
#url-box textarea:focus, #url-box input:focus {
    border-color: #00ff41 !important;
    box-shadow: 0 0 0 2px rgba(0,255,65,0.07) !important;
    outline: none !important;
}
#url-box textarea::placeholder, #url-box input::placeholder {
    color: #1a3d1a !important;
}
#url-box label span {
    font-size: 10px !important;
    font-weight: 400 !important;
    color: #1e4d1e !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
#analyze-btn {
    background: transparent !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    letter-spacing: 0.14em !important;
    padding: 12px 0 !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
#analyze-btn:hover {
    background: rgba(0,255,65,0.05) !important;
    box-shadow: 0 0 14px rgba(0,255,65,0.12) !important;
    transform: translateY(-1px) !important;
}
#analyze-btn:active {
    transform: translateY(0) !important;
}

/* ── Examples ── */
.examples-holder table td button,
.gr-samples-table td button {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    background: transparent !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 3px !important;
    color: #2a5c2a !important;
    padding: 4px 10px !important;
    transition: all 0.12s !important;
}
.examples-holder table td button:hover,
.gr-samples-table td button:hover {
    border-color: #00ff41 !important;
    color: #00ff41 !important;
    background: rgba(0,255,65,0.04) !important;
}

/* ── Result placeholder ── */
.result-placeholder {
    border: 1px solid #111;
    border-radius: 4px;
    padding: 48px 24px;
    text-align: center;
    color: #1a3d1a;
    font-size: 12px;
    letter-spacing: 0.1em;
    background: #0d0d0d;
}

/* ── Accordion ── */
.gradio-accordion {
    background: #0d0d0d !important;
    border: 1px solid #151515 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}
.gradio-accordion > .label-wrap button {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    color: #2a5c2a !important;
    letter-spacing: 0.1em !important;
    padding: 13px 18px !important;
}

/* ── Dataframe ── */
.gr-dataframe table {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    background: #0d0d0d !important;
}
.gr-dataframe th {
    background: #0d0d0d !important;
    color: #2a5c2a !important;
    font-weight: 400 !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 16px !important;
    border-bottom: 1px solid #151515 !important;
}
.gr-dataframe td {
    padding: 6px 16px !important;
    color: #00cc35 !important;
    border-bottom: 1px solid #0f0f0f !important;
}

/* ── Footer ── */
.ph-footer {
    text-align: center;
    padding: 24px 0 36px;
    font-size: 10px;
    color: #1a3d1a;
    letter-spacing: 0.1em;
}

/* ── Scanline overlay ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.04) 2px,
        rgba(0,0,0,0.04) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────

HEADER = """
<div class="ph-header">
    <div class="ph-prompt">root@phishguard:~$ ./scan --model hybrid_bilstm --init</div>
    <h1 class="ph-title">PHISHING URL DETECTOR</h1>
    <p class="ph-sub">BiLSTM + Self-Attention + Structural Features // ENSA Fès 2026</p>
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
    &gt; awaiting target URL...
</div>
"""

FOOTER = """
<div class="ph-footer">
    [ PhishTank · Tranco Top 1M · Hybrid BiLSTM + Dense Fusion · ENSA Fès 2026 ]
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
            gr.HTML('<span class="section-label">&gt; target URL</span>')
            url_input = gr.Textbox(
                label="",
                placeholder="https://example.com/path?query=value",
                elem_id="url-box",
                lines=1,
                show_label=False,
            )
            analyze_btn = gr.Button(
                "[ RUN SCAN ]",
                variant="primary",
                elem_id="analyze-btn",
                size="lg",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=url_input,
                label="// quick examples",
                elem_id="examples-holder",
            )

        with gr.Column(scale=5, min_width=320):
            gr.HTML('<span class="section-label">&gt; scan output</span>')
            result_out = gr.HTML(value=PLACEHOLDER)

    with gr.Accordion("// 18 structural signals extracted from URL", open=False):
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
