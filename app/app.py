"""
app.py
------
Interface de démonstration Gradio pour la détection de phishing en temps réel.

Usage:
    python app/app.py
    → Ouvre http://localhost:7860
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

# ─────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────

def load_model_and_pipeline():
    model_path = MODELS_DIR / "hybrid_best.keras"
    if not model_path.exists():
        return None, None, "⚠️ Modèle non trouvé. Lance d'abord : python src/training/train.py"
    model = tf.keras.models.load_model(str(model_path))
    pipeline = FeaturePipeline.load(PIPELINE_DIR)
    return model, pipeline, None


model, pipeline, load_error = load_model_and_pipeline()


# ─────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────

def predict_url(url: str):
    if load_error:
        return load_error, "", None

    if not url or len(url.strip()) < 5:
        return "⚠️ Entrez une URL valide", "", None

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    df = pd.DataFrame({"url": [url], "label": [0]})
    X_seq, X_feat, _ = pipeline.transform(df)
    prob = float(model.predict([X_seq, X_feat], verbose=0)[0][0])

    if prob < 0.3:
        risk_label = "LÉGITIME"
        icon = "✦"
        bar_color = "#4ade80"
        bg_color = "#f0fdf4"
        border_color = "#86efac"
        text_color = "#166534"
        advice = "Cette URL semble sûre et fiable."
    elif prob < 0.7:
        risk_label = "SUSPECT"
        icon = "◈"
        bar_color = "#fb923c"
        bg_color = "#fff7ed"
        border_color = "#fdba74"
        text_color = "#9a3412"
        advice = "Procédez avec prudence."
    else:
        risk_label = "PHISHING"
        icon = "⚠"
        bar_color = "#f43f5e"
        bg_color = "#fff1f2"
        border_color = "#fda4af"
        text_color = "#9f1239"
        advice = "Cette URL est probablement malveillante !"

    pct = int(prob * 100)
    bar_width = pct

    result_html = f"""
    <div style="
        background: {bg_color};
        border: 1.5px solid {border_color};
        border-radius: 16px;
        padding: 28px 32px;
        font-family: 'Georgia', serif;
        animation: fadeIn .4s ease;
    ">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
            <span style="font-size:2rem; color:{bar_color};">{icon}</span>
            <span style="font-size:1.6rem; font-weight:700; color:{text_color}; letter-spacing:.05em;">{risk_label}</span>
        </div>

        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-size:.85rem; color:#6b7280; font-family:'Courier New', monospace;">Score de phishing</span>
                <span style="font-size:.85rem; font-weight:700; color:{text_color}; font-family:'Courier New', monospace;">{prob:.1%}</span>
            </div>
            <div style="background:#e5e7eb; border-radius:999px; height:8px; overflow:hidden;">
                <div style="
                    width:{bar_width}%;
                    height:100%;
                    background: linear-gradient(90deg, {bar_color}cc, {bar_color});
                    border-radius:999px;
                    transition: width .6s ease;
                "></div>
            </div>
        </div>

        <p style="
            margin:12px 0 0;
            font-size:.9rem;
            color:#6b7280;
            font-style:italic;
            font-family:'Georgia', serif;
        ">{advice}</p>
    </div>
    <style>
        @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:none; }} }}
    </style>
    """

    features = extract_url_features(url)
    feat_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Valeur": [round(v, 4) for v in features]
    })

    return result_html, feat_df


# ─────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────

EXAMPLES = [
    ["https://www.google.com"],
    ["http://paypal-secure-login.xyz/verify?user=admin"],
    ["https://github.com/tensorflow/tensorflow"],
    ["http://amazon-account-update.tk/confirm&session=abc123"],
    ["https://www.wikipedia.org/wiki/Deep_learning"],
    ["http://192.168.1.1/login?bank=chase&password=reset"],
]

CSS = """
/* ── Global ── */
body, .gradio-container {
    background: #fafaf8 !important;
    font-family: 'Georgia', serif !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 40px 0 24px;
}

/* ── Input ── */
#url-input textarea, #url-input input {
    font-family: 'Courier New', monospace !important;
    font-size: 1rem !important;
    background: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    transition: border-color .2s;
}
#url-input textarea:focus, #url-input input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px #6366f115 !important;
}

/* ── Button ── */
.analyze-btn {
    background: #111827 !important;
    color: #fafaf8 !important;
    border-radius: 10px !important;
    font-family: 'Courier New', monospace !important;
    font-size: .95rem !important;
    letter-spacing: .08em !important;
    border: none !important;
    transition: background .2s, transform .1s !important;
}
.analyze-btn:hover {
    background: #374151 !important;
    transform: translateY(-1px) !important;
}

/* ── Examples ── */
.examples-holder button {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
    font-size: .78rem !important;
    color: #374151 !important;
    transition: border-color .15s, background .15s !important;
}
.examples-holder button:hover {
    border-color: #6366f1 !important;
    background: #f5f3ff !important;
}

/* ── Accordion ── */
.gradio-accordion {
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

/* ── Footer text ── */
.footer-txt {
    text-align: center;
    font-size: .8rem;
    color: #9ca3af;
    letter-spacing: .04em;
    margin-top: 8px;
}
"""

HEADER_HTML = """
<div class="app-header">
    <div style="
        display: inline-block;
        background: #111827;
        color: #fafaf8;
        font-family: 'Courier New', monospace;
        font-size: .7rem;
        letter-spacing: .2em;
        padding: 4px 14px;
        border-radius: 999px;
        margin-bottom: 16px;
        text-transform: uppercase;
    ">Deep Learning · NLP · ENSA Fès</div>

    <h1 style="
        font-family: 'Georgia', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 8px;
        letter-spacing: -.01em;
    ">Phishing URL Detector</h1>

    <p style="
        font-family: 'Courier New', monospace;
        font-size: .85rem;
        color: #6b7280;
        margin: 0;
        letter-spacing: .04em;
    ">BiLSTM + Self-Attention + Structural Features — Hybrid Model</p>

    <div style="
        width: 48px;
        height: 2px;
        background: #111827;
        margin: 20px auto 0;
        border-radius: 1px;
    "></div>
</div>
"""

FOOTER_HTML = """
<div class="footer-txt">
    <span style="font-family:'Courier New',monospace;">
        Architecture · BiLSTM(128) + MultiHead Attention(4) + Dense Fusion
        &nbsp;·&nbsp; Data · PhishTank + Alexa Top 1M
    </span>
</div>
"""

with gr.Blocks(title="Phishing URL Detector", css=CSS) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            url_input = gr.Textbox(
                label="URL à analyser",
                placeholder="https://example.com/path?query=value",
                elem_id="url-input",
                lines=1,
            )
            analyze_btn = gr.Button(
                "→ Analyser",
                variant="primary",
                size="lg",
                elem_classes=["analyze-btn"],
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=url_input,
                label="Exemples",
                elem_id="examples-holder",
            )

        with gr.Column(scale=2):
            result_html = gr.HTML(label="Résultat")

    with gr.Accordion("Features URL extraites ( 18 signaux structurels )", open=False):
        feat_table = gr.Dataframe(
            headers=["Feature", "Valeur"],
            label="",
        )

    gr.HTML(FOOTER_HTML)

    analyze_btn.click(
        fn=predict_url,
        inputs=url_input,
        outputs=[result_html, feat_table],
    )
    url_input.submit(
        fn=predict_url,
        inputs=url_input,
        outputs=[result_html, feat_table],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
