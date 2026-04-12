"""
app.py
------
Interface de démonstration Gradio pour la détection de phishing en temps réel.

Usage:
    python app/app.py
    → Ouvre http://localhost:7860 sur le localhost
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

    # Préparer les entrées
    df = pd.DataFrame({"url": [url], "label": [0]})
    X_seq, X_feat, _ = pipeline.transform(df)

    # Prédiction
    prob = float(model.predict([X_seq, X_feat], verbose=0)[0][0])

    # Risk level
    if prob < 0.3:
        risk = "🟢 LÉGITIME"
        color = "green"
        advice = "Cette URL semble sûre."
    elif prob < 0.7:
        risk = "🟠 SUSPECT"
        color = "orange"
        advice = "Soyez prudent avec cette URL."
    else:
        risk = "🔴 PHISHING"
        color = "red"
        advice = "⚠️ Cette URL est probablement malveillante !"

    # Features tableau
    features = extract_url_features(url)
    feat_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Valeur": [round(v, 4) for v in features]
    })

    result_html = f"""
    <div style="padding:20px; border-radius:10px; border: 2px solid {color}; text-align:center;">
        <h2 style="color:{color}; margin:0">{risk}</h2>
        <p style="font-size:1.2em; margin:10px 0">Score de phishing : <b>{prob:.1%}</b></p>
        <p style="color:gray">{advice}</p>
    </div>
    """

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
.gradio-container { font-family: 'Segoe UI', sans-serif; }
#url-input { font-size: 1.1em; }
"""

with gr.Blocks(title="🎣 Phishing Detector", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎣 Détecteur de Phishing par Deep Learning
    ### Modèle hybride : BiLSTM + Analyse structurelle d'URL
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="🔗 URL à analyser",
                placeholder="Ex: http://paypal-secure-login.xyz/verify",
                elem_id="url-input",
                lines=1,
            )
            analyze_btn = gr.Button("🔍 Analyser", variant="primary", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=url_input,
                label="📌 Exemples",
            )

        with gr.Column(scale=1):
            result_html = gr.HTML(label="Résultat")

    with gr.Accordion("📊 Features URL extraites", open=False):
        feat_table = gr.Dataframe(
            headers=["Feature", "Valeur"],
            label="18 features structurelles",
        )

    gr.Markdown("""
    ---
    **Architecture** : BiLSTM (128 units) + Self-Attention + Features Branch (128→64) + Fusion Layer  
    **Données** : PhishTank (phishing vérifiés) + Alexa Top 1M (légitimes)  
    **ENSA de Fès** — Module Deep Learning & NLP
    """)

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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
