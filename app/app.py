import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE, MODELS_DIR
from src.gradcam import generate_gradcam, get_last_conv_layer, overlay_gradcam

st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: #0B0E14;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: #11141C;
        border-right: 1px solid #232838;
    }

    section[data-testid="stSidebar"] * {
        color: #C7CCDA !important;
    }

    section[data-testid="stSidebar"] h1 {
        color: #00E5A0 !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: #F4F6FB !important;
    }

    .hero-card {
        background: linear-gradient(135deg, #151A26 0%, #0F1420 100%);
        border: 1px solid #232838;
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }

    .hero-card h1 {
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
    }

    .hero-card p {
        color: #8A93A8;
        font-size: 1rem;
        margin: 0;
    }

    .pred-card {
        background: #12161F;
        border: 1px solid #232838;
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
    }

    .pred-card.top {
        border: 1px solid #00E5A0;
        background: linear-gradient(135deg, #10231C 0%, #12161F 70%);
    }

    .pred-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #F4F6FB;
    }

    .pred-score {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6E7690;
        margin-top: 0.1rem;
    }

    .pred-card.top .pred-score {
        color: #00E5A0;
    }

    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #00E5A0, #00B4D8);
    }

    .metric-pill {
        display: inline-block;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }

    .metric-pill.primary {
        background: rgba(0, 229, 160, 0.12);
        color: #00E5A0;
        border: 1px solid rgba(0, 229, 160, 0.35);
    }

    .metric-pill.secondary {
        background: rgba(255, 87, 179, 0.12);
        color: #FF57B3;
        border: 1px solid rgba(255, 87, 179, 0.35);
    }

    div[data-testid="stFileUploader"] {
        background: #12161F;
        border: 1px dashed #333B4F;
        border-radius: 16px;
        padding: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00E5A0, #00B4D8);
        color: #08111A;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 2rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #232838;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(f"{MODELS_DIR}/food101_mobilenetv2_finetuned.keras")
    with open(f"{MODELS_DIR}/class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_resource
def get_base_model(_model):
    for layer in _model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested base model found inside the loaded model.")


def preprocess(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.array(image).astype("float32")
    return array


def predict(model, class_names, array, top_k=5):
    batch = np.expand_dims(array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_indices]


with st.sidebar:
    st.title("🍔 Food-101 AI")
    st.markdown("A fine-tuned MobileNetV2 classifier trained on 101 food categories.")
    st.markdown("---")
    top_k = st.slider("Top-K Predictions", min_value=3, max_value=10, value=5)
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    st.markdown("---")
    st.markdown("Built by **Om Mishra**")
    st.markdown("[GitHub](https://github.com/omxmishra) · [LinkedIn](https://linkedin.com/in/om--mishra)")

st.markdown(
    """
    <div class="hero-card">
        <h1>🍽️ Food-101 Image Classifier</h1>
        <p>Upload a food photo and let the fine-tuned MobileNetV2 model tell you what it sees, complete with Grad-CAM visual explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, class_names = load_model()

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    array = preprocess(image)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    predictions = predict(model, class_names, array, top_k=top_k)
    top_label, top_score = predictions[0]

    with col2:
        st.markdown(
            f"""
            <div class="metric-pill primary">🏆 {top_label.replace('_', ' ').title()}</div>
            <div class="metric-pill secondary">🔥 {top_score * 100:.1f}% confidence</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Top Predictions")

        for i, (label, score) in enumerate(predictions):
            css_class = "pred-card top" if i == 0 else "pred-card"
            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="pred-label">{label.replace('_', ' ').title()}</div>
                    <div class="pred-score">{score * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(score, 1.0))

    if show_gradcam:
        st.markdown("### 🔍 Grad-CAM Explanation")
        try:
            base_model = get_base_model(model)
            heatmap = generate_gradcam(model, base_model, array)
            overlay = overlay_gradcam(array.astype("uint8"), heatmap, IMAGE_SIZE)
            st.image(overlay, caption="Model attention heatmap", use_container_width=False, width=350)
        except Exception as error:
            st.warning(f"Grad-CAM could not be generated: {error}")
else:
    st.info("Upload an image above to get started.")