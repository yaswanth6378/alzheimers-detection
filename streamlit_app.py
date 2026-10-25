"""Streamlit demo interface for the Alzheimer's MRI classification project.

This is a lightweight inference interface that reuses the trained Keras models
from the original Flask project. It does not include the original Flask login,
patient history, Doctor Dashboard, Admin Panel, or persistent database.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]
MODEL_FILES = {
    "ResNet50": BASE_DIR / "resnet50_alzheimer.keras",
    "Custom CNN": BASE_DIR / "custom_cnn_alzheimer.keras",
}
# Same public model file ID used by the existing Flask application's fallback.
RESNET50_GDRIVE_ID = "133sc-4AOEn79U1QTL_SNgK6iGvmXVAkg"
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]

st.set_page_config(
    page_title="Alzheimer's MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.05rem; font-weight: 750; margin-bottom: 0.2rem; }
    .sub-title { color: #65748b; margin-top: 0; }
    .small-note { color: #65748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_real_model_file(path: Path) -> bool:
    """Return False for missing, tiny, or Git-LFS pointer files."""
    try:
        if not path.is_file() or path.stat().st_size < 1000:
            return False
        with path.open("rb") as model_file:
            header = model_file.read(160)
        return b"version https://git-lfs.github.com/spec/v1" not in header
    except OSError:
        return False


def resnet_download_path() -> Path:
    """Use temporary cache storage if the repository model is absent/LFS pointer."""
    cache_dir = Path(tempfile.gettempdir()) / "alzheimer_model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "resnet50_alzheimer.keras"


@st.cache_resource(show_spinner=False)
def load_keras_model(model_label: str, model_file: str):
    """Load a model once per Streamlit process; download ResNet50 if needed."""
    path = Path(model_file)
    if not is_real_model_file(path):
        if model_label != "ResNet50":
            raise FileNotFoundError(
                "The Custom CNN model file is not available as a complete .keras file. "
                "Check that custom_cnn_alzheimer.keras is committed to the repository "
                "and that Git LFS has uploaded the full file."
            )
        path = resnet_download_path()
        if not is_real_model_file(path):
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("Missing dependency 'gdown'. Add it to requirements.txt.") from exc
            url = f"https://drive.google.com/uc?id={RESNET50_GDRIVE_ID}"
            downloaded_path = gdown.download(url, str(path), quiet=False)
            if not downloaded_path or not is_real_model_file(path):
                raise RuntimeError(
                    "Could not download the ResNet50 model from its configured Google Drive file. "
                    "Check that the Drive file is accessible to anyone with the link, or add the "
                    "full model file to the repository using Git LFS."
                )

    # Import TensorFlow only when the user actually requests a prediction.
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(str(path), compile=False)


def image_to_model_input(image: Image.Image) -> np.ndarray:
    """Match the preprocessing used by the existing Flask app (224x224, /255)."""
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def to_probabilities(model_output: np.ndarray) -> np.ndarray:
    """Convert four model outputs to normalized probabilities if needed."""
    scores = np.asarray(model_output, dtype=np.float64).reshape(-1)
    if scores.size != len(CLASS_NAMES):
        raise ValueError(
            f"The selected model returned {scores.size} outputs; expected {len(CLASS_NAMES)} classes."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned non-finite prediction values.")
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=0.05):
        probabilities = scores
    else:
        # Stable softmax for models that return logits rather than probabilities.
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum()
    return probabilities


st.markdown('<div class="main-title">🧠 Alzheimer’s MRI Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">CNN and ResNet50-based brain MRI image classification demo</p>',
    unsafe_allow_html=True,
)
st.caption("Academic project | Four dataset categories | Streamlit demonstration interface")

with st.sidebar:
    st.header("Project details")
    st.write("**Student:** Uppalapalli Yaswanth Reddy")
    st.write("**Enrollment:** 23SE02ML212")
    st.write("**Program:** MLAI 7C")
    st.divider()
    st.subheader("Model selection")
    available_models = ["ResNet50"]
    if is_real_model_file(MODEL_FILES["Custom CNN"]):
        available_models.append("Custom CNN")
    selected_model = st.selectbox("Choose a trained model", available_models)
    if "Custom CNN" not in available_models:
        st.caption(
            "Custom CNN selection is hidden because its complete model file was not detected. "
            "ResNet50 can be loaded from the configured model backup if needed."
        )
    st.divider()
    st.caption("Images are processed for prediction in this session; this demo does not save scan history.")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("1. Upload an MRI image")
    uploaded_file = st.file_uploader(
        "Select a brain MRI image",
        type=ALLOWED_TYPES,
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, and WEBP.",
    )
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_image, caption="Uploaded MRI image", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            uploaded_image = None
            st.error("This file could not be opened as an image. Please upload a valid MRI image.")
    else:
        uploaded_image = None
        st.info("Upload an MRI image to enable classification.")

with right:
    st.subheader("2. Generate prediction")
    st.write(f"**Selected model:** {selected_model}")
    st.write("**Expected input:** RGB image resized to 224 × 224 pixels")
    st.write("**Possible categories:**")
    for category in CLASS_NAMES:
        st.write(f"• {category}")
    run_prediction = st.button(
        "Classify MRI",
        type="primary",
        use_container_width=True,
        disabled=uploaded_image is None,
    )

if run_prediction and uploaded_image is not None:
    model_file = str(MODEL_FILES[selected_model])
    try:
        with st.spinner("Loading the trained model and analysing the image..."):
            model = load_keras_model(selected_model, model_file)
            model_input = image_to_model_input(uploaded_image)
            raw_output = model.predict(model_input, verbose=0)
            probabilities = to_probabilities(raw_output)
        best_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[best_index]
        confidence = float(probabilities[best_index] * 100.0)

        st.divider()
        st.subheader("3. Prediction result")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Predicted category", predicted_class)
        metric_col2.metric("Model confidence score", f"{confidence:.2f}%")

        result_df = pd.DataFrame(
            {"Category": CLASS_NAMES, "Score (%)": probabilities * 100.0}
        ).sort_values("Score (%)", ascending=False)
        st.write("**Scores across all four categories**")
        st.bar_chart(result_df, x="Category", y="Score (%)", horizontal=True)
        st.dataframe(
            result_df.assign(**{"Score (%)": result_df["Score (%)"].map(lambda x: f"{x:.2f}%")}),
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "This is an educational research prototype. The confidence score is the model's output for this image, "
            "not the model's overall accuracy or a medically validated diagnosis. Consult a qualified clinician "
            "for medical interpretation."
        )
    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.caption(
            "If the error mentions model loading or memory, confirm that the .keras file is complete and "
            "that the hosting service has enough RAM."
        )

st.divider()
st.markdown("**Project repository:** [GitHub — alzheimers-detection](https://github.com/yaswanth6378/alzheimers-detection)")
st.caption("For educational demonstration only. This tool is not intended for clinical diagnosis or treatment decisions.")
