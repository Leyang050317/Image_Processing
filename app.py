"""Streamlit application for Liwen's banana-ripeness classifier."""

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
LIWEN_FOLDER = PROJECT_ROOT / "liwen"
MODEL_PATH = PROJECT_ROOT / "deep_learning" / "models" / "lw_final.keras"

# Liwen's pipeline uses local imports such as ``from resize import ...``.
# Add only that folder so the existing preprocessing modules can be reused.
if str(LIWEN_FOLDER) not in sys.path:
    sys.path.insert(0, str(LIWEN_FOLDER))

from pipeline import preprocess_image  # noqa: E402


CLASS_NAMES = ("Overripe", "Ripe", "Rotten", "Unripe")
MODEL_IMAGE_SIZE = (224, 224)


@st.cache_resource(show_spinner="Loading Liwen's trained model...")
def load_model():
    """Load the trained model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare_uploaded_image(uploaded_file):
    """Decode an uploaded image and return RGB and BGR representations."""
    rgb_image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.asarray(rgb_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return rgb_image, bgr_array


def predict_ripeness(model, bgr_image):
    """Apply Liwen preprocessing and predict one of the four ripeness classes."""
    processed_bgr, surface_features = preprocess_image(bgr_image)

    # Training files were written by OpenCV and decoded by TensorFlow as RGB.
    # Reproduce that channel order when passing an array directly to the model.
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    model_input = cv2.resize(
        processed_rgb,
        MODEL_IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    model_input = np.expand_dims(model_input.astype(np.float32), axis=0)

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return (
        CLASS_NAMES[predicted_index],
        float(probabilities[predicted_index]),
        probabilities,
        processed_rgb,
        surface_features,
    )


def render_probability_table(probabilities):
    """Show all class confidence scores in training class order."""
    rows = [
        {"Ripeness": name, "Confidence": float(probability)}
        for name, probability in zip(CLASS_NAMES, probabilities)
    ]
    st.dataframe(
        rows,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            )
        },
        hide_index=True,
        use_container_width=True,
    )


def main():
    st.set_page_config(
        page_title="Banana Ripeness Detection",
        page_icon="🍌",
        layout="centered",
    )

    st.title("🍌 Banana Ripeness Detection")
    st.write(
        "Upload a raw banana image to classify it as **Ripe**, **Unripe**, "
        "**Overripe**, or **Rotten** using Liwen's preprocessing pipeline and "
        "MobileNetV2 model."
    )
    st.caption("Liwen model test accuracy: 98.64%")

    uploaded_file = st.file_uploader(
        "Upload a banana image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="For the clearest result, use one banana image with good lighting.",
    )

    if uploaded_file is None:
        st.info("Choose an image above to begin.")
        return

    try:
        original_image, bgr_image = prepare_uploaded_image(uploaded_file)
    except Exception as error:
        st.error(f"The uploaded file could not be read as an image: {error}")
        return

    with st.spinner("Preprocessing the image and detecting ripeness..."):
        try:
            model = load_model()
            (
                predicted_class,
                confidence,
                probabilities,
                processed_rgb,
                surface_features,
            ) = predict_ripeness(model, bgr_image)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

    st.success(f"Prediction: {predicted_class}")
    st.metric("Confidence", f"{confidence:.2%}")

    original_column, processed_column = st.columns(2)
    with original_column:
        st.subheader("Raw image")
        st.image(original_image, use_container_width=True)
    with processed_column:
        st.subheader("Liwen preprocessed")
        st.image(processed_rgb, use_container_width=True)

    st.subheader("Class confidence")
    render_probability_table(probabilities)

    with st.expander("Surface-analysis details"):
        st.json(surface_features)


if __name__ == "__main__":
    main()
