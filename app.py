import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------------------------
# ✅ Streamlit Page Config (must be first)
# -------------------------------------------------
st.set_page_config(
    page_title="Medicinal Plant Identifier 🌿",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------------------------
# 🧠 Load Model Safely (cached)
# -------------------------------------------------
MODEL_PATH = "./Models/medicinal_plant_model.keras"
DATA_DIR = "./Database/data"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    model = load_model(MODEL_PATH)
    return model

# Try loading model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# -------------------------------------------------
# 🌿 Load Class Labels
# -------------------------------------------------
if not os.path.exists(DATA_DIR):
    st.error(f"❌ Data folder not found at: {DATA_DIR}")
    st.stop()

class_names = sorted(os.listdir(DATA_DIR))
inv_class_indices = {i: name for i, name in enumerate(class_names)}

# -------------------------------------------------
# 🖼️ Streamlit UI
# -------------------------------------------------
st.title("🌱 Medicinal Plant Identification By Shasha Vali")
st.markdown(
    "Upload a **leaf image** of an Indian medicinal plant to identify it using a trained deep learning model."
)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_rgb, (224, 224))
        img_norm = img_resized.astype("float32") / 255.0
        img_expanded = np.expand_dims(img_norm, axis=0)

        # Run model prediction with spinner
        with st.spinner("🔍 Identifying plant... please wait..."):
            preds = model.predict(img_expanded)
            pred_idx = np.argmax(preds, axis=1)[0]
            confidence = np.max(preds)
            predicted_label = inv_class_indices[pred_idx]

        # Display uploaded image
        st.image(image_rgb, caption="📷 Uploaded Leaf", use_column_width=True)

        # 🔸 Confidence Threshold Check (Below 60% = Not Defined)
        if confidence * 100 < 60:
            st.warning("⚠️ **Prediction Confidence is below 60%.**")
            st.error("🌿 **Predicted Plant:** Not Defined")
            st.info(f"✨ **Confidence:** {confidence*100:.2f}%")
        else:
            st.success(f"🌿 **Predicted Plant:** {predicted_label}")
            st.info(f"✨ **Confidence:** {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

else:
    st.warning("📥 Please upload a leaf image to identify the plant.")
