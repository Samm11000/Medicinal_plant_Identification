import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# 🧠 Load model once
# -------------------------------
MODEL_PATH = r"./Models/medicinal_plant_model.keras"
DATA_DIR = r"./Database/data"

@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# -------------------------------
# 🌿 Load class labels
# -------------------------------
class_names = sorted(os.listdir(DATA_DIR))
inv_class_indices = {i: name for i, name in enumerate(class_names)}

# -------------------------------
# 🖼️ Streamlit UI
# -------------------------------
st.set_page_config(page_title="Medicinal Plant Identifier 🌿", layout="centered")

st.title("🌱 Medicinal Plant Identification using MobileNet")
st.markdown("Upload a **leaf image** of an Indian medicinal plant to identify it.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_norm = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)

    # Prediction
    preds = model.predict(img_expanded)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    predicted_label = inv_class_indices[pred_idx]

    # Display result
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    st.success(f"🌿 Predicted Plant: **{predicted_label}**")
    st.info(f"✨ Confidence: **{confidence*100:.2f}%**")
else:
    st.warning("Please upload a leaf image to identify the plant.")
