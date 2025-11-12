import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------------------------
# ✅ Streamlit Page Config
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
SAMPLE_DIR = "./Sample_Test"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

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
# 🧩 Prediction Function
# -------------------------------------------------
def predict_image(image_rgb):
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_norm = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)

    with st.spinner("🔍 Identifying plant... please wait..."):
        preds = model.predict(img_expanded)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)
        predicted_label = inv_class_indices[pred_idx]
    return predicted_label, confidence

# -------------------------------------------------
# 🧹 Session State
# -------------------------------------------------
if "image_rgb" not in st.session_state:
    st.session_state.image_rgb = None
    st.session_state.predicted_label = None
    st.session_state.confidence = None
if "sample_selected" not in st.session_state:
    st.session_state.sample_selected = False

# -------------------------------------------------
# 🖼️ UI Layout
# -------------------------------------------------
st.title("🌱 Medicinal Plant Identification")
st.markdown(
    "Upload a **leaf image** below or click **'See Sample Images'** to test with preloaded samples."
)

# -------------------------------------------------
# 📤 Upload Section (Now First)
# -------------------------------------------------
st.markdown("### 📤 Upload Your Image")

col_upload, col_clear = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])


# -------------------------------------------------
# 🚀 Handle Upload or Clear
# -------------------------------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state.sample_selected = False
elif uploaded_file is None and not st.session_state.sample_selected:
    # Only clear if no upload AND no sample selected
    st.session_state.image_rgb = None
    st.session_state.predicted_label = None
    st.session_state.confidence = None

# -------------------------------------------------
# 🔍 Prediction & Display
# -------------------------------------------------
if st.session_state.image_rgb is not None:
    predicted_label, confidence = predict_image(st.session_state.image_rgb)
    st.session_state.predicted_label = predicted_label
    st.session_state.confidence = confidence

    # Centered image display
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.image(st.session_state.image_rgb, caption="📷 Input Leaf", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction output
    if confidence * 100 < 60:
        st.warning("⚠️ **Prediction Confidence is below 60%.**")
        st.error("🌿 **Predicted Plant:** Not Defined")
        st.info(f"✨ **Confidence:** {confidence*100:.2f}%")
    else:
        st.success(f"🌿 **Predicted Plant:** {predicted_label}")
        st.info(f"✨ **Confidence:** {confidence*100:.2f}%")
else:
    st.info("📸 Upload or select a sample image to start.")

# -------------------------------------------------
# 🌿 Sample Images (Collapsible)
# -------------------------------------------------
st.markdown("---")
st.subheader("🌿 Sample Images")

SAMPLES = {
    "Mint": "./Sample_Test/mint1.jpeg",
    "Rasna": "./Sample_Test/Rasna1.png",
    "Jamun": "./Sample_Test/Jamun1.png",
    "Tulsi": "./Sample_Test/tulsi.png"
}

with st.expander("📁 See Sample Images"):
    cols = st.columns(4)
    for i, (name, path) in enumerate(SAMPLES.items()):
        if os.path.exists(path):
            with cols[i]:
                try:
                    st.image(path, caption=name, use_container_width=True)
                except TypeError:
                    st.image(path, caption=name, use_column_width=True)
                if st.button(f"Use {name}", key=name):
                    image = cv2.imread(path)
                    st.session_state.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.session_state.sample_selected = True
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
        else:
            with cols[i]:
                st.warning(f"⚠️ {name} image not found")

# -------------------------------------------------
# 🌐 Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        👨‍💻 Developed by <b>Shaik Shasha Vali</b><br><br>
        <a href="https://www.linkedin.com/in/shasha-vali-ab539428a/" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="30px"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
