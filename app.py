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
SAMPLE_DIR = "./Sample_Test"  # Folder for sample images

@st.cache_resource
def load_trained_model():
    """Load the trained model safely with caching."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    model = load_model(MODEL_PATH)
    return model

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
st.title("🌱 Medicinal Plant Identification")
st.markdown(
    "Upload a **leaf image** or choose a **sample image** below to identify the medicinal plant using a trained deep learning model."
)

# -------------------------------------------------
# 📤 Manual Upload Section + Clear Button
# -------------------------------------------------
st.markdown("---")
st.markdown("### 📤 Upload Image ")

col_upload, col_clear = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])



# -------------------------------------------------
# 🧪 Sample Image Section
# -------------------------------------------------
SAMPLES = {
    "Mint": "./Sample_Test/mint1.jpeg",
    "Rasna": "./Sample_Test/Rasna1.png",
    "Jamun": "./Sample_Test/Jamun1.png",
    "Tulsi": "./Sample_Test/tulsi.png"
}

st.subheader("🌿 Try with Sample Images")

cols = st.columns(4)
selected_sample = None

for i, (name, path) in enumerate(SAMPLES.items()):
    if os.path.exists(path):
        with cols[i]:
            # ✅ Compatibility-safe image display
            try:
                st.image(path, caption=name, use_container_width=True)
            except TypeError:
                st.image(path, caption=name, use_column_width=True)
            if st.button(f"Use {name}", key=name):
                selected_sample = path
    else:
        with cols[i]:
            st.warning(f"⚠️ {name} image not found")


# -------------------------------------------------
# 🧩 Prediction Function
# -------------------------------------------------
def predict_image(image_rgb):
    """Run model prediction on an input image."""
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
# 🧹 Session State Initialization
# -------------------------------------------------
if "image_rgb" not in st.session_state:
    st.session_state.image_rgb = None
    st.session_state.predicted_label = None
    st.session_state.confidence = None

# -------------------------------------------------
# 🚀 Input Handling (Upload or Sample)
# -------------------------------------------------
# Handle sample image selection
if selected_sample:
    image = cv2.imread(selected_sample)
    st.session_state.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Handle file upload
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ✅ Auto-clear output if image removed
elif uploaded_file is None and not selected_sample:
    st.session_state.image_rgb = None
    st.session_state.predicted_label = None
    st.session_state.confidence = None

# -------------------------------------------------
# 🔍 Prediction and Display
# -------------------------------------------------
if st.session_state.image_rgb is not None:
    predicted_label, confidence = predict_image(st.session_state.image_rgb)
    st.session_state.predicted_label = predicted_label
    st.session_state.confidence = confidence

    # 🖼️ Display smaller & centered image
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.image(st.session_state.image_rgb, caption="📷 Input Leaf", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # 🌿 Display prediction result
    if confidence * 100 < 60:
        st.warning("⚠️ **Prediction Confidence is below 60%.**")
        st.error("🌿 **Predicted Plant:** Not Defined")
        st.info(f"✨ **Confidence:** {confidence*100:.2f}%")
    else:
        st.success(f"🌿 **Predicted Plant:** {predicted_label}")
        st.info(f"✨ **Confidence:** {confidence*100:.2f}%")

else:
    st.info("📸 Select a sample image or upload your own to begin.")

# -------------------------------------------------
# 🌐 Footer with LinkedIn
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
