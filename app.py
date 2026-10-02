import os
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = "Final_Marvellous_Crack_Detection_Model.keras"
IMAGE_SIZE = 128

st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🔎",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #111827;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">🔎 Surface Crack Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Industrial Surface Crack Detection using CNN'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** Custom CNN")
    st.write("**Input:** 128 × 128")
    st.write("**Classes:**")
    st.write("• Crack")
    st.write("• No Crack")

    st.divider()

    st.write("Upload a surface image to perform detection.")


# =========================
# MODEL LOADER
# =========================

@st.cache_resource
def load_model():

    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


# =========================
# UPLOAD
# =========================

st.subheader("📤 Upload Surface Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)


# =========================
# WAIT FOR IMAGE
# =========================

if uploaded_file is None:

    st.info("👆 Upload an image to start crack detection.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input Size", "128 × 128")

    with col2:
        st.metric("Model", "CNN")

    with col3:
        st.metric("Classes", "2")

    st.stop()


# =========================
# LOAD MODEL
# =========================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model not found:\n\n{MODEL_PATH}"
    )

    st.stop()


with st.spinner("🤖 Loading CNN model..."):

    model = load_model()


# =========================
# IMAGE
# =========================

image = Image.open(uploaded_file).convert("RGB")


col1, col2 = st.columns(2)


# =========================
# DISPLAY IMAGE
# =========================

with col1:

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


# =========================
# PREDICTION
# =========================

with col2:

    st.subheader("🤖 Prediction")

    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = np.array(
        resized,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    with st.spinner("🔍 Analyzing image..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]


    # IMPORTANT:
    # flow_from_directory alphabetically creates:
    #
    # Crack   = 0
    # NoCrack = 1

    no_crack_probability = float(prediction)

    crack_probability = 1 - no_crack_probability


    if prediction >= 0.5:

        result = "NO CRACK"
        confidence = no_crack_probability

        st.markdown(
            f"""
            <div class="result">
                <h1 style="color:#22c55e;">
                    ✅ NO CRACK
                </h1>
                <h3>
                    Confidence: {confidence*100:.2f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        result = "CRACK DETECTED"
        confidence = crack_probability

        st.markdown(
            f"""
            <div class="result">
                <h1 style="color:#ef4444;">
                    ⚠️ CRACK DETECTED
                </h1>
                <h3>
                    Confidence: {confidence*100:.2f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# PROBABILITIES
# =========================

st.divider()

st.subheader("📊 Prediction Probabilities")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Crack",
        f"{crack_probability*100:.2f}%"
    )

with col2:

    st.metric(
        "No Crack",
        f"{no_crack_probability*100:.2f}%"
    )


st.divider()

st.caption(
    "Surface Crack Detection • CNN Computer Vision Project"
)