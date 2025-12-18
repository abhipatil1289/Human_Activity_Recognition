import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ----------------------------
# Page Config (MUST be first Streamlit command)
# ----------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    layout="centered"
)

st.title("🤖 Human Activity Recognition")
st.write("Upload an image to classify the activity")

# ----------------------------
# Check Model File
# ----------------------------
MODEL_PATH = "outputs/model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.h5 not found. Please place it in the same folder as app.py")
    st.stop()

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("outputs/model.h5")

model = load_model()


model = load_model()

# ----------------------------
# Class Names (ORDER MUST MATCH TRAINING)
# ----------------------------
class_names = ["kick", "punch", "selfie"]

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)

        # ----------------------------
        # Prediction
        # ----------------------------
        predictions = model.predict(processed_image)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        st.subheader("Prediction Result")
        st.success(f"🟢 Activity: **{class_names[predicted_class].upper()}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # ----------------------------
        # Class Probabilities
        # ----------------------------
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")



