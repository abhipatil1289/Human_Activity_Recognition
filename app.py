import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    layout="centered"
)

st.title("🤖 Human Activity Recognition")
st.write("Upload an image to classify the activity")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# ----------------------------
# Class Names (ORDER MATTERS!)
# ----------------------------
class_names = ["kick", "punch", "selfie"]

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    # ----------------------------
    # Prediction
    # ----------------------------
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("Prediction Result")
    st.success(f"🟢 Activity: **{class_names[predicted_class].upper()}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Optional: Show probabilities
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")


