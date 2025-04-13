# Streamlit_App.ipynb

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
print(tf.__version__)


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.title("Currency Classifier System")
st.write("Upload an image of an Indian currency note to classify its denomination.")

# Define a mapping for display
denom_classes = {
    0: "₹10",
    1: "₹20",
    2: "₹50",
    3: "₹100",
    4: "₹200",
    5: "₹500",
    6: "₹2000"
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("currency_classifier.keras")

try:
    model = load_trained_model()
except Exception as e:
    st.error("Error loading model. Ensure 'currency_classifier.keras' exists.")
    st.stop()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Error reading the image!")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        img_resized = cv2.resize(img_rgb, (224, 224))
        input_image = img_resized.astype("float32") / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        pred = model.predict(input_image)
        predicted_class = int(pred.argmax())
        predicted_denom = denom_classes.get(predicted_class, "Unknown")
        
        st.write(f"**Predicted Currency Note:** {predicted_denom}")
