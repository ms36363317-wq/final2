import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_efficientnetb3.keras")

IMG_SIZE = (300, 300)

def predict(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return pred

st.title("Eye Disease Detection 👁️")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    result = predict(img)

    classes = ["Glaucoma", "Normal"]
    pred_class = classes[np.argmax(result)]
    confidence = np.max(result)

    st.write(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}")
