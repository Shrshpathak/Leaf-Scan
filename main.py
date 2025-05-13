import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "spplantdisease.h5")

model = tf.keras.models.load_model(model_path)

class_indices_path = os.path.join(working_dir, "class_indices.json")
class_indices = json.load(open(class_indices_path))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_name, confidence

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.markdown('<h1 class="main-header">🌿 Plant Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Upload a plant leaf image to identify possible diseases</h2>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<h3 class="sub-header">Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(uploaded_image, use_container_width=True)

    with col2:
        st.markdown('<h3 class="sub-header">Prediction Result</h3>', unsafe_allow_html=True)
        predicted_class, confidence = predict_image_class(model, uploaded_image, class_indices)
        st.markdown(f'<div class="result-box">Predicted Class: <b>{predicted_class}</b><br>Confidence: <b>{confidence:.2f}%</b></div>', unsafe_allow_html=True)

        if st.button('Try Another Image'):
            st.experimental_rerun()
