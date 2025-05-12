import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
<<<<<<< HEAD

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/spplantdisease.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

=======
import requests
import tempfile

google_drive_url = "https://drive.google.com/file/d/1W7m51ucePuEhzuZHZbOch2e7p-EGYHmu/view?usp=sharing"

def download_model_from_google_drive(url, filename="spplantdisease.h5"):
    file_id = url.split("/")[5]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        model_path = temp_file.name
    return model_path

model_path = download_model_from_google_drive(google_drive_url)
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open("class_indices.json"))
>>>>>>> origin/main

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

<<<<<<< HEAD

=======
>>>>>>> origin/main
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_name, confidence

<<<<<<< HEAD

=======
>>>>>>> origin/main
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.7em; color: #2E8B57; text-shadow: 1px 1px 2px #ffffff; background: none;}
    .sub-header {font-size: 1.5em; color: #4682B4; margin-bottom: 20px; background: none;}
    .footer {font-size: 0.9em; text-align: center; padding-top: 30px; color: grey; background: none;}
<<<<<<< HEAD

    
    .result-box {
        padding: 20px; 
        border-radius: 10px; 
        background: none; 
        border: 2px solid #4682B4; 
        margin-top: 20px;
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }

    
    .stButton > button {
        background-color: #4682B4;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        box-shadow: 0px 0px 10px rgba(70, 130, 180, 0.7);
        transform: scale(1.05);
    }

    /* Smooth fade-in for uploaded images */
    .image-container img {
        animation: fadeInImage 1s ease-in-out;
    }

    @keyframes fadeInImage {
        0% {opacity: 0; transform: scale(0.9);}
        100% {opacity: 1; transform: scale(1);}
    }
=======
    .result-box {padding: 20px; border-radius: 10px; background: none; border: 2px solid #4682B4; margin-top: 20px; animation: fadeIn 1s ease-in-out;}
    @keyframes fadeIn {0% {opacity: 0;} 100% {opacity: 1;}}
    .stButton > button {background-color: #4682B4; color: white; border-radius: 8px; padding: 10px 20px; transition: 0.3s;}
    .stButton > button:hover {box-shadow: 0px 0px 10px rgba(70, 130, 180, 0.7); transform: scale(1.05);}
    .image-container img {animation: fadeInImage 1s ease-in-out;}
    @keyframes fadeInImage {0% {opacity: 0; transform: scale(0.9);} 100% {opacity: 1; transform: scale(1);}}
>>>>>>> origin/main
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌿 Plant Disease Classifier</h1>', unsafe_allow_html=True)
<<<<<<< HEAD
st.markdown('<h2 class="sub-header">Upload a plant leaf image to identify possible diseases</h2>',
            unsafe_allow_html=True)
=======
st.markdown('<h2 class="sub-header">Upload a plant leaf image to identify possible diseases</h2>', unsafe_allow_html=True)
>>>>>>> origin/main

uploaded_image = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns([1, 2])
<<<<<<< HEAD

    with col1:
        st.markdown('<h3 class="sub-header">Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(uploaded_image, use_container_width=True)

    with col2:
        st.markdown('<h3 class="sub-header">Prediction Result</h3>', unsafe_allow_html=True)
        predicted_class, confidence = predict_image_class(model, uploaded_image, class_indices)
        st.markdown(
            f'<div class="result-box">Predicted Class: <b>{predicted_class}</b><br>Confidence: <b>{confidence:.2f}%</b></div>',
            unsafe_allow_html=True)

        if st.button('Try Another Image'):
            st.experimental_rerun()

st.sidebar.title("Comments")
st.sidebar.markdown("""
    <div>
        This application allows users to upload a plant leaf image to identify possible diseases using a pre-trained machine learning model.
        The classifier integrates additional data sources and employs advanced image processing techniques to improve prediction accuracy
        and provide comprehensive disease management solutions.
    </div>
    """, unsafe_allow_html=True)
=======
    with col1:
        st.markdown('<h3 class="sub-header">Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(uploaded_image, use_container_width=True)
    with col2:
        st.markdown('<h3 class="sub-header">Prediction Result</h3>', unsafe_allow_html=True)
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())
        predicted_class_name, confidence = predict_image_class(model, "temp_image.jpg", class_indices)
        st.markdown(f'<div class="result-box"><b>Predicted Class:</b> {predicted_class_name}<br><b>Confidence:</b> {confidence:.2f}%</div>', unsafe_allow_html=True)
        os.remove("temp_image.jpg")
>>>>>>> origin/main
