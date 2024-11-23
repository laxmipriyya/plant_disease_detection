
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

import os

# Load model
model = load_model('model.h5')

# Define class names
class_names = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites (Two-Spotted Spider Mite)",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Streamlit app
st.title("Tomato Plant Disease Detection")
st.write("Upload an image of a tomato leaf for classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    IMG_SIZE = (299, 299)
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)


    

    # Display result
    st.write(f"Predicted class: {class_names[predicted_class[0]]}")


model_path = 'model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"{model_path} not found!")

