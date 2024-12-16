import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Set up the Streamlit app title
st.title("Potato Leaf Analyzer")

# Load the trained model
model = load_model("analyzer.h5")
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Replace with actual class names from your dataset

# Define the image size the model expects
IMG_SIZE = (256, 256)

# Helper function to preprocess an image
def preprocess_image(image):
    # Resize the image
    image = image.resize(IMG_SIZE)
    # Convert the image to an array and rescale pixel values to [0, 1]
    image_array = img_to_array(image) / 255.0
    # Add a batch dimension
    return np.expand_dims(image_array, axis=0)

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    preprocessed_image = preprocess_image(image)

    # Make predictions
    with st.spinner("Analyzing the image..."):
        predictions = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    # Display the prediction results
    st.subheader("Prediction Results:")
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Add color-coded output
    if predicted_class == "Class 1":
        st.success(f"Prediction: {predicted_class} with {confidence:.2f}% confidence.")
    elif predicted_class == "Class 2":
        st.warning(f"Prediction: {predicted_class} with {confidence:.2f}% confidence.")
    else:
        st.info(f"Prediction: {predicted_class} with {confidence:.2f}% confidence.")