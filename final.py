import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from keras.layers import TFSMLayer  # For loading TensorFlow SavedModel in Keras 3

# Disable TensorFlow Metal Plugin if causing issues on macOS
os.environ["TF_METAL_ENABLED"] = "0"

# Load the TensorFlow SavedModel as a TFSMLayer
model = TFSMLayer("model", call_endpoint="serving_default")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input
    image_np = np.array(image)  # Convert to NumPy array
    img_batch = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return img_batch

# Function to capture an image from the camera
def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from camera.")
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Streamlit UI
st.title("Potato Leaf Disease Detection")

# Choose between file upload and camera input
option = st.radio("Choose Input Method:", ("File Upload", "Camera Capture"))

if option == "File Upload":
    uploaded_file = st.file_uploader("Drag and drop a file here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_batch = preprocess_image(image)

        # Make predictions
        with st.spinner("Making Predictions..."):
            predictions = model(img_batch)  # Inference using TFSMLayer
            predictions = predictions.numpy()  # Convert Tensor to NumPy array

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display prediction results
        st.subheader("Prediction Result:")
        st.write(f"Class: {predicted_class}")

        # Add color or icon based on the predicted class
        if predicted_class == "Early Blight":
            st.success("Early Blight Detected")
        elif predicted_class == "Late Blight":
            st.warning("Late Blight Detected")
        elif predicted_class == "Healthy":
            st.info("Healthy Potato Leaf")

        # Display confidence
        st.subheader("Prediction Confidence:")
        confidence_percentage = confidence * 100
        st.progress(int(confidence_percentage))
        st.write(f"Confidence: {confidence_percentage:.2f}%")

else:
    # Use the camera for input
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not access the camera. Make sure it is properly connected and accessible.")
    else:
        # Display placeholders for camera feed and captured frame
        camera_placeholder = st.empty()
        captured_frame_placeholder = st.empty()

        # Button to capture a single frame
        if st.button("Capture Frame"):
            captured_image = capture_image(cap)
            if captured_image is not None:
                # Preprocess the captured image
                img_batch = preprocess_image(captured_image)

                # Display the captured frame
                captured_frame_placeholder.image(captured_image, caption="Captured Frame", use_column_width=True)

                # Make predictions
                with st.spinner("Making Predictions..."):
                    predictions = model(img_batch)  # Inference using TFSMLayer
                    predictions = predictions.numpy()  # Convert Tensor to NumPy array

                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                # Display prediction results
                st.subheader("Prediction Result:")
                st.write(f"Class: {predicted_class}")

                # Add color or icon based on the predicted class
                if predicted_class == "Early Blight":
                    st.success("Early Blight Detected")
                elif predicted_class == "Late Blight":
                    st.warning("Late Blight Detected")
                elif predicted_class == "Healthy":
                    st.info("Healthy Potato Leaf")

                # Display confidence
                st.subheader("Prediction Confidence:")
                confidence_percentage = confidence * 100
                st.progress(int(confidence_percentage))
                st.write(f"Confidence: {confidence_percentage:.2f}%")

        # Release the camera when done
        cap.release()