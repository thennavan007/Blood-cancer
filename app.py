import streamlit as st
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Nasnet.h5')

# Define class labels
class_labels = {0: 'E-pre-B', 1: 'Pre-B', 2: 'Pro-B', 3: 'Benign'}

# Threshold for classifying as not a valid image
confidence_threshold = 0.5

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to check if the predicted class is valid
def is_valid_class(prediction):
    max_confidence = np.max(prediction)
    return max_confidence > confidence_threshold

# Streamlit app
def main():
    st.title("Bone Marrow Blood Cancer Classification and Prediction")
    st.write("Upload an image of a blood cell, and we'll predict its class.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a blood cell image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # Preprocess the image
                img = preprocess_image(uploaded_file)
                # Make prediction
                prediction = model.predict(img)
                # Check if the predicted class is valid
                if is_valid_class(prediction):
                    # Get the predicted class label
                    predicted_class = np.argmax(prediction)
                    predicted_label = class_labels.get(predicted_class, "Unknown")
                    # Display the prediction result
                    st.success(f'Predicted class: {predicted_label}')
                else:
                    st.error("Not a valid image")

if __name__ == '__main__':
    main()
