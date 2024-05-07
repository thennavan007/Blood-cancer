
# Updated Streamlit app with Home Page and Navigation to Different Sections
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def home_page():
    import streamlit as st
    from PIL import Image
  

    img = Image.open("blood cell.jpg")

    st.image(
        img,
        caption="Blood Cancer Classification and Prediction",
        width=800,

        channels="BGR"
    )
    st.title("Bone Marrow Blood Cancer Classification Project")
    st.header("Project Description")
    st.write(
        "This project aims to classify different types of blood cells in bone marrow for the prediction  of blood cancer.")



def prediction_page():
    # Load the trained model
    model = load_model('Nasnet (1).h5')

    # Define class labels
    class_labels = {0: 'Early-pre-B', 1: 'Pre-B', 2: 'Pro-B', 3: 'Benign'}

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



def graphs_page():
    # Function for the graphs page
    st.title("Blood Cancer Detection - Graphs")
    st.image("dataset.png")
    st.image("accuracy and loss.png")
    st.image("validation_Accuracy.png")
    st.image("validation_Loss.png")
    # Include code to display relevant graphs here

def algorithm_accuracy_page():
    # Function for the algorithm with accuracy page
    st.title("Blood Cancer Detection - Algorithm & Accuracy")
    st.write("Details about the classification algorithm and accuracy metrics will be displayed here.")
    data = {
        'precision': [0.98, 1.00],
        'recall': [1.00, 0.99],
        'f1-score': [0.99, 1.00],
        'support': [98, 106]
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data, index=[0, 1])

    # Display the DataFrame in the Streamlit app
    st.write("Precision, Recall, F1-Score, and Support:")
    st.write(df)


# Create a dictionary containing the precision, recall, f1-score, and support values


def main():
    # Define navigation section for the home page
    page = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Graphs", "Algorithm & Accuracy"])

    # Display different pages based on the selected option
    if page == "Home":
        # Add a custom CSS style to set the background image

        home_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Graphs":
        graphs_page()
    elif page == "Algorithm & Accuracy":
        algorithm_accuracy_page()


if __name__ == '__main__':
    main()
