import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ------------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="wide"
)

# ------ LOAD MODEL --------
MODEL_PATH = "cat_dog_model.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'cat_dog_model.keras' not found.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)


# ----- SIDEBAR NAVIGATION -------
with st.sidebar:

    st.title("🐱🐶 Cat vs Dog")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📖 About Project",
            "🔍 Image Prediction"
        ]
    )

    st.markdown("---")

    st.caption("Deep Learning Project")


# ABOUT PROJECT PAGE

if page == "📖 About Project":

    st.title("🐱🐶 Cat vs Dog Classifier")

    st.subheader("📖 About Project")

    st.write("""
    This is a Deep Learning based image classification
    application that identifies whether an uploaded image
    contains a Cat or a Dog.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model")

        st.write("""
        **MobileNetV2**

        • Input Image Size: 160 × 160  
        • Deep Learning Model  
        • Image Classification
        """)

    with col2:
        st.subheader("🛠️ Technologies")

        st.write("""
        • Python  
        • TensorFlow  
        • Keras  
        • NumPy  
        • Pillow  
        • Streamlit
        """)

    st.markdown("---")

    st.subheader("⚙️ How It Works")

    st.write("""
    1. User uploads a Cat or Dog image.
    2. Image is resized to 160 × 160 pixels.
    3. Image is converted into a NumPy array.
    4. The trained MobileNetV2 model processes the image.
    5. The model predicts Cat or Dog.
    6. Prediction result and confidence are displayed.
    """)

    st.markdown("---")

    st.subheader("🎯 Project Objective")

    st.write("""
    The objective of this project is to demonstrate the
    practical implementation of Deep Learning and Image
    Classification using Python and TensorFlow.
    """)


# IMAGE PREDICTION PAGE

elif page == "🔍 Image Prediction":

    st.title("🔍 Image Prediction")

    st.write("Upload an image and let the model predict whether it is a Cat or Dog.")

    uploaded_file = st.file_uploader(
        "📤 Upload Cat or Dog Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        # ---------------- IMAGE ----------------
        with col1:

            st.subheader("📷 Uploaded Image")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # ---------------- PREDICTION ----------------
        with col2:

            st.subheader("🤖 Prediction")

            # Model expects 160 x 160
            img = image.resize((160, 160))

            img_array = np.array(img)

            # Normalize
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(
                img_array,
                verbose=0
            )

            probability = prediction[0][0]

            if probability > 0.5:

                confidence = probability * 100

                st.success("🐶 DOG")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            else:

                confidence = (1 - probability) * 100

                st.success("🐱 CAT")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )