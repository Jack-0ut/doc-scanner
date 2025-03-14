import streamlit as st
import numpy as np
import cv2
from image_processing import process_image

def load_image(uploaded_file):
    """Reads an uploaded image file and converts it into a NumPy array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Streamlit UI
st.title("Document Scanner App 📝📸")
st.write("Upload an image to scan the document.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    print(f"Image path: {uploaded_file.name}")
    try:
        warped = process_image(image)

        # Display images
        st.image(image, caption="Original (Resized)", use_container_width=True)
        st.image(warped, caption="Scanned Document", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")