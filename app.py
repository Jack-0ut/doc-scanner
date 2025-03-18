import streamlit as st
import numpy as np
import cv2
import tempfile
from docscanner import DocScanner

def main():
    st.title("Document Scanner")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    #interactive_mode = st.checkbox("Manually adjust document corners")

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Save as a temporary file for OpenCV processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image_cv)  # Save the image for scanner

        scanner = DocScanner(interactive=False)
        scanned_image = scanner.scan(temp_path)  # Process image

        # Debugging: Print error if scan failed
        if scanned_image is None:
            st.error("⚠️ Document scanning failed. Please try another image.")
            return

        # Display the scanned image
        st.image(scanned_image, caption="Scanned Document", use_container_width=True)

        # Allow user to download the scanned image
        _, buffer = cv2.imencode(".jpg", scanned_image)
        st.download_button(label="Download Scanned Image",
                           data=buffer.tobytes(),
                           file_name="scanned_document.jpg",
                           mime="image/jpeg")

if __name__ == "__main__":
    main()
