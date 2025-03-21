import os
from PIL import Image
import numpy as np
import cv2
import streamlit as st



@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def save_image(uploaded_file):
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Save the uploaded file to the temp directory
    temporary_location = f"temp/{uploaded_file.name}"
    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(uploaded_file.getbuffer())  ## Read and write the file to the temporary location
    return temporary_location