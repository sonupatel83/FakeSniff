import os, io
import streamlit as st

def save_video(uploaded_file):
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Clean up any existing files
    if os.path.exists("temp/delete.jpg"):
        os.remove("temp/delete.jpg")
    if os.path.exists("temp/delete.mp4"):
        os.remove("temp/delete.mp4")
    
    # Save the uploaded file to the temp directory
    temporary_location = "temp/delete.mp4"
    with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
        out.write(uploaded_file.getbuffer())  # Read and write the file to the temporary location
    
    return temporary_location

    if video_file is not None:
        g = io.BytesIO(video_file.read())  ## BytesIO Object
        temporary_location = "temp/delete.mp4"
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        # close file
        if os.path.exists("temp/delete.jpg"):
            os.remove("temp/delete.jpg") # one file at a time
        out.close()
    else:
        print("Corrupted file!")