
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
from utils.save_uploaded_video_temp import save_video
from utils.save_uploaded_image_temp import save_image
from utils.clear_temp_folder import clean

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Create necessary directories
for directory in ["temp", "models", "results"]:
    os.makedirs(directory, exist_ok=True)

# Clean up any existing files
clean()

# favicon and page configs
favicon = './assets/icon.png'
st.set_page_config(page_title='FakeSniff', page_icon = favicon, initial_sidebar_state = 'expanded')

import warnings
warnings.filterwarnings("ignore")

# importing the necessary packages
from datetime import datetime
import random
import csv
import os
import pandas as pd
from statistics import mean
import json
import importlib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid

# import from utils folder
from utils.download_image_video_from_url import download_image_from_url
from utils.download_from_url import download
from utils.save_deepfake_media_url import save
from utils.is_url_image import is_url_image
from utils.download_youtube_video import download_video
from utils.del_module import delete_module
from utils.delete_temp_on_reload import clear_temp_folder_and_reload
clear_temp_folder_and_reload()

# Helper function for links
def open_link_in_new_tab(url, text):
    st.markdown(f'<a href="{url}" target="_blank">{text}</a>', unsafe_allow_html=True)

#references
from utils.get_references_of_models import get_reference

#examples
from examples.show_sample_deepfakes_from_url import examples

# Initialize model lists
models_list_image = []
models_list_video = []

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Create __init__.py in models directory if it doesn't exist
if not os.path.exists("models/__init__.py"):
    with open("models/__init__.py", "w") as f:
        pass

# Load available models
for model_file in os.listdir("models"):
    if model_file.endswith(".py") and model_file != "__init__.py":
        model_name = model_file[:-3]  # Remove .py extension
        if model_name.endswith("_video"):
            models_list_video.append(model_name)
        elif model_name.endswith("_image"):
            models_list_image.append(model_name)

print("Available video models:", models_list_video)
print("Available image models:", models_list_image)

# this is needed to calculate total inference time
start_time = time.time()
start_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px;
        max-width: 250px;
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: 0px;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebar"] [aria-expanded="true"] > div:first-child [data-testid="stSidebarContent"] {
        margin-top: 0px;
    }
    [data-testid="stSidebarToggleButton"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        " ",
        ("Detector", "Examples", "Learn", "Benchmark", "About")
    )

if add_radio == "Detector":
    model_option = "NaN"
    show_real_fake_button = False

    # Introduction
    st.write("## FakeSniff")

    # Upload button
    uploaded_file = st.file_uploader("Choose the image/video", type=['jpg', 'png', 'jpeg', 'mp4', 'mov'])
    url = st.text_input("Or paste the URL below", key="text")

    did_file_download = False

    if uploaded_file is not None:
        did_user_upload_file = True
        file_name = uploaded_file.name
        extension = file_name.split(".")[-1].lower()

        if extension in ["jpg", "jpeg", "png"]:
            uploaded_image = Image.open(uploaded_file)
            save_image(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            # Use all image models automatically
            model_option = [model[:-6].title() for model in models_list_image]
            if model_option:
                st.info("Analyzing image with available detection models...")
            else:
                st.error("No detection models available for images")

        elif extension in ["mp4", "mov"]:
            try:
                save_video(uploaded_file)
                if os.path.exists('temp/delete.mp4'):
                    with open('temp/delete.mp4', 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes, start_time=0)

                    model_option = [model[:-6].title() for model in models_list_video]
                    if model_option:
                        st.info("Analyzing video with available detection models...")

                        model_inference_probability_list = []
                        model_inference_time_list = []
                        model_name_list = []

                        for model_name in models_list_video:
                            try:
                                model = importlib.import_module(f"models.{model_name}")
                                with st.spinner(f'Running {model_name} model...'):
                                    probability, inference_time = model.detect()
                                    model_inference_probability_list.append(probability)
                                    model_inference_time_list.append(inference_time)
                                    model_name_list.append(model_name)
                            except Exception as e:
                                st.error(f"Error running {model_name}: {str(e)}")
                                continue

                        if model_inference_probability_list:
                            try:
                                valid_probabilities = [p for p in model_inference_probability_list if p > 0]
                                if valid_probabilities:
                                    probab = round(float(np.mean(valid_probabilities)), 5)
                                    if probab > 0.5:
                                        st.error(f"⚠️ This video is likely FAKE (Confidence: {probab*100:.1f}%)")
                                    else:
                                        st.success(f"✅ This video appears to be REAL (Confidence: {(1-probab)*100:.1f}%)")

                                    results_df = pd.DataFrame({
                                        'Model': model_name_list,
                                        'Probability': model_inference_probability_list,
                                        'Inference Time (s)': model_inference_time_list
                                    })
                                    st.dataframe(results_df)

                                    csv = convert_df(results_df)
                                    st.download_button("Download Results", csv, "video_analysis_results.csv", "text/csv", key='download-csv')
                                else:
                                    st.warning("⚠️ Could not determine authenticity - all detections failed")
                            except Exception as e:
                                st.error(f"Error calculating results: {str(e)}")
                    else:
                        st.error("No detection models available for videos")
                else:
                    st.error("Failed to save video file")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

        else:
            st.error("Unsupported file format")
            model_option = []

        # Automatic detection
        if isinstance(model_option, list) and model_option:
            model_inference_probability_list = []
            model_inference_time_list = []
            sorted_model_option = sorted(model_option)

            with st.spinner('Analyzing media for potential deepfakes...'):
                for model_name in sorted_model_option:
                    mod = model_name.lower()
                    mod += "_video" if extension in ["mp4", "mov"] else "_image"

                    try:
                        model = importlib.import_module("models." + mod)
                        probability, inference_time = model.detect()
                        model_inference_probability_list.append(probability)
                        model_inference_time_list.append(inference_time)
                    except Exception as e:
                        st.error(f"Error running {model_name}: {str(e)}")
                        continue

            try:
                valid_probabilities = [p for p in model_inference_probability_list if p > 0]
                if valid_probabilities:
                    probab = round(float(np.mean(valid_probabilities)), 5)
                    if probab > 0.5:
                        st.error(f"⚠️ This media is likely FAKE (Confidence: {probab*100:.1f}%)")
                    else:
                        st.success(f"✅ This media appears to be REAL (Confidence: {(1-probab)*100:.1f}%)")
                else:
                    st.warning("⚠️ Could not determine authenticity - detection models failed")
            except Exception as e:
                st.error(f"Error calculating confidence: {str(e)}")

            st.subheader("Detailed Analysis")

            result_df = pd.DataFrame({
                'Model': sorted_model_option,
                'DeepFake Probability': model_inference_probability_list,
                'Inference Time (s)': model_inference_time_list
            })
            st.dataframe(result_df)

            csv = convert_df(result_df)
            st.download_button(
                label="Download detailed results as CSV ⬇️",
                data=csv,
                file_name='deepsafe_stats.csv',
                mime='text/csv',
            )

            # Plot
            st.subheader('DeepFake Probability VS Detectors')
            st.bar_chart(result_df.set_index('Model')['DeepFake Probability'])

            st.subheader('Inference Time VS Detectors')
            st.bar_chart(result_df.set_index('Model')['Inference Time (s)'])

            st.balloons()

    elif url != "":
        did_user_upload_file = False
        if is_url_image(url):
            did_file_download = download_image_from_url(url)
        else:
            did_file_download = download(url)

        if did_file_download:
            file_name = url.split("/")[-1]
            extension = file_name.split(".")[-1]

            if extension == "png" or extension == "PNG":
                uploaded_image = Image.open('temp/delete.jpg')
                save_image(uploaded_image)
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

                # Use all available image models automatically
                model_option = [model[:-6].title() for model in models_list_image]
                if model_option:
                    st.info("Analyzing image with available detection models...")
                else:
                    st.error("No detection models available for images")

            elif extension == "jpeg" or extension == "JPEG":
                uploaded_image = Image.open('temp/delete.jpg')
                save_image(uploaded_image)
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

                # Use all available image models automatically
                model_option = [model[:-6].title() for model in models_list_image]
                if model_option:
                    st.info("Analyzing image with available detection models...")
                else:
                    st.error("No detection models available for images")

            elif extension == "jpg" or extension == "JPG":
                uploaded_image = Image.open('temp/delete.jpg')
                save_image(uploaded_image)
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

                # Use all available image models automatically
                model_option = [model[:-6].title() for model in models_list_image]
                if model_option:
                    st.info("Analyzing image with available detection models...")
                else:
                    st.error("No detection models available for images")

            elif extension == "mp4" or extension == "MP4":
                try:
                    if os.path.exists('temp/delete.mp4'):
                        video_file = open('temp/delete.mp4', 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes, start_time=0)
                        video_file.close()

                        # Use all available video models automatically
                        model_option = [model[:-6].title() for model in models_list_video]
                        if model_option:
                            st.info("Analyzing video with available detection models...")
                        else:
                            st.error("No detection models available for videos")
                    else:
                        st.error("Failed to download video file")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

            elif extension == "mov" or extension == "MOV":
                try:
                    if os.path.exists('temp/delete.mp4'):
                        video_file = open('temp/delete.mp4', 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        video_file.close()

                        # Use all available video models automatically
                        model_option = [model[:-6].title() for model in models_list_video]
                        if model_option:
                            st.info("Analyzing video with available detection models...")
                        else:
                            st.error("No detection models available for videos")
                    else:
                        st.error("Failed to download video file")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
            else:
                st.error("Unsupported file format")
                model_option = []

            # Automatically run detection if models are available
            if model_option:
                model_inference_probability_list = []
                model_inference_time_list = []
                model_option = sorted(model_option)

                with st.spinner('Analyzing media for potential deepfakes...'):
                    for model in model_option:
                        model = model.lower()
                        if extension in ["mp4", "MP4", "mov", "MOV"]:
                            model = model + "_video"
                        else:
                            model = model + "_image"

                        print(model)
                        model = importlib.import_module("models." + model)
                        model_inference_probability, model_inference_time = model.detect()
                        model_inference_probability_list.append(model_inference_probability)
                        model_inference_time_list.append(model_inference_time)

                print(model_option, model_inference_probability_list, model_inference_time_list)

                if model_option:
                    try:
                        # Filter out failed detections (0.0 probabilities)
                        valid_probabilities = [p for p in model_inference_probability_list if p > 0]
                        if valid_probabilities:
                            probab = round(float(mean(valid_probabilities)), 5)
                            
                            # Show overall result prominently
                            if probab > 0.5:
                                st.error(f"⚠️ This media is likely FAKE (Confidence: {probab*100:.1f}%)")
                            else:
                                st.success(f"✅ This media appears to be REAL (Confidence: {(1-probab)*100:.1f}%)")
                        else:
                            st.warning("⚠️ Could not determine authenticity - detection models failed")
                    except Exception as e:
                        st.error(f"Error calculating confidence: {str(e)}")
                        pass

                    print("--------------------------------------------------")

                    st.subheader("Detailed Analysis")

                    if extension in ["mp4", "MP4", "mov", "MOV"]:
                        video_array = np.array([model_inference_probability_list, model_inference_time_list])
                        video_array_df = pd.DataFrame(video_array, columns=model_option, index=["DF Probability", "Inference Time in seconds"])

                        video_array_df = video_array_df.T

                        st.table(video_array_df)

                        csv_1 = convert_df(video_array_df)

                        st.download_button(
                            label="Download detailed results as CSV ⬇️",
                            data=csv_1,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )

                        # Replace matplotlib plotting with Streamlit plotting
                        if model_inference_probability_list:
                            # Create DataFrame for plotting
                            plot_df = pd.DataFrame({
                                'Model': model_option,
                                'DeepFake Probability': model_inference_probability_list,
                                'Inference Time': model_inference_time_list
                            })
                            
                            # Plot DeepFake Probability
                            st.subheader('DeepFake Probability VS Detectors')
                            st.bar_chart(plot_df.set_index('Model')['DeepFake Probability'])
                            
                            # Plot Inference Time
                            st.subheader('Inference Time VS Detectors')
                            st.bar_chart(plot_df.set_index('Model')['Inference Time'])

                            # Show detailed results table
                            st.subheader("Detailed Analysis")
                            st.dataframe(plot_df)
                            
                            # Download results
                            csv = convert_df(plot_df)
                            st.download_button(
                                label="Download detailed results as CSV ⬇️",
                                data=csv,
                                file_name='deepsafe_stats.csv',
                                mime='text/csv',
                            )
                            
                            st.balloons()

                    else:
                        image_array = np.array([model_inference_probability_list, model_inference_time_list])
                        image_array_df = pd.DataFrame(image_array, columns=model_option, index=["DF Probability", "Inference Time in seconds"])

                        image_array_df = image_array_df.T

                        st.table(image_array_df)

                        csv_1 = convert_df(image_array_df)

                        st.download_button(
                            label="Download detailed results as CSV ⬇️",
                            data=csv_1,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )

                        # Replace matplotlib plotting with Streamlit plotting
                        if model_inference_probability_list:
                            # Create DataFrame for plotting
                            plot_df = pd.DataFrame({
                                'Model': model_option,
                                'DeepFake Probability': model_inference_probability_list,
                                'Inference Time': model_inference_time_list
                            })
                            
                            # Plot DeepFake Probability
                            st.subheader('DeepFake Probability VS Detectors')
                            st.bar_chart(plot_df.set_index('Model')['DeepFake Probability'])
                            
                            # Plot Inference Time
                            st.subheader('Inference Time VS Detectors')
                            st.bar_chart(plot_df.set_index('Model')['Inference Time'])

                            # Show detailed results table
                            st.subheader("Detailed Analysis")
                            st.dataframe(plot_df)
                            
                            # Download results
                            csv = convert_df(plot_df)
                            st.download_button(
                                label="Download detailed results as CSV ⬇️",
                                data=csv,
                                file_name='deepsafe_stats.csv',
                                mime='text/csv',
                            )
                            
                            st.balloons()

                    st.balloons()
        else:
            st.error("Failed to download file")
            model_option = []

    # Show detect button if models are selected
    if 'model_option' in locals() and model_option:
        if st.button("Detect DeepFake"):
            model_inference_probability_list = []
            model_inference_time_list = []
            model_option = sorted(model_option)

            for model in model_option:
                model = model.lower()
                if extension in ["mp4", "MP4", "mov", "MOV"]:
                    model = model + "_video"
                else:
                    model = model + "_image"

                print(model)
                model = importlib.import_module("models." + model)
                model_inference_probability, model_inference_time = model.detect()
                model_inference_probability_list.append(model_inference_probability)
                model_inference_time_list.append(model_inference_time)

            print(model_option, model_inference_probability_list, model_inference_time_list)

            if model_option:
                try:
                    # Filter out failed detections (0.0 probabilities)
                    valid_probabilities = [p for p in model_inference_probability_list if p > 0]
                    if valid_probabilities:
                        probab = round(float(mean(valid_probabilities)), 5)
                        
                        # Show overall result prominently
                        if probab > 0.5:
                            st.error(f"⚠️ This media is likely FAKE (Confidence: {probab*100:.1f}%)")
                        else:
                            st.success(f"✅ This media appears to be REAL (Confidence: {(1-probab)*100:.1f}%)")
                    else:
                        st.warning("⚠️ Could not determine authenticity - detection models failed")
                except Exception as e:
                    st.error(f"Error calculating confidence: {str(e)}")
                    pass

                print("--------------------------------------------------")

                st.subheader("DeepFake Detection Stats")

                if extension in ["mp4", "MP4", "mov", "MOV"]:
                    video_array = np.array([model_inference_probability_list, model_inference_time_list])
                    video_array_df = pd.DataFrame(video_array, columns=model_option, index=["DF Probability", "Inference Time in seconds"])

                    video_array_df = video_array_df.T

                    st.table(video_array_df)

                    csv_1 = convert_df(video_array_df)

                    st.download_button(
                        label="Download data as CSV ⬇️",
                        data=csv_1,
                        file_name='deepsafe_stats.csv',
                        mime='text/csv',
                    )

                    # Replace matplotlib plotting with Streamlit plotting
                    if model_inference_probability_list:
                        # Create DataFrame for plotting
                        plot_df = pd.DataFrame({
                            'Model': model_option,
                            'DeepFake Probability': model_inference_probability_list,
                            'Inference Time': model_inference_time_list
                        })
                        
                        # Plot DeepFake Probability
                        st.subheader('DeepFake Probability VS Detectors')
                        st.bar_chart(plot_df.set_index('Model')['DeepFake Probability'])
                        
                        # Plot Inference Time
                        st.subheader('Inference Time VS Detectors')
                        st.bar_chart(plot_df.set_index('Model')['Inference Time'])

                        # Show detailed results table
                        st.subheader("Detailed Analysis")
                        st.dataframe(plot_df)
                        
                        # Download results
                        csv = convert_df(plot_df)
                        st.download_button(
                            label="Download detailed results as CSV ⬇️",
                            data=csv,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )
                        
                        st.balloons()

                else:
                    image_array = np.array([model_inference_probability_list, model_inference_time_list])
                    image_array_df = pd.DataFrame(image_array, columns=model_option, index=["DF Probability", "Inference Time in seconds"])

                    image_array_df = image_array_df.T

                    st.table(image_array_df)

                    csv_1 = convert_df(image_array_df)

                    st.download_button(
                        label="Download data as CSV ⬇️",
                        data=csv_1,
                        file_name='deepsafe_stats.csv',
                        mime='text/csv',
                    )

                    # Replace matplotlib plotting with Streamlit plotting
                    if model_inference_probability_list:
                        # Create DataFrame for plotting
                        plot_df = pd.DataFrame({
                            'Model': model_option,
                            'DeepFake Probability': model_inference_probability_list,
                            'Inference Time': model_inference_time_list
                        })
                        
                        # Plot DeepFake Probability
                        st.subheader('DeepFake Probability VS Detectors')
                        st.bar_chart(plot_df.set_index('Model')['DeepFake Probability'])
                        
                        # Plot Inference Time
                        st.subheader('Inference Time VS Detectors')
                        st.bar_chart(plot_df.set_index('Model')['Inference Time'])

                        # Show detailed results table
                        st.subheader("Detailed Analysis")
                        st.dataframe(plot_df)
                        
                        # Download results
                        csv = convert_df(plot_df)
                        st.download_button(
                            label="Download detailed results as CSV ⬇️",
                            data=csv,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )
                        
                        st.balloons()

                st.balloons()

elif add_radio == "Examples":
    examples()

elif add_radio == "Learn":
    st.write("""
    ## Learn about DeepFakes
    """)
    st.write("""
    DeepFakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. While the act of faking content is not new, deepfakes leverage powerful techniques from machine learning and artificial intelligence to manipulate or generate visual and audio content with a high potential to deceive.
    """)
    st.write("""
    ### How do DeepFakes work?
    """)
    st.write("""
    DeepFakes are created using deep learning techniques, particularly Generative Adversarial Networks (GANs). The process involves:
    1. Training a neural network on a large dataset of images/videos of the target person
    2. Using this trained network to generate new images/videos that look like the target person
    3. Blending these generated images/videos with the original content
    """)
    st.write("""
    ### Why are DeepFakes concerning?
    """)
    st.write("""
    DeepFakes can be used for:
    - Misinformation and fake news
    - Identity theft
    - Political manipulation
    - Revenge porn
    - Financial fraud
    """)
    st.write("""
    ### How can we detect DeepFakes?
    """)
    st.write("""
    DeepFake detection methods include:
    1. Visual artifacts analysis
    2. Facial inconsistencies detection
    3. Audio-visual synchronization checks
    4. Deep learning-based detection models
    5. Metadata analysis
    """)
    st.write("""
    ### Best practices for staying safe
    """)
    st.write("""
    1. Be skeptical of unverified media
    2. Check multiple sources
    3. Look for inconsistencies
    4. Use DeepFake detection tools
    5. Report suspicious content
    """)

elif add_radio == "Benchmark":
    st.write("""
    ## Benchmark DeepFake Detection Models
    """)
    st.write("""
    This section allows you to benchmark different DeepFake detection models on a set of test images or videos.
    """)

    # File uploader for benchmark dataset
    benchmark_files = st.file_uploader("Upload benchmark dataset (images/videos)", type=['jpg', 'png', 'jpeg', 'mp4', ".mov"], accept_multiple_files=True)

    if benchmark_files:
        st.write(f"Uploaded {len(benchmark_files)} files for benchmarking")
        
        # Model selection for benchmarking
        benchmark_models = st.multiselect(
            "Select models to benchmark",
            ["Deepware", "CVIT", "Selim", "Boken"]
        )

        if benchmark_models and st.button("Run Benchmark"):
            results = []
            status_text = st.empty()
            
            for i, file in enumerate(benchmark_files):
                status_text.text(f"Processing file {i+1}/{len(benchmark_files)}...")
                
                # Save the file
                if file.name.lower().endswith(('.mp4', '.mov')):
                    save_video(file)
                else:
                    save_image(file)
                
                # Process with each model
                file_results = []
                for model in benchmark_models:
                    try:
                        model = model.lower()
                        if file.name.lower().endswith(('.mp4', '.mov')):
                            model = model + "_video"
                        else:
                            model = model + "_image"
                            
                        model_module = importlib.import_module("models." + model)
                        prob, time = model_module.detect()
                        
                        file_results.append({
                            "Model": model.replace("_video", "").replace("_image", ""),
                            "Probability": prob,
                            "Inference Time": time
                        })
                    except Exception as e:
                        st.error(f"Error processing {file.name} with {model}: {str(e)}")
                        continue
                
                results.extend(file_results)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate metrics
            metrics = {}
            for model in benchmark_models:
                model_results = results_df[results_df["Model"] == model.lower()]
                if not model_results.empty:
                    metrics[model] = {
                        "Average Probability": model_results["Probability"].mean(),
                        "Average Inference Time": model_results["Inference Time"].mean(),
                        "Total Files Processed": len(model_results)
                    }
            
            # Display results
            st.write("### Benchmark Results")
            st.table(pd.DataFrame(metrics).T)
            
            # Save results to CSV
            csv_results = convert_df(results_df)
            st.download_button(
                        label="Download Benchmark Results as CSV ⬇️",
                        data=csv_results,
                        file_name='benchmark_results.csv',
                        mime='text/csv'
                    )
                    
            st.write("### Benchmarking Completed")
            status_text.text("")

elif add_radio == "About":
    st.write("""
    ## About DeepSafe
    """)
    st.write("""
    DeepSafe is an open-source DeepFake detection platform that provides multiple detection methods and benchmarking capabilities.
    """)
    st.write("""
    ### Features
    - Multiple detection models
    - Support for both images and videos
    - Benchmarking tools
    - User-friendly interface
    - Open source code
    """)
    st.write("""
    ### Contributing
    We welcome contributions! Please feel free to submit pull requests or open issues.
    """)
    st.write("""
    ### License
    This project is licensed under the MIT License - see the LICENSE file for details.
    """)

# Clean up at the end
clean()

# Add helper function for links
def open_link_in_new_tab(url, text):
    st.markdown(f'<a href="{url}" target="_blank">{text}</a>', unsafe_allow_html=True) 