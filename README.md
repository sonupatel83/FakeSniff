# FakeSniff  - Open Source DeepFake Detection

FakeSniff is a Streamlit-based web application for DeepFake detection, offering an easy-to-use interface for analyzing images and videos. Users can add their own deepfake detection models and compare them with existing models out of the box.

## WebApp

[Live here]()

## Features

✨ **Multi-model Support**: Users can select from multiple DeepFake detection models for both images and videos.  
📁 **File Upload**: Supports uploading images (jpg, png, jpeg) and videos (mp4, mov).  
🌐 **URL Input**: Allows users to input URLs for image or video analysis.  
⚙️ **Processing Unit Selection**: Option to use GPU for supported models (default is CPU).  
📊 **Result Visualization**: 
- 1.Displays DeepFake detection stats in a table format.
- 2.Provides downloadable CSV of detection results.
- 3.Visualizes results with bar charts for DeepFake probability and inference time.

## Usage

1. Select the "Detector" option from the sidebar.
2. Upload an image/video or provide a URL.
3. Choose the DeepFake detection model(s) you want to use.
4. Optionally select GPU processing if available.
5. Click "Real or Fake? 🤔" to start the analysis.
6. View the results in the displayed charts and tables.

## Installation

1. Create a conda environment:
    ```bash
    conda create -n FakeSniff python==3.8 -y
    conda activate FakeSniff
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the application:
    ```bash
    streamlit run main.py
    ```

## Future Work

FakeSniff acts as a platform where newer models can be incorporated into the app.

### 👥 Team Members
- [SONU PATEL]
- [ASHUTOSH SINGH]
- [AKSHAY JAIN]
- [GOURAV AGARWAL]


## Contact Information

For questions, please contact me at sonu17072003[@]gmail.com

Made with ❤️ by [Crackers--Augument.Ai]
