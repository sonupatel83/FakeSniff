import subprocess
import os
import shutil
import requests
from tqdm import tqdm

def download(url):
    """
    Downloads a video from the specified URL.
    First tries using you-get, then falls back to direct download.
    
    Args:
        url (str): The URL of the video to be downloaded.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Clean up any existing files
    if os.path.exists("temp/delete.jpg"):
        os.remove("temp/delete.jpg")
    if os.path.exists("temp/delete.mp4"):
        os.remove("temp/delete.mp4")
    
    # First try using you-get
    try:
        output_dir = "temp/"
        output_filename = "delete"
        command = ["you-get", url, "--output-dir", output_dir, "--output-filename", output_filename]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists("temp/delete.mp4"):
            return True
    except Exception as e:
        print(f"you-get failed: {str(e)}")
    
    # If you-get fails, try direct download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open("temp/delete.mp4", 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        
        return os.path.exists("temp/delete.mp4")
    except Exception as e:
        print(f"Direct download failed: {str(e)}")
        return False
