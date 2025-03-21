import os
import shutil
import time

def clean():
    temp_dir = './temp'
    
    # Remove the temp directory if it exists
    if os.path.exists(temp_dir):
        try:
            # Try to remove the directory
            shutil.rmtree(temp_dir)
        except PermissionError:
            # If permission error occurs, try to remove individual files
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except PermissionError:
                        # If file is still in use, wait a bit and try again
                        time.sleep(0.1)
                        try:
                            os.remove(os.path.join(root, name))
                        except:
                            pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except PermissionError:
                        pass
            # Try to remove the directory one final time
            try:
                os.rmdir(temp_dir)
            except:
                pass
        except Exception as e:
            print(f"Error cleaning temp directory: {e}")
            pass
    else:
        # Create the temp directory
        os.makedirs(temp_dir, exist_ok=True)

    # Remove result.txt files in the models directory if it exists
    models_dir = './models'
    if os.path.exists(models_dir):
        for model in os.listdir(models_dir):
            result_file = os.path.join(models_dir, f"{model}result.txt")
            if os.path.exists(result_file):
                os.remove(result_file)
    else:
        # Optionally create the models directory if it's needed later
        os.makedirs(models_dir)

clean()
