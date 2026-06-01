import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Initialize Torch and YOLO in the main thread before starting Streamlit
import torch
from ultralytics import YOLO

# Load the model once in the main thread to prevent macOS segmentation faults
print("Loading model in the main thread, please wait...")
model_path = "model/best.pt"
dummy_model = YOLO(model_path)
print("Model loaded successfully. Launching the interface...")

# Call Streamlit CLI
from streamlit.web import cli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app_st.py"]
    sys.exit(cli.main())
