# 🚁 Building Debris Detection with YOLO

This project is a deep learning-based tool designed to automatically detect debris and heavily damaged buildings from aerial photographs taken after earthquakes.

Using datasets obtained from the Kahramanmaraş earthquakes, we applied **Fine-Tuning** to the **YOLOv12x** (Extra Large) architecture to specialize it for disaster zones. The model can instantly detect destroyed and severely damaged structures in aerial imagery by drawing bounding boxes around them.

### 🗄️ Dataset
The model was fine-tuned using the custom Kahramanmaraş earthquake aerial dataset, which is publicly available on Roboflow:
🔗 **[Kahramanmaraş Debris Dataset (Roboflow Universe)](https://universe.roboflow.com/aeb-9lieg/maras-tpx5p-me1m7)**

## 🚀 Installation & Usage

### 1. Install Dependencies
First, install the required Python libraries to run the project:
```bash
pip install -r requirements.txt
```
*(Note: If MacOS users encounter system package errors, they can use `pip install -r requirements.txt --break-system-packages` or use a Python virtual environment.)*

### 2. Launch the Interface
The project includes a modern **Streamlit** web interface. To bypass PyTorch threading issues (segmentation faults) specifically on Apple Silicon/MacOS devices, always run our wrapper script:
```bash
python3 run_app.py
```
After running this command, the interface will automatically open in your web browser. (Alternatively, on Windows/Linux, you can directly run `streamlit run app_st.py`).

## 📊 Model Performance Metrics
Trained in a Kaggle environment at 480x480 resolution for 500 Epochs (with Early Stopping), the model achieved the following metrics on the validation dataset:
- **Precision:** 50.5%
- **Recall:** 35.3%
- **mAP@50:** 36.8%

## 📂 Directory Structure
- `app_st.py`: The main file containing the Streamlit interface code.
- `run_app.py`: A wrapper script that initializes the model in the main thread to prevent macOS crashes.
- `model/`: The directory containing the trained model weights (`best.pt`).
- `sample_photos/`: Sample aerial photos from the Kahramanmaraş dataset to easily test the interface.
