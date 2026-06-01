import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Earthquake Debris Detection", page_icon="🚁", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        padding-top: 20px;
    }
    .sub-header {
        font-size: 18px;
        color: #888888;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px !important;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 75, 43, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(255, 75, 43, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #FF4B2B;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🚁 YOLOv12 Earthquake Debris Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Automatic Damage and Debris Analysis</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

# Left Panel (Sidebar) - Settings and Info
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    
    # Allow user to adjust the confidence threshold dynamically
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Increasing this value shows only high-confidence predictions. Decreasing it allows the model to predict more freely."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance (Test)")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label="Precision", value="50.5%")
    col_m2.metric(label="Recall", value="35.3%")
    st.metric(label="Mean Average Precision (mAP@50)", value="36.8%")
    
    st.markdown("---")
    st.markdown("#### 👨‍💻 Project Details")
    st.info("This model was **fine-tuned** using a custom aerial photography dataset obtained from the Kahramanmaraş earthquake, based on the **YOLOv12x** architecture. \\n\\nYou can access the dataset here: [Kahramanmaraş Dataset (Roboflow)](https://universe.roboflow.com/aeb-9lieg/maras-tpx5p-me1m7)")

# Main Screen Layout
uploaded_file = st.file_uploader("📥 Upload Image for Analysis (Drag & drop or browse)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(pil_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #555;'>📸 Original Photo</h3>", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='text-align: center; color: #555;'>🎯 Analysis Result</h3>", unsafe_allow_html=True)
        
        # Using empty columns to center the button
        b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
        with b_col2:
            analyze_button = st.button("🚀 Detect Debris", use_container_width=True)
            
        if analyze_button:
            with st.spinner('AI is processing the image...'):
                # Pass numpy array to YOLO, dynamic confidence threshold
                results = model.predict(source=image_array, conf=conf_threshold, device="cpu", verbose=False)
                res_plotted = results[0].plot()
                res_rgb = res_plotted[..., ::-1]
                
                st.image(res_rgb, use_container_width=True)
                
                # Count found bounding boxes
                detection_count = len(results[0].boxes)
                if detection_count > 0:
                    st.success(f"✅ Analysis complete! A total of **{detection_count}** suspected debris/damaged structures were detected.")
                else:
                    st.warning("⚠️ No debris exceeding the confidence threshold was detected in this photo. You can try lowering the 'Confidence Threshold' from the left menu.")
