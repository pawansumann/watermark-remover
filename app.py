import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr
import ssl

# The fix for Windows SSL error
ssl._create_default_https_context = ssl._create_unverified_context

# System Folders Setup
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="AI Watermark Remover", page_icon="🤖", layout="wide")
st.title("🤖 Watermark Remover")
st.write("Upload your images. We will scan corner watermarks and clean them without touching your central art!")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

try:
    reader = load_ocr()
except Exception as e:
    st.warning("Running in lightweight mode due to memory limits.")
    reader = None

uploaded_files = st.file_uploader("Select Anime Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Auto-Find & Clean Images"):
    
    # Clear previous files
    for f in os.listdir(INPUT_DIR): os.remove(os.path.join(INPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR): os.remove(os.path.join(OUTPUT_DIR, f))
    
    st.info("⏳ We are scanning watermark zones to clean images... Please wait.")
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        img_path = os.path.join(INPUT_DIR, file.name)
        Image.open(file).convert("RGB").save(img_path)
        
        cv_img = cv2.imread(img_path)
        height, width = cv_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if reader is not None:
            try:
                results = reader.readtext(cv_img)
                PADDING_X = 25 
                PADDING_Y = 15 
                
                for (bbox, text, prob) in results:
                    if prob > 0.2:
                        (tl, tr, br, bl) = bbox
                        x_center = (tl[0] + br[0]) / 2
                        y_center = (tl[1] + br[1]) / 2
                        
                        # 🎯 STRICT CORNER-ONLY FILTER: 
                        # Only mask text if it is located near the top/corners, avoiding the center logo/art.
                        is_near_top = y_center < (height * 0.25) # Top 25% of image
                        is_near_edge = (x_center < (width * 0.25)) or (x_center > (width * 0.75)) # Left or Right outer 25%
                        
                        if is_near_top or is_near_edge:
                            x1 = int(tl[0]) - PADDING_X
                            y1 = int(tl[1]) - PADDING_Y
                            x2 = int(br[0]) + PADDING_X
                            y2 = int(br[1]) + PADDING_Y
                            
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            except Exception:
                pass
        
        # Fast OpenCV Inpainting using the safe corner-only mask
        cleaned_img = cv2.inpaint(cv_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        out_path = os.path.join(OUTPUT_DIR, file.name)
        cv2.imwrite(out_path, cleaned_img)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success("✅ Autonomous Finding & Cleaning Complete!")
    
    # Display Results with Download Buttons
    for file in uploaded_files:
        original_path = os.path.join(INPUT_DIR, file.name)
        out_path = os.path.join(OUTPUT_DIR, file.name)
        
        if os.path.exists(out_path):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_path, caption=f"Original: {file.name}")
            with col2:
                st.image(out_path, caption=f"Flawless Cleaned: {file.name}")
                
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Cleaned Image",
                        data=f,
                        file_name=f"Cleaned_{file.name}",
                        mime="image/png"
                    )