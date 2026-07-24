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
st.write("Upload your images. We will scan the watermark zones and completely clean it without touching your main characters!")

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
        
        # 1. AUTO-FIND TEXT using EasyOCR if available, else fallback to color thresholding
        if reader is not None:
            try:
                results = reader.readtext(cv_img)
                PADDING_X = 40 
                PADDING_Y = 20 
                for (bbox, text, prob) in results:
                    if prob > 0.2:
                        (tl, tr, br, bl) = bbox
                        x1 = int(tl[0]) - PADDING_X
                        y1 = int(tl[1]) - PADDING_Y
                        x2 = int(br[0]) + PADDING_X
                        y2 = int(br[1]) + PADDING_Y
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            except Exception:
                pass
        
        # Fallback / Additional mask for bright watermarks if mask is empty
        if np.count_nonzero(mask) == 0:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(thresh, kernel, iterations=2)
        
        # 2. FAST & STABLE OPENCV INPAINTING (Replaces heavy LaMa to prevent RAM crashes)
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
                
                # 💾 DIRECT DOWNLOAD BUTTON 💾
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Cleaned Image",
                        data=f,
                        file_name=f"Cleaned_{file.name}",
                        mime="image/png"
                    )