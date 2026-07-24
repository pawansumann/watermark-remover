import os
import cv2
import numpy as np
import streamlit as st
import easyocr
from PIL import Image

# Directories setup
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Heavy AI Watermark Remover", page_icon="🔥", layout="wide")
st.title("🔥 Heavy AI Watermark Remover (OCR)")
st.write("Using Deep Learning (EasyOCR) to read, target, and flawlessly erase watermarks.")

# 🧠 Cache the heavy OCR model so it doesn't reload for every single image
@st.cache_resource
def load_ocr_model():
    # Will use GPU if available, otherwise fallback to CPU (Heavy RAM usage)
    return easyocr.Reader(['en'])

reader = load_ocr_model()

uploaded_files = st.file_uploader("Select High-Res Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Deploy Heavy Scan & Clean"):
    
    for f in os.listdir(INPUT_DIR): os.remove(os.path.join(INPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR): os.remove(os.path.join(OUTPUT_DIR, f))
    
    st.info("⏳ Initializing Deep Learning OCR... This might take a moment.")
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        img_path = os.path.join(INPUT_DIR, file.name)
        Image.open(file).convert("RGB").save(img_path)
        
        cv_img = cv2.imread(img_path)
        height, width = cv_img.shape[:2]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 1. AI OCR SCAN
        results = reader.readtext(cv_img)
        
        for (bbox, text, prob) in results:
            # Check if the AI found our target text (case-insensitive)
            if "ai" in text.lower() or "generated" in text.lower() or "ai-generated" in text.lower():
                
                # Extract coordinates of the text
                (tl, tr, br, bl) = bbox
                x_min = int(min(tl[0], bl[0]))
                y_min = int(min(tl[1], tr[1]))
                x_max = int(max(tr[0], br[0]))
                y_max = int(max(bl[1], br[1]))
                
                # 2. AGGRESSIVE PADDING (Fixes the "Bone" issue)
                # Text mil gaya, ab box ko chaaro taraf se bada kar do taaki white pill poora cover ho jaye
                pad_x = 40  # Horizontal padding to swallow the rounded corners
                pad_y = 15  # Vertical padding
                
                x1 = max(0, x_min - pad_x)
                y1 = max(0, y_min - pad_y)
                x2 = min(width, x_max + pad_x)
                y2 = min(height, y_max + pad_y)
                
                # Draw the expanded solid mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 3. HIGH-QUALITY INPAINTING (Navier-Stokes algorithm for better texture blending)
        cleaned_img = cv2.inpaint(cv_img, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
        
        out_path = os.path.join(OUTPUT_DIR, file.name)
        cv2.imwrite(out_path, cleaned_img)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success("✅ Heavy OCR Scanning & Removal Complete! The background is pristine.")
    
    for file in uploaded_files:
        original_path = os.path.join(INPUT_DIR, file.name)
        out_path = os.path.join(OUTPUT_DIR, file.name)
        
        if os.path.exists(out_path):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_path, caption=f"Original: {file.name}")
            with col2:
                st.image(out_path, caption=f"8K Quality Cleaned: {file.name}")
                
                with open(out_path, "rb") as f:
                    st.download_button(label="💾 Download HD Image", data=f, file_name=f"Cleaned_{file.name}", mime="image/png")