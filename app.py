import streamlit as st
import os
import subprocess
import cv2
import numpy as np
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
st.title("🤖 Watermark Remover ")
st.write("Upload your images. we will scan the watermark zones and completely clean it without touching your main characters!")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

uploaded_files = st.file_uploader("Select Anime Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Auto-Find & Clean Images"):
    
    # Clear previous files
    for f in os.listdir(INPUT_DIR): os.remove(os.path.join(INPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR): os.remove(os.path.join(OUTPUT_DIR, f))
    
    st.info("⏳ we are scanning watermark zones to clean images... Please wait.")
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        img_path = os.path.join(INPUT_DIR, file.name)
        Image.open(file).convert("RGB").save(img_path)
        
        cv_img = cv2.imread(img_path)
        mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        height, width = cv_img.shape[:2]
        
        # 1. AUTO-FIND TEXT
        results = reader.readtext(cv_img)
        
        PADDING_X = 40 
        PADDING_Y = 20 
        
        for (bbox, text, prob) in results:
            # 🚨 NEW LOGIC: Scans 100% of the image, targets multiple watermarks safely 🚨
            # Only targets text if the AI is confident it is actually text (prob > 0.3)
            if prob > 0.3:
                (tl, tr, br, bl) = bbox
                
                x1 = int(tl[0]) - PADDING_X
                y1 = int(tl[1]) - PADDING_Y
                x2 = int(br[0]) + PADDING_X
                y2 = int(br[1]) + PADDING_Y
                
                # Keep mask boundaries within the image size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw solid rectangles for every text found
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        mask_name = f"mask_{file.name}"
        mask_path = os.path.join(INPUT_DIR, mask_name)
        cv2.imwrite(mask_path, mask)
        
        # 3. RUN LAMA MODEL
        command = f"iopaint run --image \"{img_path}\" --mask \"{mask_path}\" --output \"{OUTPUT_DIR}\" --model lama --device cpu"
        subprocess.run(command, shell=True)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success("✅ Autonomous Finding & Cleaning Complete!")
    
    # Display Results with Download Buttons
    for file in uploaded_files:
        out_path = os.path.join(OUTPUT_DIR, file.name)
        out_path_png = os.path.join(OUTPUT_DIR, file.name.rsplit('.', 1)[0] + '.png')
        final_display = out_path_png if os.path.exists(out_path_png) else out_path
        
        if os.path.exists(final_display):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_path, caption=f"Original: {file.name}")
            with col2:
                st.image(final_display, caption=f"Flawless AI Cleaned: {file.name}")
                
                # 💾 DIRECT DOWNLOAD BUTTON 💾
                with open(final_display, "rb") as f:
                    btn = st.download_button(
                        label="💾 Download Cleaned Image",
                        data=f,
                        file_name=f"Cleaned_{file.name}",
                        mime="image/png" if final_display.endswith('.png') else "image/jpeg"
                    )