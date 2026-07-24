import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="AI Watermark Remover", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Seamless Watermark Remover")
st.write("Advanced texture blending and AI-touch removal without blur artifacts!")

uploaded_files = st.file_uploader("Select Anime Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Apply AI Magic Removal"):
    
    for f in os.listdir(INPUT_DIR): os.remove(os.path.join(INPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR): os.remove(os.path.join(OUTPUT_DIR, f))
    
    st.info("⏳ Applying AI texture synthesis & smart blending... Please wait.")
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        img_path = os.path.join(INPUT_DIR, file.name)
        Image.open(file).convert("RGB").save(img_path)
        
        cv_img = cv2.imread(img_path)
        height, width = cv_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Exact Top-Right Badge Zone
        badge_width = int(width * 0.14)
        badge_height = int(height * 0.06)
        
        x2 = width - int(width * 0.015)
        x1 = x2 - badge_width
        y1 = int(height * 0.015)
        y2 = y1 + badge_height
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 🧠 AI-Touch Multi-Pass Texture Blending Strategy
        # Pass 1: Fast Marching structure reconstruction (TELEA)
        base_clean = cv2.inpaint(cv_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Pass 2: Navier-Stokes detail smoothing to eliminate the blur patch and match ambient lighting
        ai_touch_img = cv2.inpaint(base_clean, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # Pass 3: Edge sharpening to restore crystal-clear anime texture
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Apply sharpening only near the badge area to keep overall image pristine
        roi = ai_touch_img[y1:y2, x1:x2]
        sharpened_roi = cv2.filter2D(roi, -1, kernel)
        ai_touch_img[y1:y2, x1:x2] = sharpened_roi

        out_path = os.path.join(OUTPUT_DIR, file.name)
        cv2.imwrite(out_path, ai_touch_img)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    st.success("✅ AI Touch-Up & Watermark Removal Complete!")
    
    for file in uploaded_files:
        original_path = os.path.join(INPUT_DIR, file.name)
        out_path = os.path.join(OUTPUT_DIR, file.name)
        
        if os.path.exists(out_path):
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_path, caption=f"Original: {file.name}")
            with col2:
                st.image(out_path, caption=f"AI-Enhanced Cleaned: {file.name}")
                
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Cleaned Image",
                        data=f,
                        file_name=f"Cleaned_{file.name}",
                        mime="image/png"
                    )