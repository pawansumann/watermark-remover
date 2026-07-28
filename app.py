import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
from simple_lama_inpainting import SimpleLama
import io
import zipfile

# Setup Page
st.set_page_config(page_title="AI Watermark Remover", page_icon="🔥", layout="wide")
st.title("🔥 AI Watermark Remover (Pro Quality)")
st.write("Powered by EasyOCR & LaMa AI for flawless generative inpainting.")

# Cache Models (Loads only once to save server memory)
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)
    lama = SimpleLama(device="cpu") 
    return reader, lama

reader, lama = load_models()

def remove_watermark_pro(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    results = reader.readtext(img_cv)
    
    for (bbox, text, prob) in results:
        text_lower = text.lower().strip()
        
        if "ai-generated" in text_lower or "generated" in text_lower or text_lower == "ai":
            (tl, tr, br, bl) = bbox
            x_min, y_min = int(min(tl[0], bl[0])), int(min(tl[1], tr[1]))
            x_max, y_max = int(max(tr[0], br[0])), int(max(bl[1], br[1]))
            
            pad_x, pad_y = 30, 20 
            x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
            x2, y2 = min(width, x_max + pad_x), min(height, y_max + pad_y)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    mask_pil = Image.fromarray(mask).convert('L')
    cleaned_img = lama(image, mask_pil)
    
    return cleaned_img

# UI Elements
uploaded_files = st.file_uploader("Select High-Res Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Clean Images"):
    processed_images = []
    
    for file in uploaded_files:
        init_image = Image.open(file).convert("RGB")
        st.write(f"Processing {file.name}...")
        
        # Call the Pro function
        final_img = remove_watermark_pro(init_image)
        
        # Display result
        st.image(final_img, caption=f"Flawlessly Cleaned: {file.name}")
        
        # Store in list for ZIP creation
        processed_images.append((file.name, final_img))
        
    # Create ZIP file in memory (Server friendly)
    if processed_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, img in processed_images:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG") # PNG ensures no quality loss
                zip_file.writestr(f"cleaned_{file_name}", img_buffer.getvalue())
        
        st.success("✨ All images processed successfully!")
        
        # The All-in-One Download Button
        st.download_button(
            label="📦 Download All Cleaned Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Aniviewer_Cleaned_Images.zip",
            mime="application/zip"
        )