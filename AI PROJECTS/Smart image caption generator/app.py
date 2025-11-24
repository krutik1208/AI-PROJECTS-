# app.py
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import io

# Import BLIP model from Hugging Face
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title='Smart Image Caption Generator', layout='wide')
st.title('🖼️ Smart Image Caption Generator')
st.write('Upload an image, crop or enhance it, then generate an automatic caption.')
st.sidebar.header('Controls')

# ---------- File Upload ----------
uploaded_file = st.sidebar.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])

# ---------- Load BLIP Model ----------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# ---------- Helper Functions ----------
def crop_image(image, x, y, w, h):
    return image.crop((x, y, x + w, y + h))

def enhance_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    return image

def detect_blur(image):
    arr = np.array(image.convert('L'))
    return float(np.var(np.gradient(arr)))

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------- App Logic ----------
col1, col2 = st.columns([1, 1])

if uploaded_file is None:
    st.info('Upload an image from the left sidebar to begin.')
else:
    image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.write('Image size:', image.size)

    st.header('Preview & Edit')
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)

    # Crop controls
    st.subheader('Crop')
    iw, ih = image.size
    x = st.number_input('x (left)', min_value=0, max_value=iw - 1, value=0)
    y = st.number_input('y (top)', min_value=0, max_value=ih - 1, value=0)
    w = st.number_input('width', min_value=1, max_value=iw - x, value=min(ih, iw)//2)
    h = st.number_input('height', min_value=1, max_value=ih - y, value=min(ih, iw)//2)

    cropped = crop_image(image, int(x), int(y), int(w), int(h))

    with col2:
        st.image(cropped, caption='Cropped Preview', use_column_width=True)

    # Blur detection
    blur_var = detect_blur(cropped)
    st.write('Blur metric (Laplacian variance):', round(blur_var, 2))

    # Enhancement
    enhance = st.checkbox('Apply enhancement (sharpen + contrast)')
    if enhance:
        enhanced = enhance_image(cropped)
    else:
        enhanced = cropped

    st.image(enhanced, caption='Edited Image', use_column_width=True)

    # Caption Generation
    st.header('Caption')

if st.button('Generate caption'):
    st.info("🔍 Checking if BLIP model works...")

    try:
        # print model check
        st.write("Model type:", type(blip_model))

        inputs = processor(images=enhanced, return_tensors="pt")
        st.write("Processor output keys:", list(inputs.keys()))

        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"📝 Caption: {caption}")

    except Exception as e:
        st.error(f"❌ BLIP failed: {e}")


    # Download button
    buf = io.BytesIO()
    enhanced.save(buf, format='JPEG')
    st.download_button('📥 Download Edited Image', data=buf.getvalue(), file_name='edited.jpg', mime='image/jpeg')

    # Notes
    st.markdown('---')
    st.subheader('Notes')
    st.write(
        '- This app uses a pretrained **BLIP model** for automatic image captioning.\n'
        '- Works best with clear, well-lit photos.\n'
        '- No local TensorFlow model required.'
    )
