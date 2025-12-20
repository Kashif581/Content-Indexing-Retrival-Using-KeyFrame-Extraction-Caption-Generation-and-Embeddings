import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st

@st.cache_resource
def load_fusecap():
    """Initializes the FuseCap model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "noamrot/FuseCap" 
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model, device

def get_image_caption(frame_np):
    """Generates a dense caption for a given numpy image."""
    processor, model, device = load_fusecap()
    
    # Pre-process the image
    raw_image = Image.fromarray(frame_np)
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    
    # Generate tokens
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)
    return caption