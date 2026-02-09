import cv2
import base64
import streamlit as st
from ollama import Client

@st.cache_resource
def get_ollama_client():
    """Initializes the Ollama client."""
    # Replace with your actual Ollama host if different
    return Client(
        host="https://ollama.com",
        headers={
            "Authorization": "Bearer a96e2b8b2e88434c9f18de82946f46eb.qMrDL0mQPu5IxVRHukinHe2I"
        }
    )

def get_image_caption(frame_np):
    """
    Generates a dense caption using Gemma 3:27b-cloud.
    Accepts a numpy array (RGB) from the video pipeline.
    """
    client = get_ollama_client()
    
    # 1. Convert Numpy (RGB) to BGR for OpenCV encoding
    # (Only needed if your input is RGB, which it is in your current app.py loop)
    bgr_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    # 2. Encode image to memory buffer as JPEG
    _, buffer = cv2.imencode('.jpg', bgr_frame)
    
    # 3. Convert buffer to Base64 string
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    prompt = """You are an image captioning model. Generate one caption that:
    - Describes only what is clearly visible
    - Uses simple, literal language
    - Follows present tense
    - Mentions main subject, action, and setting
    - Is 12-20 words long and ends with a period."""

    try:
        response = client.chat(
            model="gemma3:27b-cloud",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_base64]
                }
            ]
        )
        caption = response["message"]["content"].strip()
        print(f"Gemma Caption: {caption}")
        return caption
    except Exception as e:
        st.error(f"Ollama API Error: {e}")
        return "Caption generation failed."