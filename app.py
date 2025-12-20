import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import your custom modules
from key_frame_extractor import extract_distinct_frames, extract_deep_features, refine_key_frames
from captions_generator import get_image_caption
from vector_embeddings import get_text_embedding
from store_embeddings import VideoDB

st.set_page_config(layout="wide", page_title="VideoAI Insight", initial_sidebar_state="expanded")

# --- 1. Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .ui-frame {
        border: 2px solid #444;
        border-radius: 20px;
        padding: 20px;
        background-color: #111;
        margin-bottom: 20px;
    }
    [data-testid="stChatMessage"] {
        background-color: #1e1e1e !important;
        border-radius: 15px !important;
        border: 1px solid #333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Initialize Session State & DB ---
if "active_video" not in st.session_state:
    st.session_state.active_video = None
if "messages" not in st.session_state:
    st.session_state.messages = []

db = VideoDB()

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.image("logo.avif", use_container_width=True)
    st.title("Control Panel")
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'], label_visibility="collapsed")
    
    if uploaded_file:
        if st.button("Process Video", use_container_width=True):
            with st.spinner("Analyzing frames & generating AI captions..."):
                # Save bytes to session state for display
                video_bytes = uploaded_file.getvalue()
                st.session_state.active_video = video_bytes
                
                # Write to temp file for OpenCV access
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(video_bytes)
                    tpath = tfile.name


                # PIPELINE EXECUTION
                dq_frames, dq_indices, fps = extract_distinct_frames(tpath)
                
                if dq_frames:
                    deep_feats = extract_deep_features(dq_frames)
                    final_frames, final_indices, final_timestamps = refine_key_frames(dq_frames, dq_indices, deep_feats, fps)
                    
                    # Store mapping of frame_id to image in memory (Qdrant doesn't store raw images well)
                    st.session_state.image_cache = {}
                    
                    for i in range(len(final_frames)):
                        rgb_frame = cv2.cvtColor(final_frames[i], cv2.COLOR_BGR2RGB)
                        caption = get_image_caption(rgb_frame)
                        vector = get_text_embedding(caption)
                        print(final_timestamps)

                        frame_id = int(final_indices[i])
                        db.upload_frame_data(
                            frame_idx=frame_id,
                            timestamp=final_timestamps[i],
                            caption = caption,
                            vector = vector
                        )
                        # Keep image in session state to show in chat later
                        st.session_state.image_cache[frame_id] = rgb_frame
                    
                    
                    st.sidebar.success(f"Processed {len(final_frames)} Key Scenes!")
                
                os.remove(tpath)

# --- 4. Main Content Area ---
st.title("Video Intelligence Dashboard")

if st.session_state.active_video:
    st.markdown('<div class="ui-frame">', unsafe_allow_html=True)
    st.video(st.session_state.active_video)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload and process a video from the sidebar to start chatting.")

st.divider()

# --- 5. Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"], caption=msg.get("img_caption"))

if prompt := st.chat_input("Ask about a scene in the video..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Assistant Logic
    with st.chat_message("assistant"):
        query_vec = get_text_embedding(prompt)
        results = db.search_video(query_vec, limit=1)
        print(results)
        if results:
            # hit = results[0]
            # frame_id = hit.payload['frame_index']
            # timestamp = hit.payload['timestamp']
            # caption = hit.payload['caption']

            # res_text = f"Found at {timestamp}: {caption}"
            res_text = f"Found at {results}"
            st.write(res_text)

            # Retrieve image from our local cache using the ID from Qdrant
            # if "image_cache" in st.session_state and frame_id in st.session_state.image_cache:
            #     matched_img = st.session_state.image_cache[frame_id]
            #     st.image(matched_img)
                
            #     st.session_state.messages.append({
            #         "role": "assistant", "content": res_text, 
            #         "image": matched_img, "img_caption": f"Scene at {timestamp}"
            #     })
        else:
            res_text = "I couldn't find a matching scene in the database."
            st.write(res_text)
            st.session_state.messages.append({"role": "assistant", "content": res_text})