import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import your custom modules
from key_frame_extractor import get_candidate_scenes, refine_with_efficientnet, get_policy_thresholds, feature_extractor, format_timestamp, get_seconds
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

                # Getting Video Metadata
                cap = cv2.VideoCapture(tpath)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps
                cap.release()

                hist_t, cnn_t, minlen = get_policy_thresholds(duration_sec)


                # PIPELINE EXECUTION
                # Phase 1: Get Candidates (Histogram + Time)
                candidates, fps = get_candidate_scenes(tpath, hist_t, minlen)
                
                
                if candidates:
                    final_scenes = refine_with_efficientnet(candidates, cnn_t, feature_extractor)
                    
                    # Store mapping of frame_id to image in memory (Qdrant doesn't store raw images well)
                    st.session_state.image_cache = {}
                    
                    for i, scene in enumerate(final_scenes):
                        rgb_frame = cv2.cvtColor(scene['frame'], cv2.COLOR_BGR2RGB)
                        caption = get_image_caption(rgb_frame)
                        vector = get_text_embedding(caption)

                        frame_id = scene['index']
                        timestamp_str = format_timestamp(frame_id, fps)
                        db.upload_frame_data(
                            frame_idx=frame_id,
                            timestamp=timestamp_str,
                            caption = caption,
                            vector = vector
                        )
                        # Keep image in session state to show in chat later
                        st.session_state.image_cache[frame_id] = rgb_frame
                    
                    
                    st.sidebar.success(f"Processed {len(final_scenes)} Key Scenes!")
                
                os.remove(tpath)

# --- 4. Main Content Area ---
st.title("Video Intelligence Dashboard")

if st.session_state.active_video:
    st.markdown('<div class="ui-frame">', unsafe_allow_html=True)
    current_start = st.session_state.get("video_start_time", 82.959)
    st.video(st.session_state.active_video, 
             start_time=82.959,
            
             )
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
        print(results.points[0])
        if results.points[0]:
            hit = results.points[0]
            frame_id = hit.payload['frame_index']
            timestamp_str = hit.payload['timestamp']
            caption = hit.payload['caption']

            start_seconds = get_seconds(timestamp_str)
            print(start_seconds)

            res_text = f"Found at {timestamp_str}: {caption}"
            st.write(res_text)


            # ADD THE JUMP BUTTON HERE
            if st.button(f"Play from {timestamp_str}"):
                st.session_state.video_start_time = start_seconds
                st.rerun()

            #Retrieve image from our local cache using the ID from Qdrant
            if "image_cache" in st.session_state and frame_id in st.session_state.image_cache:
                matched_img = st.session_state.image_cache[frame_id]
                st.image(matched_img)
                
                st.session_state.messages.append({
                    "role": "assistant", "content": res_text, 
                    "image": matched_img, "img_caption": f"Scene at {timestamp_str}"
                })
        else:
            res_text = "I couldn't find a matching scene in the database."
            st.write(res_text)
            st.session_state.messages.append({"role": "assistant", "content": res_text})