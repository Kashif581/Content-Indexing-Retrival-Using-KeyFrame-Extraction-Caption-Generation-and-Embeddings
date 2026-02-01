import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import custom modules
from key_frame_extractor import get_candidate_scenes, refine_with_efficientnet, get_policy_thresholds, feature_extractor, format_timestamp, get_seconds
from captions_generator import get_image_caption
from sklearn.preprocessing import normalize
from vector_embeddings import load_dense_model, get_tfidf_vectorizer, get_sparse_batch, search_with_prf
from store_embeddings import VideoDB

import base64
import streamlit.components.v1 as components

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
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
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


                # -------------------------- KeyFrame Extractions -------------------------------------

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

                    all_captions = []
                    frame_indices = []
                    timestamps = []
    
                    
                    # Store mapping of frame_id to image in memory
                    st.session_state.image_cache = {}
                    # ------------------------- Captions Generation -----------------------------------
                    caption_progress = st.progress(0)
                    for i, scene in enumerate(final_scenes):
                        rgb_frame = cv2.cvtColor(scene['frame'], cv2.COLOR_BGR2RGB)
                        caption = get_image_caption(rgb_frame)

                        all_captions.append(caption)
                        frame_id = scene['index']
                        frame_indices.append(frame_id)

                        t_str = format_timestamp(frame_id, fps)
                        timestamps.append(get_seconds(t_str)) # Store as double for Milvus

                        st.session_state.image_cache[frame_id] = rgb_frame
                        caption_progress.progress((i + 1) / len(final_scenes))

                    # Dense Vectors (E5-small-v2) - Note the 'passage: ' prefix
                    dense_model = load_dense_model()
                    docs_prefixed = [f"passage: {c}" for c in all_captions]
                    dense_vectors = dense_model.encode(docs_prefixed, normalize_embeddings=True)

                    # Sparse Vectors (TF-IDF)
                    # We fit the vectorizer on the current video's context
                    tfidf_vectorizer = get_tfidf_vectorizer(all_captions)
                    st.session_state.tfidf_vectorizer = tfidf_vectorizer
                    sparse_vectors = get_sparse_batch(tfidf_vectorizer, all_captions)
                    sparse_matrix = tfidf_vectorizer.transform(all_captions)
                    # sparse_vectors = sparse_matrix.toarray().astype("float32")
                    # Normalize sparse vectors for Cosine Similarity
                    sparse_vectors = normalize(sparse_vectors, norm='l2', axis=1)

                    # 3. BATCH UPLOAD TO MILVUS
                    st.info("Indexing to Milvus Cloud...")
                    db.insert_batch(
                        frame_indices=frame_indices,
                        timestamps=timestamps,
                        captions=all_captions,
                        dense_vecs=dense_vectors,
                        sparse_vecs=sparse_vectors
                    )
                    
                    
                    st.sidebar.success(f"Processed {len(final_scenes)} Key Scenes!")
                
                os.remove(tpath)

# --- 4. Main Content Area ---
st.title("Video Intelligence Dashboard")

if st.session_state.active_video:
    video_base64 = base64.b64encode(st.session_state.active_video).decode()
    st.markdown('<div class="ui-frame">', unsafe_allow_html=True)
    video_html = f"""
        <video id="myVideo" width="100%" controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload and process a video to start.")

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
        if st.session_state.tfidf_vectorizer is not None:
            # if "tfidf_vectorizer" in st.session_state:
            resutls = search_with_prf(
                    db, 
                    prompt, 
                    st.session_state.tfidf_vectorizer, 
                    load_dense_model()
                    )


            if resutls:
                hit = resutls[0]
                caption = hit.entity.get('caption')
                timestamp = hit.entity.get('timestamp')
                frame_id = hit.entity.get('frame_index')
                # if "image_cache" in st.session_state and frame_id in st.session_state.image_cache:
                #         st.image(st.session_state.image_cache[frame_id], width=400)
                res_text = f"I found this at {timestamp}s: {caption}"
                st.write(res_text)

                # Show match image
                if frame_id in st.session_state.get("image_cache", {}):
                        matched_img = st.session_state.image_cache[frame_id]
                        st.image(matched_img, width=400)

                    # JavaScript Jump Logic
                js_trigger = f"""
                    <script>
                        var v = window.parent.document.getElementById('myVideo');
                        if(v) {{ v.currentTime = {timestamp}; v.play(); }}
                    </script>
                    """
                components.html(js_trigger, height=0)

                st.session_state.messages.append({"role": "assistant", "content": res_text, "image": matched_img})
            else:
                st.write("No matching scenes found.")
                