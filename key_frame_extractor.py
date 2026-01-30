import cv2
import numpy as np
import os
import tempfile
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity



import streamlit as st

@st.cache_resource
def load_feature_model():
    base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

feature_extractor = load_feature_model()
# Formatting Time
def format_timestamp(frame_index, fps):
    total_seconds = frame_index / fps
    hours= int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# Converting time into seconds
def get_seconds(time_str):
    """Converts '00:00:42.159' to 42.159 float seconds"""
    try:
        # Splitting HH:MM:SS.mmm
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0

# comparing histogram of two frames
def get_bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# 
def get_policy_thresholds(duration_sec):
    """
    Direct implementation of the Paper's Decision Policy.
    Returns: (histogram_threshold, cnn_similarity_threshold, min_scene_length)
    """
    if duration_sec < 1800:   # < 30 mins: ADAPTIVE
        return 0.12, 0.96, 12 
    elif duration_sec < 7200: # 30-120 mins: FALLBACK
        return 0.18, 0.98, 15 
    else:                     # > 120 mins: CONTENT/REGULAR
        return 0.25, 0.99, 15
    

# --- PHASE 1: FAST HISTOGRAM FILTERING ---
def get_candidate_scenes(video_path, hist_t, minlen):
    """
    Scans the video and finds candidate frames based on color changes.
    Implements the 'Boundary Prediction' logic.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    candidates = []
    prev_hist = None
    last_boundary_time = -minlen
    count = 0
    sample_interval = max(1, int(fps / 2)) # Sample at 2FPS for speed.

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if count % sample_interval == 0:
            current_time = count / fps
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                # Apply paper constraints: change > threshold AND duration > minlen [cite: 35, 111]
                if dist > hist_t and (current_time - last_boundary_time) >= minlen:
                    candidates.append({"frame": frame, "index": count, "time": current_time})
                    last_boundary_time = current_time
            else:
                # Always add the first frame [cite: 31]
                candidates.append({"frame": frame, "index": count, "time": current_time})
            
            prev_hist = hist
        count += 1
    cap.release()
    return candidates, fps


# --- PHASE 2: DEEP SEMANTIC REFINEMENT ---
def refine_with_efficientnet(candidates, cnn_t, model):
    """
    Uses EfficientNetB7 to filter out candidates that are visually 
    similar but had different histograms (e.g., lighting shifts)[cite: 33, 59].
    """
    if not candidates: return []

    # Batch process frames at 600x600 for B7 accuracy
    imgs = [preprocess_input(cv2.resize(c['frame'], (600, 600))) for c in candidates]
    imgs_array = np.array(imgs)
    features = model.predict(imgs_array, batch_size = 8, verbose=0)

    refined = [candidates[0]] # Always keep first candidate
    last_feat = features[0].reshape(1, -1)

    for i in range(1, len(features)):
        curr_feat = features[i].reshape(1, -1)
        sim = cosine_similarity(last_feat, curr_feat)[0][0]

        if sim < cnn_t: # If not semantically similar, it's a new scene [cite: 31, 35]
            refined.append(candidates[i])
            last_feat = curr_feat
    print(refined)    
    return refined




