import cv2
import numpy as np
import os
import tempfile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity


# Initialize model once at the module level to avoid reloading it
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


def format_timestamp(frame_index, fps):
    total_seconds = frame_index / fps
    hours= int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def get_seconds(time_str):
    """Converts '00:00:42.159' to 42.159 float seconds"""
    try:
        # Splitting HH:MM:SS.mmm
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0

def get_bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def extract_distinct_frames(video_path, histogram_threshold=0.15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    dq_k_frames, frame_indices = [], []
    prev_frame_hist = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_frame_hist is None:
            dq_k_frames.append(frame)
            frame_indices.append(frame_count)
        else:
            Sm = get_bhattacharyya_distance(prev_frame_hist, hist)
            if Sm > histogram_threshold:
                dq_k_frames.append(frame)
                frame_indices.append(frame_count)

        prev_frame_hist = hist
        frame_count += 1

    cap.release()
    return dq_k_frames, frame_indices, fps


def extract_deep_features(frame_list):
    features = []
    for frame in frame_list:
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature_vector = feature_extractor.predict(img, verbose=0)
        features.append(feature_vector.flatten())
    return np.array(features)


def refine_key_frames(dq_k_frames, frame_indices, feature_vectors, fps, similarity_threshold=0.98):
    if not dq_k_frames: return [], [], []
    final_key_frames, final_indices, final_timestamps = [dq_k_frames[0]], [frame_indices[0]], [format_timestamp(frame_indices[0], fps)]
    last_key_frame_feature = feature_vectors[0].reshape(1, -1)

    for i in range(1, len(dq_k_frames)):
        current_frame_feature = feature_vectors[i].reshape(1, -1)
        similarity = cosine_similarity(last_key_frame_feature, current_frame_feature)[0][0]
        if similarity < similarity_threshold:
            final_key_frames.append(dq_k_frames[i])
            final_indices.append(frame_indices[i])

            final_timestamps.append(format_timestamp(frame_indices[i], fps))
            last_key_frame_feature = current_frame_feature
    return final_key_frames, final_indices, final_timestamps