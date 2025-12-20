from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedder():
    """Loads the lightweight SBERT model for vectorization."""
    # 'all-MiniLM-L6-v2' is optimized for speed and semantic accuracy
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    """Converts a string into a numerical vector embedding."""
    model = load_embedder()
    print(model.encode(text))
    # Returns a 1D numpy array representing the 'meaning' of the text
    return model.encode(text)