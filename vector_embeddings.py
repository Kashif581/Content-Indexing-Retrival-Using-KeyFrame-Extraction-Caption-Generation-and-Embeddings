from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import streamlit as st

@st.cache_resource
def load_dense_model():
    """Loads the E5 model for high-quality semantic vectors."""
    return SentenceTransformer("intfloat/e5-small-v2")

@st.cache_resource
def get_tfidf_vectorizer(all_captions):
    """
    Initializes and fits the TF-IDF vectorizer.
    NOTE: TF-IDF needs to 'see' all captions to understand word importance.
    """
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    if all_captions:
        vectorizer.fit(all_captions)
    return vectorizer

def get_hybrid_embeddings(caption):
    """
    Generates both dense (E5) and sparse (TF-IDF) vectors for a single caption.
    """
    # 1. Generate Dense Vector (E5)
    dense_model = load_dense_model()
    # E5 models REQUIRE the 'passage: ' prefix for document indexing
    prefixed_text = f"passage: {caption}"
    
    dense_vector = dense_model.encode(
        [prefixed_text], 
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")[0]

    return dense_vector

def get_sparse_batch(vectorizer, all_captions):
    """
    Transforms captions into sparse vectors and ensures they match 
    the 5000-dimension schema required by Milvus.
    """
    # Transform to sparse matrix
    sparse_matrix = vectorizer.transform(all_captions)
    
    # Convert to dense array for Milvus float_vector field
    sparse_vectors = sparse_matrix.toarray().astype("float32")
    
    # Check if the vectorizer vocabulary is smaller than 5000
    current_dim = sparse_vectors.shape[1]
    target_dim = 5000
    
    if current_dim < target_dim:
        # Pad with zeros to reach exactly 5000 dimensions
        padding = np.zeros((sparse_vectors.shape[0], target_dim - current_dim), dtype="float32")
        sparse_vectors = np.hstack([sparse_vectors, padding])
    elif current_dim > target_dim:
        # This shouldn't happen with max_features=5000, but just in case:
        sparse_vectors = sparse_vectors[:, :target_dim]
        
    # Normalize for Cosine Similarity
    return normalize(sparse_vectors, norm='l2', axis=1)


from pymilvus import AnnSearchRequest, RRFRanker

def hybrid_search_with_rrf(db, query_text, tfidf_vectorizer, dense_model, top_k=3):
    """
    db: Your VideoDB instance
    query_text: The user's chat input
    tfidf_vectorizer: The vectorizer fitted during the video processing
    dense_model: The E5 model
    """
    
    # 1. Prepare Query Vectors
    # Dense (E5) uses 'query: ' prefix
    query_dense = dense_model.encode([f"query: {query_text}"], normalize_embeddings=True).tolist()
    
    # Sparse (TF-IDF)
    query_sparse = tfidf_vectorizer.transform([query_text]).toarray().astype("float32")
    # Ensure it matches the 5000-dim schema
    if query_sparse.shape[1] < 5000:
        padding = np.zeros((1, 5000 - query_sparse.shape[1]), dtype="float32")
        query_sparse = np.hstack([query_sparse, padding])
    query_sparse = query_sparse.tolist()

    # 2. Hybrid Search Request (Dense + Sparse)
    req_dense = AnnSearchRequest(query_dense, "dense_vector", {"metric_type": "COSINE"}, limit=top_k)
    req_sparse = AnnSearchRequest(query_sparse, "sparse_vector", {"metric_type": "COSINE"}, limit=top_k)

    # 3. Execute Search with RRF (Reciprocal Rank Fusion)
    # This balances the importance of the dense and sparse results
    res = db.collection.hybrid_search(
        [req_dense, req_sparse], 
        rerank=RRFRanker(),
        limit=top_k,
        output_fields=["timestamp", "caption", "frame_index"],
    )
    
    return res[0] # Returns a list of hits


def _pad_sparse_vector(sparse_vector, target_dim=5000):
    """
    Ensures sparse query vectors match the fixed Milvus schema dimension.
    """
    current_dim = sparse_vector.shape[1]

    if current_dim < target_dim:
        padding = np.zeros((sparse_vector.shape[0], target_dim - current_dim), dtype="float32")
        sparse_vector = np.hstack([sparse_vector, padding])
    elif current_dim > target_dim:
        sparse_vector = sparse_vector[:, :target_dim]

    return sparse_vector.astype("float32")


def build_vocab_matrix(_tfidf_vectorizer, _dense_model, min_term_len=3):
    """
    Stage 3 setup:
    Build an embedding matrix for useful TF-IDF vocabulary terms once after
    video processing. These vocabulary embeddings are later used to expand
    the sparse query only.
    """
    generic_terms = {
        "image", "photo", "picture", "scene", "frame", "video", "person",
        "people", "man", "woman", "someone", "something", "object", "thing",
        "background", "foreground", "view", "shot", "camera"
    }

    vocabulary = _tfidf_vectorizer.get_feature_names_out()
    filtered_terms = [
        term for term in vocabulary
        if len(term) >= min_term_len
        and not term.isdigit()
        and term.lower() not in generic_terms
    ]

    if not filtered_terms:
        return [], np.array([], dtype="float32")

    vocab_queries = [f"query: {term}" for term in filtered_terms]
    vocab_matrix = _dense_model.encode(
        vocab_queries,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    return filtered_terms, vocab_matrix


def get_expansion_terms(query_text, dense_model, filtered_terms, vocab_matrix, top_n=3, min_similarity=0.35):
    """
    Stage 3 query expansion:
    Find TF-IDF vocabulary terms that are semantically close to the original
    user query.
    """
    if not filtered_terms or vocab_matrix is None or vocab_matrix.size == 0:
        return []

    query_vector = dense_model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")[0]

    similarities = np.dot(vocab_matrix, query_vector)
    ranked_indexes = np.argsort(similarities)[::-1]

    expansion_terms = []
    query_words = set(query_text.lower().split())
    for idx in ranked_indexes:
        term = filtered_terms[idx]
        if similarities[idx] < min_similarity:
            break
        if term.lower() in query_words:
            continue
        expansion_terms.append(term)
        if len(expansion_terms) == top_n:
            break

    return expansion_terms


def search_with_prf(
    db,
    query_text,
    tfidf_vectorizer,
    dense_model,
    filtered_terms=None,
    vocab_matrix=None,
    top_k=3
):
    """
    Stage 2 + Stage 3 search.

    OLD CODE:
    The app previously returned the direct Stage 2 hybrid search result:
    hybrid_search_with_rrf(db, query_text, tfidf_vectorizer, dense_model, top_k)

    NEW CODE:
    Stage 2 runs the original query, then Stage 3 enriches the sparse query
    with vocabulary terms while keeping the dense query vector unchanged.
    """
    # Stage 2: original hybrid search pass. This is kept because the method
    # describes the first pass before vocabulary-anchored expansion.
    initial_results = hybrid_search_with_rrf(
        db,
        query_text,
        tfidf_vectorizer,
        dense_model,
        top_k=top_k
    )

    expansion_terms = get_expansion_terms(
        query_text,
        dense_model,
        filtered_terms or [],
        vocab_matrix,
        top_n=3
    )

    if not expansion_terms:
        return initial_results

    expanded_sparse_query = f"{query_text} {' '.join(expansion_terms)}"

    # Dense vector stays anchored to the user's original query.
    query_dense = dense_model.encode(
        [f"query: {query_text}"],
        normalize_embeddings=True
    ).tolist()

    # Sparse vector uses the expanded vocabulary-aware query.
    query_sparse = tfidf_vectorizer.transform([expanded_sparse_query]).toarray().astype("float32")
    query_sparse = _pad_sparse_vector(query_sparse).tolist()

    req_dense = AnnSearchRequest(query_dense, "dense_vector", {"metric_type": "COSINE"}, limit=top_k)
    req_sparse = AnnSearchRequest(query_sparse, "sparse_vector", {"metric_type": "COSINE"}, limit=top_k)

    final_results = db.collection.hybrid_search(
        [req_dense, req_sparse],
        rerank=RRFRanker(),
        limit=top_k,
        output_fields=["timestamp", "caption", "frame_index"],
    )

    return final_results[0]

# Note: Sparse vectors are usually calculated in bulk after all captions 
# are generated, because TF-IDF depends on the 'vocabulary' of the whole video.
