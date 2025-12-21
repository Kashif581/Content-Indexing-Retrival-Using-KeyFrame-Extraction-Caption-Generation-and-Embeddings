from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import streamlit as st

class VideoDB:
    def __init__(self, collection_name="video_scenes"):
        # Use ":memory:" for testing, or a local path for persistence
        self.client = QdrantClient(
            url="https://daefea10-05bc-4a41-8e76-6157f720d8a0.europe-west3-0.gcp.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.O0QvvMbq25gF-UZJLVnRhH1Jy0N8ELuwstv6J5IchPM",
        ) 
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        """Creates the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # We use 384 dimensions for 'all-MiniLM-L6-v2'
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upload_frame_data(self, frame_idx, timestamp, caption, vector):
        """
        Stores frame metadata and vector into Qdrant.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=frame_idx,  # Use frame index as unique ID
                    vector=vector.tolist(),
                    payload={
                        "timestamp": timestamp,
                        "caption": caption,
                        "frame_index": frame_idx
                    }
                )
            ]
        )

    def search_video(self, query_vector, limit=3):
        """Searches the DB for the most relevant frames."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit
        )