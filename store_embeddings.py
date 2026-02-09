from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import streamlit as st
import numpy as np

class VideoDB:
    def __init__(self, collection_name="video_hybrid_search"):
        self.collection_name = collection_name
        self.connect()
        self._setup_collection()

    def connect(self):
        """Connects to Zilliz/Milvus Cloud using Streamlit secrets."""
        connections.connect(
            alias="default",
            uri="https://in03-34dd51e908e5f0a.serverless.aws-eu-central-1.cloud.zilliz.com",
            token="593375de4ee83028c9ccc94016f5dce25203efac93577ce49dff6254ddb47ab2989530fa856eaee87ef5e985579ed7c5c6d38677"
        )

    def _setup_collection(self):
        """Defines the hybrid schema: Dense (384) + Sparse (5000)."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="frame_index", dtype=DataType.INT64),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="sparse_vector", dtype=DataType.FLOAT_VECTOR, dim=5000)
        ]

        schema = CollectionSchema(fields=fields, description="Video Hybrid Search")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create IVF_FLAT indexes for both vectors
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self.collection.create_index(field_name="dense_vector", index_params=index_params)
        self.collection.create_index(field_name="sparse_vector", index_params=index_params)
        self.collection.load()

    def insert_batch(self, frame_indices, timestamps, captions, dense_vecs, sparse_vecs):
        """Inserts processed video data in batches."""
        data = [
            frame_indices,
            timestamps,
            captions,
            dense_vecs.tolist(),
            sparse_vecs.tolist()
        ]
        self.collection.insert(data)
        self.collection.flush()