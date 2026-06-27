"""
Research retrieval validation pipeline.

This script builds two separate files from the same keyframe dataset:
1. train/index file: one stable caption per image/keyframe, embedded into Milvus
2. test/query file: multiple query styles per image, used only for evaluation

Example:
    python retrieval_evaluation.py --video-path sample.mp4 --top-k 1 5 10
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from captions_generator import get_image_caption
from key_frame_extractor import (
    feature_extractor,
    format_timestamp,
    get_candidate_scenes,
    get_policy_thresholds,
    get_seconds,
    refine_with_efficientnet,
)
from store_embeddings import VideoDB
from vector_embeddings import (
    get_sparse_batch,
    get_tfidf_vectorizer,
    load_dense_model,
    hybrid_search_with_rrf,
)


DEFAULT_OUTPUT_DIR = Path("evaluation_outputs")
CAPTION_MAX_CHARS = 500


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def clean_caption_for_milvus(caption: str) -> str:
    """Respect the current Milvus schema: caption VARCHAR max_length=500."""
    caption = " ".join(caption.split())
    if len(caption) <= CAPTION_MAX_CHARS:
        return caption
    return caption[: CAPTION_MAX_CHARS - 1].rstrip() + "."


def extract_keyframes(video_path: Path) -> tuple[List[Dict], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not fps or fps <= 0:
        raise ValueError(f"Could not read FPS from video: {video_path}")

    duration_sec = total_frames / fps
    hist_t, cnn_t, minlen = get_policy_thresholds(duration_sec)
    candidates, fps = get_candidate_scenes(str(video_path), hist_t, minlen)
    return refine_with_efficientnet(candidates, cnn_t, feature_extractor), fps


def build_train_records(video_path: Path) -> List[Dict]:
    scenes, fps = extract_keyframes(video_path)
    records = []

    for rank, scene in enumerate(scenes):
        frame_index = int(scene["index"])
        timestamp = get_seconds(format_timestamp(frame_index, fps))
        rgb_frame = cv2.cvtColor(scene["frame"], cv2.COLOR_BGR2RGB)
        caption = clean_caption_for_milvus(get_image_caption(rgb_frame))

        records.append(
            {
                "image_id": str(frame_index),
                "frame_index": frame_index,
                "timestamp": timestamp,
                "caption": caption,
                "source_video": str(video_path),
                "rank": rank,
            }
        )

    return records


def make_sparse_query(caption: str, max_terms: int = 10) -> str:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "this",
        "that",
        "there",
        "shows",
        "showing",
    }
    words = []
    for raw_word in caption.lower().replace(".", " ").replace(",", " ").split():
        word = "".join(ch for ch in raw_word if ch.isalnum())
        if len(word) > 2 and word not in stop_words and word not in words:
            words.append(word)
        if len(words) >= max_terms:
            break
    return " ".join(words)


def generate_queries_from_caption(caption: str) -> Dict[str, str]:
    compact_caption = " ".join(caption.split())
    caption_without_period = compact_caption.rstrip(".")

    return {
        "easy": caption_without_period,
        "medium": f"A scene showing {caption_without_period.lower()}",
        "hard": f"An image about the main visual idea and context of: {caption_without_period.lower()}",
        "sparse": make_sparse_query(compact_caption),
    }


def build_test_records(train_records: List[Dict]) -> List[Dict]:
    test_records = []

    for record in train_records:
        queries = generate_queries_from_caption(record["caption"])
        for query_type, query_text in queries.items():
            test_records.append(
                {
                    "query_id": f"{record['image_id']}::{query_type}",
                    "image_id": record["image_id"],
                    "frame_index": record["frame_index"],
                    "query_type": query_type,
                    "query": query_text,
                    "source_caption": record["caption"],
                }
            )

    return test_records


def index_train_records(db: VideoDB, train_records: List[Dict]) -> object:
    captions = [record["caption"] for record in train_records]
    frame_indices = [int(record["frame_index"]) for record in train_records]
    timestamps = [float(record["timestamp"]) for record in train_records]

    dense_model = load_dense_model()
    dense_vectors = dense_model.encode(
        [f"passage: {caption}" for caption in captions],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    tfidf_vectorizer = get_tfidf_vectorizer(captions)
    sparse_vectors = np.asarray(get_sparse_batch(tfidf_vectorizer, captions), dtype="float32")

    db.insert_batch(
        frame_indices=frame_indices,
        timestamps=timestamps,
        captions=captions,
        dense_vecs=dense_vectors,
        sparse_vecs=sparse_vectors,
    )

    return tfidf_vectorizer


def rank_of_expected_frame(hits, expected_frame_index: int) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if int(hit.entity.get("frame_index")) == int(expected_frame_index):
            return rank
    return None


def evaluate_queries(
    db: VideoDB,
    test_records: List[Dict],
    tfidf_vectorizer,
    top_k_values: List[int],
) -> tuple[Dict, List[Dict]]:
    dense_model = load_dense_model()
    max_k = max(top_k_values)
    detailed_results = []

    for record in test_records:
        hits = hybrid_search_with_rrf(
            db=db,
            query_text=record["query"],
            tfidf_vectorizer=tfidf_vectorizer,
            dense_model=dense_model,
            top_k=max_k,
        )
        expected_rank = rank_of_expected_frame(hits, record["frame_index"])
        retrieved_frame_indices = [int(hit.entity.get("frame_index")) for hit in hits]

        detailed_results.append(
            {
                "query_id": record["query_id"],
                "query_type": record["query_type"],
                "query": record["query"],
                "expected_frame_index": int(record["frame_index"]),
                "retrieved_frame_indices": retrieved_frame_indices,
                "rank": expected_rank,
                "reciprocal_rank": 0.0 if expected_rank is None else 1.0 / expected_rank,
            }
        )

    total = len(detailed_results)
    metrics = {
        "total_queries": total,
        "mrr": (
            sum(result["reciprocal_rank"] for result in detailed_results) / total
            if total
            else 0.0
        ),
    }

    for k in top_k_values:
        hits_at_k = [
            result
            for result in detailed_results
            if result["rank"] is not None and result["rank"] <= k
        ]
        metrics[f"recall@{k}"] = len(hits_at_k) / total if total else 0.0

    for query_type in sorted({record["query_type"] for record in test_records}):
        subset = [
            result
            for result in detailed_results
            if result["query_type"] == query_type
        ]
        if not subset:
            continue
        metrics[f"{query_type}_mrr"] = sum(
            result["reciprocal_rank"] for result in subset
        ) / len(subset)
        for k in top_k_values:
            metrics[f"{query_type}_recall@{k}"] = len(
                [
                    result
                    for result in subset
                    if result["rank"] is not None and result["rank"] <= k
                ]
            ) / len(subset)

    return metrics, detailed_results


def write_metrics_csv(path: Path, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval evaluation.")
    parser.add_argument("--video-path", type=Path, help="Video used to build keyframe dataset.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "train_index.jsonl",
        help="Path for the train/index JSONL file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "test_queries.jsonl",
        help="Path for the test/query JSONL file.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "retrieval_results.jsonl",
        help="Path for per-query retrieval results.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "retrieval_metrics.csv",
        help="Path for aggregate metrics CSV.",
    )
    parser.add_argument(
        "--collection-name",
        default="video_hybrid_eval",
        help="Milvus collection name for evaluation indexing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Top-K values for Recall@K.",
    )
    parser.add_argument(
        "--reuse-files",
        action="store_true",
        help="Reuse existing train/test files instead of regenerating captions and queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reuse_files:
        train_records = read_jsonl(args.train_file)
        test_records = read_jsonl(args.test_file)
    else:
        if args.video_path is None:
            raise ValueError("--video-path is required unless --reuse-files is set.")
        train_records = build_train_records(args.video_path)
        test_records = build_test_records(train_records)
        write_jsonl(args.train_file, train_records)
        write_jsonl(args.test_file, test_records)

    if not train_records:
        raise ValueError("No train records found. Check keyframe extraction or train file.")
    if not test_records:
        raise ValueError("No test records found. Check query generation or test file.")

    db = VideoDB(collection_name=args.collection_name)
    tfidf_vectorizer = index_train_records(db, train_records)
    metrics, detailed_results = evaluate_queries(
        db=db,
        test_records=test_records,
        tfidf_vectorizer=tfidf_vectorizer,
        top_k_values=sorted(set(args.top_k)),
    )

    write_jsonl(args.results_file, detailed_results)
    write_metrics_csv(args.metrics_file, metrics)

    print("Evaluation complete.")
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Results file: {args.results_file}")
    print(f"Metrics file: {args.metrics_file}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
