"""Run InsightFace verification on the official LFW pairs protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lfw_protocol import build_image_path, cross_validated_accuracy, cosine_similarity, limit_pairs, parse_pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("lfw"))
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--det-size", type=int, default=320, help="Face detector input size (default: 320)")
    parser.add_argument("--output", type=Path, default=Path("lfw_insightface_eval/outputs/lfw_metrics.json"))
    parser.add_argument("--save-pairs", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=100, help="Maximum official pairs to evaluate (default: 100)")
    return parser.parse_args()


def _load_analyzer(model_name: str, provider: str, det_size: int):
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError("InsightFace and OpenCV are required; install requirements.txt first") from exc
    analyzer = FaceAnalysis(name=model_name, providers=[provider])
    analyzer.prepare(ctx_id=0 if provider != "CPUExecutionProvider" else -1, det_size=(det_size, det_size))
    return analyzer, cv2


def run_evaluation(dataset_root: Path, pairs_path: Path, model_name: str, provider: str, save_pairs: bool = False, max_pairs: int | None = 50, det_size: int = 320) -> dict:
    records = limit_pairs(parse_pairs(pairs_path), max_pairs)
    analyzer, cv2 = _load_analyzer(model_name, provider, det_size)
    embedding_cache: dict[Path, np.ndarray] = {}
    pair_results = []
    failures = []

    def embedding(path: Path) -> np.ndarray:
        if path in embedding_cache:
            return embedding_cache[path]
        if not path.is_file():
            raise FileNotFoundError(path)
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {path}")
        faces = analyzer.get(image)
        if not faces:
            raise RuntimeError(f"No face detected: {path}")
        largest = max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))
        vector = getattr(largest, "normed_embedding", None)
        if vector is None:
            vector = largest.embedding
        vector = np.asarray(vector, dtype=np.float32)
        embedding_cache[path] = vector
        return vector

    for record in records:
        left_path = build_image_path(dataset_root, record.left)
        right_path = build_image_path(dataset_root, record.right)
        try:
            score = cosine_similarity(embedding(left_path), embedding(right_path))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append({"left": str(left_path), "right": str(right_path), "error": str(exc)})
            continue
        pair_results.append({"score": score, "same": record.same, "left": str(left_path), "right": str(right_path)})

    if failures:
        raise RuntimeError(f"Failed to evaluate {len(failures)} pair(s); first failure: {failures[0]}")
    scores = np.array([item["score"] for item in pair_results], dtype=float)
    labels = np.array([item["same"] for item in pair_results], dtype=bool)
    metrics = cross_validated_accuracy(scores, labels, 10)
    result = {
        "model": model_name,
        "provider": provider,
        "dataset_root": str(dataset_root),
        "pairs_file": str(pairs_path),
        "pair_count": len(pair_results),
        "embedding_count": len(embedding_cache),
        "metrics": metrics,
    }
    if save_pairs:
        result["pairs"] = pair_results
    return result


def main() -> int:
    args = _parse_args()
    pairs_path = args.pairs or args.dataset_root / "pairs.txt"
    started = time.perf_counter()
    try:
        result = run_evaluation(args.dataset_root, pairs_path, args.model, args.provider, args.save_pairs, args.max_pairs, args.det_size)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
