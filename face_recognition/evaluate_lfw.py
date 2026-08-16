"""Evaluate a ResNet50-IR encoder checkpoint on official LFW pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from face_recognition.model import ResNet50IR
from face_recognition.lfw_protocol import (
    build_image_path,
    cross_validated_accuracy,
    parse_pairs,
)


def output_paths(checkpoint_path: Path, pair_count: int, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Return the metrics and ROC output paths for an evaluation run."""
    directory = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    return directory / f"lfw_{pair_count}_metrics.json", directory / f"lfw_{pair_count}_roc.png"


def select_pairs_by_fold(records, max_pairs: int, fold_count: int = 10):
    """Select equal same/different pairs from every official LFW fold."""
    if max_pairs % fold_count or (max_pairs // fold_count) % 2:
        raise ValueError("max-pairs must be divisible by 20 for balanced ten-fold evaluation")
    if len(records) % fold_count:
        raise ValueError("Official LFW pair count must be divisible by the fold count")
    per_fold = max_pairs // fold_count
    selected = []
    for fold in range(fold_count):
        candidates = records[fold * (len(records) // fold_count) : (fold + 1) * (len(records) // fold_count)]
        same = [record for record in candidates if record.same][: per_fold // 2]
        different = [record for record in candidates if not record.same][: per_fold // 2]
        if len(same) != per_fold // 2 or len(different) != per_fold // 2:
            raise ValueError(f"LFW fold {fold + 1} lacks the requested pair balance")
        selected.extend(same + different)
    return selected


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC endpoints and trapezoidal AUC without sklearn."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same non-empty shape")
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC requires at least one same and one different pair")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels, dtype=float)
    false_positives = np.cumsum(~sorted_labels, dtype=float)
    final_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1]
    true_positive_rate = np.r_[0.0, true_positives[final_indices] / positive_count]
    false_positive_rate = np.r_[0.0, false_positives[final_indices] / negative_count]
    return false_positive_rate, true_positive_rate, float(np.trapz(true_positive_rate, false_positive_rate))


def load_encoder(checkpoint_path: Path, device: torch.device) -> ResNet50IR:
    """Load only the encoder weights from a training checkpoint in eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain an encoder state: {checkpoint_path}")
    encoder = ResNet50IR().to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    return encoder


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB").resize((112, 112))
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).sub(0.5).div(0.5)


def embed_images(model: ResNet50IR, paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    """Produce fused, normalized original-plus-horizontal-flip embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([_image_tensor(path) for path in batch_paths]).to(device)
            features = model(torch.cat((images, torch.flip(images, dims=(3,))), dim=0))
            original, flipped = features.chunk(2, dim=0)
            vectors.append(F.normalize(F.normalize(original, dim=1) + F.normalize(flipped, dim=1), dim=1).cpu())
    return torch.cat(vectors).numpy() if vectors else np.empty((0, 512), dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lfw-root", type=Path, default=WORKSPACE_ROOT / "lfw")
    parser.add_argument("--pairs", type=Path, default=None, help="Official pairs.txt; defaults to --lfw-root/pairs.txt")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run_evaluation(
    checkpoint_path: Path,
    lfw_root: Path,
    pairs_path: Path,
    max_pairs: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    records = select_pairs_by_fold(parse_pairs(pairs_path), max_pairs)
    if len(records) % 10:
        raise ValueError("The selected pair count must be divisible by 10 for the LFW protocol")
    pair_paths = [(build_image_path(lfw_root, record.left), build_image_path(lfw_root, record.right)) for record in records]
    missing_paths = [path for pair in pair_paths for path in pair if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing LFW image: {missing_paths[0]}")

    unique_paths = list(dict.fromkeys(path for pair in pair_paths for path in pair))
    encoder = load_encoder(checkpoint_path, device)
    embeddings = embed_images(encoder, unique_paths, device, batch_size)
    by_path = dict(zip(unique_paths, embeddings, strict=True))
    scores = np.array([float(np.dot(by_path[left], by_path[right])) for left, right in pair_paths])
    labels = np.array([record.same for record in records], dtype=bool)
    accuracy = cross_validated_accuracy(scores, labels, 10)
    false_positive_rate, true_positive_rate, area = roc_curve(scores, labels)
    return {
        "checkpoint": str(checkpoint_path),
        "lfw_root": str(lfw_root),
        "pairs_file": str(pairs_path),
        "pair_count": len(records),
        "embedding_count": len(unique_paths),
        "device": str(device),
        "accuracy": accuracy,
        "roc_auc": area,
        "roc": {"false_positive_rate": false_positive_rate.tolist(), "true_positive_rate": true_positive_rate.tolist()},
    }


def save_roc_plot(false_positive_rate: list[float], true_positive_rate: list[float], area: float, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {area:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="LFW ROC")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    pairs_path = args.pairs or args.lfw_root / "pairs.txt"
    metrics_path, plot_path = output_paths(args.checkpoint, args.max_pairs, args.output_dir)
    started = time.perf_counter()
    try:
        result = run_evaluation(args.checkpoint, args.lfw_root, pairs_path, args.max_pairs, device, args.batch_size)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Evaluation failed: {error}")
        return 1
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_roc_plot(result["roc"]["false_positive_rate"], result["roc"]["true_positive_rate"], result["roc_auc"], plot_path)
    print(f"Accuracy: {result['accuracy']['accuracy']:.4f} +/- {result['accuracy']['std']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved ROC plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
