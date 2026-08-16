"""Compare FP32 and dynamic INT8 ResNet50-IR encoders on CPU."""

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
import torch
from torch import nn


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluate_lfw import _image_tensor, embed_images, roc_curve, select_pairs_by_fold
from face_recognition.model import ResNet50IR
from lfw_insightface_eval.lfw_protocol import build_image_path, cross_validated_accuracy, parse_pairs
from quantize_model import ARTIFACT_FORMAT, load_fp32_encoder, quantize_dynamic_encoder


WARMUP_RUNS = 3
TIMED_RUNS = 20
BENCHMARK_SEED = 0
REPORT_NAME = "quantization_comparison.json"
PLOT_NAME = "quantization_comparison.png"


def percent_change(reference: float, candidate: float) -> float:
    """Return the percentage change from a reference value to a candidate value."""
    if reference == 0:
        raise ValueError("reference must be non-zero")
    return round((candidate - reference) / reference * 100, 4)


def load_quantized_encoder(artifact_path: Path) -> nn.Module:
    """Reconstruct an encoder from the encoder-only dynamic INT8 state artifact."""
    artifact_path = Path(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT or "encoder" not in artifact:
        raise ValueError(f"Not a {ARTIFACT_FORMAT} artifact: {artifact_path}")

    encoder = quantize_dynamic_encoder(ResNet50IR())
    encoder.load_state_dict(artifact["encoder"])
    encoder.eval()
    return encoder


def benchmark_encoder(encoder: nn.Module, label: str, warmup_runs: int = WARMUP_RUNS, timed_runs: int = TIMED_RUNS) -> dict[str, float | int]:
    """Benchmark one CPU 112x112 image after fixed warmup and timing passes."""
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be positive")
    sample = torch.randn((1, 3, 112, 112), generator=torch.Generator(device="cpu").manual_seed(BENCHMARK_SEED))
    encoder = encoder.to("cpu").eval()
    with torch.inference_mode():
        for _ in range(warmup_runs):
            encoder(sample)
        started = time.perf_counter()
        for run in range(1, timed_runs + 1):
            encoder(sample)
            if run % 5 == 0 or run == timed_runs:
                print(f"Benchmark {label}: {run}/{timed_runs}", flush=True)
        elapsed_seconds = time.perf_counter() - started
    mean_latency_seconds = elapsed_seconds / timed_runs
    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_latency_ms": mean_latency_seconds * 1000,
        "throughput_images_per_second": timed_runs / elapsed_seconds,
    }


def evaluate_encoder(encoder: nn.Module, records, lfw_root: Path, batch_size: int) -> dict:
    """Evaluate an already-loaded CPU encoder on an already-selected LFW pair set."""
    pair_paths = [(build_image_path(lfw_root, record.left), build_image_path(lfw_root, record.right)) for record in records]
    missing_paths = [path for pair in pair_paths for path in pair if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing LFW image: {missing_paths[0]}")
    unique_paths = list(dict.fromkeys(path for pair in pair_paths for path in pair))
    embeddings = embed_images(encoder, unique_paths, torch.device("cpu"), batch_size)
    by_path = dict(zip(unique_paths, embeddings, strict=True))
    scores = np.array([float(np.dot(by_path[left], by_path[right])) for left, right in pair_paths])
    labels = np.array([record.same for record in records], dtype=bool)
    accuracy = cross_validated_accuracy(scores, labels, 10)
    false_positive_rate, true_positive_rate, area = roc_curve(scores, labels)
    return {
        "pair_count": len(records),
        "embedding_count": len(unique_paths),
        "accuracy": accuracy,
        "roc_auc": area,
        "roc": {"false_positive_rate": false_positive_rate.tolist(), "true_positive_rate": true_positive_rate.tolist()},
    }


def output_paths(checkpoint_path: Path) -> tuple[Path, Path]:
    """Return comparison artifacts written beside the FP32 checkpoint."""
    directory = Path(checkpoint_path).parent
    return directory / REPORT_NAME, directory / PLOT_NAME


def save_comparison_plot(report: dict, output_path: Path) -> None:
    """Save a compact visual comparison of evaluation, latency, and artifact size."""
    fp32 = report["fp32"]
    int8 = report["dynamic_int8"]
    metrics = [
        ("LFW accuracy", fp32["evaluation"]["accuracy"]["accuracy"], int8["evaluation"]["accuracy"]["accuracy"]),
        ("ROC AUC", fp32["evaluation"]["roc_auc"], int8["evaluation"]["roc_auc"]),
        ("Latency (ms)", fp32["benchmark"]["mean_latency_ms"], int8["benchmark"]["mean_latency_ms"]),
        ("File size (MB)", fp32["file_size_bytes"] / 1_000_000, int8["file_size_bytes"] / 1_000_000),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9, 6))
    for axis, (title, fp32_value, int8_value) in zip(axes.flat, metrics, strict=True):
        bars = axis.bar(("FP32", "Dynamic INT8"), (fp32_value, int8_value), color=("#4C78A8", "#F58518"))
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.4g", padding=3)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("CPU Quantization Comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_comparison(checkpoint_path: Path, quantized_path: Path, lfw_root: Path, pairs_path: Path, max_pairs: int, batch_size: int, timed_runs: int) -> dict:
    """Measure both artifacts against the exact same stratified LFW records."""
    print("Loading and selecting LFW pairs...", flush=True)
    records = select_pairs_by_fold(parse_pairs(pairs_path), max_pairs)
    print("Loading FP32 encoder on CPU...", flush=True)
    fp32_encoder = load_fp32_encoder(checkpoint_path)
    print("Loading dynamic INT8 encoder on CPU...", flush=True)
    int8_encoder = load_quantized_encoder(quantized_path)
    print("Evaluating FP32 encoder on LFW...", flush=True)
    fp32_evaluation = evaluate_encoder(fp32_encoder, records, lfw_root, batch_size)
    print("Evaluating dynamic INT8 encoder on LFW...", flush=True)
    int8_evaluation = evaluate_encoder(int8_encoder, records, lfw_root, batch_size)
    print("Benchmarking FP32 CPU latency...", flush=True)
    fp32_benchmark = benchmark_encoder(fp32_encoder, "FP32", timed_runs=timed_runs)
    print("Benchmarking dynamic INT8 CPU latency...", flush=True)
    int8_benchmark = benchmark_encoder(int8_encoder, "Dynamic INT8", timed_runs=timed_runs)
    fp32_size = Path(checkpoint_path).stat().st_size
    int8_size = Path(quantized_path).stat().st_size
    return {
        "checkpoint": str(checkpoint_path),
        "quantized_artifact": str(quantized_path),
        "lfw_root": str(lfw_root),
        "pairs_file": str(pairs_path),
        "fp32": {"file_size_bytes": fp32_size, "evaluation": fp32_evaluation, "benchmark": fp32_benchmark},
        "dynamic_int8": {"file_size_bytes": int8_size, "evaluation": int8_evaluation, "benchmark": int8_benchmark},
        "changes_percent": {
            "file_size": percent_change(fp32_size, int8_size),
            "mean_latency": percent_change(fp32_benchmark["mean_latency_ms"], int8_benchmark["mean_latency_ms"]),
            "throughput": percent_change(fp32_benchmark["throughput_images_per_second"], int8_benchmark["throughput_images_per_second"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="FP32 training checkpoint")
    parser.add_argument("--quantized", type=Path, required=True, help="Encoder-only dynamic INT8 artifact")
    parser.add_argument("--lfw-root", type=Path, default=WORKSPACE_ROOT / "lfw")
    parser.add_argument("--pairs", type=Path, default=None, help="Official pairs.txt; defaults to --lfw-root/pairs.txt")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timed-runs", type=int, default=TIMED_RUNS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs_path = args.pairs or args.lfw_root / "pairs.txt"
    report_path, plot_path = output_paths(args.checkpoint)
    try:
        report = run_comparison(args.checkpoint, args.quantized, args.lfw_root, pairs_path, args.max_pairs, args.batch_size, args.timed_runs)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Comparison failed: {error}")
        return 1
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_comparison_plot(report, plot_path)
    for label, result in (("FP32", report["fp32"]), ("Dynamic INT8", report["dynamic_int8"])):
        print(f"{label}: accuracy={result['evaluation']['accuracy']['accuracy']:.4f}, ROC AUC={result['evaluation']['roc_auc']:.4f}, latency={result['benchmark']['mean_latency_ms']:.3f} ms, throughput={result['benchmark']['throughput_images_per_second']:.2f} img/s, size={result['file_size_bytes']} bytes")
    print(f"Changes: size={report['changes_percent']['file_size']:.2f}%, latency={report['changes_percent']['mean_latency']:.2f}%, throughput={report['changes_percent']['throughput']:.2f}%")
    print(f"Saved report to {report_path}")
    print(f"Saved plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
