"""Export an FP32 ResNet50-IR encoder to ONNX and validate it on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from face_recognition.model import ResNet50IR


ONNX_OPSET = 17
INPUT_NAME = "images"
OUTPUT_NAME = "embeddings"
REPORT_NAME = "onnx_export_report.json"
RANDOM_SEED = 0
WARMUP_RUNS = 10
TIMED_RUNS = 20
MAX_ABSOLUTE_ERROR = 1e-4


def onnx_path(checkpoint_path: Path) -> Path:
    """Return the default ONNX path beside an FP32 checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(".onnx")


def load_fp32_encoder(checkpoint_path: Path) -> ResNet50IR:
    """Load an FP32 encoder state from a training checkpoint onto CPU."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain an encoder state: {checkpoint_path}")

    encoder_state = checkpoint["encoder"]
    if not isinstance(encoder_state, dict):
        raise ValueError(f"Checkpoint encoder state is invalid: {checkpoint_path}")
    non_fp32_weights = [
        name
        for name, value in encoder_state.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32
    ]
    if non_fp32_weights:
        raise ValueError(f"Checkpoint is not an FP32 encoder: {checkpoint_path}")

    encoder = ResNet50IR()
    encoder.load_state_dict(encoder_state)
    return encoder.cpu().eval()


def deterministic_images() -> torch.Tensor:
    """Create a reproducible single aligned-face-sized FP32 input tensor."""
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    return torch.randn((1, 3, 112, 112), generator=generator, dtype=torch.float32)


def export_encoder(encoder: ResNet50IR, output_path: Path, sample: torch.Tensor) -> None:
    """Export an eval-mode encoder with stable IO names and a dynamic batch axis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            encoder.cpu().eval(),
            sample,
            output_path,
            opset_version=ONNX_OPSET,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        )
    onnx.checker.check_model(str(output_path))


def run_onnx_cpu(output_path: Path, images: np.ndarray) -> tuple[ort.InferenceSession, np.ndarray]:
    """Run the exported model with ONNX Runtime's CPU execution provider only."""
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime did not select the CPU execution provider")
    embeddings = session.run([OUTPUT_NAME], {INPUT_NAME: images})[0]
    return session, embeddings


def benchmark_onnx_cpu(session: ort.InferenceSession, images: np.ndarray) -> dict[str, float | int]:
    """Benchmark one image with exactly ten warmups and twenty timed CPU passes."""
    for _ in range(WARMUP_RUNS):
        session.run([OUTPUT_NAME], {INPUT_NAME: images})

    started = time.perf_counter()
    for _ in range(TIMED_RUNS):
        session.run([OUTPUT_NAME], {INPUT_NAME: images})
    elapsed_seconds = time.perf_counter() - started
    return {
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "mean_latency_ms": elapsed_seconds / TIMED_RUNS * 1000,
        "throughput_images_per_second": TIMED_RUNS / elapsed_seconds,
    }


def export_and_validate(
    checkpoint_path: Path, output_path: Path | None = None, report_path: Path | None = None
) -> tuple[Path, Path, dict[str, object]]:
    """Export one FP32 checkpoint encoder, compare outputs, and write a JSON report."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path) if output_path is not None else onnx_path(checkpoint_path)
    report_path = Path(report_path) if report_path is not None else checkpoint_path.parent / REPORT_NAME
    if output_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output path must not overwrite the source checkpoint")
    if report_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Report path must not overwrite the source checkpoint")

    encoder = load_fp32_encoder(checkpoint_path)
    sample = deterministic_images()
    export_encoder(encoder, output_path, sample)

    with torch.inference_mode():
        pytorch_embeddings = encoder(sample).numpy()
    session, onnx_embeddings = run_onnx_cpu(output_path, sample.numpy())
    max_error = float(np.max(np.abs(pytorch_embeddings - onnx_embeddings)))
    if max_error > MAX_ABSOLUTE_ERROR:
        raise RuntimeError(
            f"ONNX output error {max_error:.8g} exceeds {MAX_ABSOLUTE_ERROR:.8g}"
        )

    report = {
        "source_checkpoint": str(checkpoint_path),
        "onnx_model": str(output_path),
        "onnx_opset": ONNX_OPSET,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "dynamic_batch": True,
        "input_seed": RANDOM_SEED,
        "max_embedding_absolute_error": max_error,
        "max_allowed_absolute_error": MAX_ABSOLUTE_ERROR,
        "onnxruntime_cpu_benchmark": benchmark_onnx_cpu(session, sample.numpy()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path, report_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="FP32 training checkpoint")
    parser.add_argument("--output", type=Path, help="ONNX model output path")
    parser.add_argument("--report", type=Path, help="JSON validation report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path, report_path, report = export_and_validate(
        args.checkpoint, args.output, args.report
    )
    print(f"Saved ONNX encoder: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Max embedding absolute error: {report['max_embedding_absolute_error']:.8g}")
    print(f"ONNX Runtime CPU mean latency: {report['onnxruntime_cpu_benchmark']['mean_latency_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
