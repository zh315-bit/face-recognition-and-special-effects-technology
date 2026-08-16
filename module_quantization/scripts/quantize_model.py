"""Export a ResNet50-IR encoder with CPU dynamic INT8 linear layers."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from face_recognition.model import ResNet50IR


ARTIFACT_FORMAT = "pytorch_dynamic_int8_encoder_state_dict_v1"
REPORT_NAME = "dynamic_quantization_report.json"
RANDOM_SEED = 0


def quantized_path(checkpoint_path: Path) -> Path:
    """Return the default sibling path without changing the source checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_dynamic_int8.pt")


def load_fp32_encoder(checkpoint_path: Path) -> ResNet50IR:
    """Load the encoder portion of a training checkpoint onto CPU in eval mode."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain an encoder state: {checkpoint_path}")

    encoder = ResNet50IR()
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    return encoder


def quantize_dynamic_encoder(encoder: nn.Module) -> nn.Module:
    """Dynamically quantize only encoder linear layers for CPU inference."""
    encoder = encoder.to("cpu").eval()
    return quantize_dynamic(encoder, {nn.Linear}, dtype=torch.qint8, inplace=False)


def quantized_module_names(encoder: nn.Module) -> list[str]:
    """Identify the FP32 linear modules that are represented as INT8 in the export."""
    return [name for name, module in encoder.named_modules() if isinstance(module, nn.Linear)]


def max_embedding_absolute_error(fp32_encoder: nn.Module, int8_encoder: nn.Module) -> float:
    """Compare both encoders using one reproducible 112x112 face-shaped input."""
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    sample = torch.randn((1, 3, 112, 112), generator=generator)
    with torch.inference_mode():
        fp32_embedding = fp32_encoder(sample)
        int8_embedding = int8_encoder(sample)
    return float((fp32_embedding - int8_embedding).abs().max().item())


def export_quantized_encoder(
    checkpoint_path: Path, output_path: Path | None = None, report_path: Path | None = None
) -> tuple[Path, Path, dict[str, object]]:
    """Create an encoder-only dynamic INT8 artifact and its JSON comparison report."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path) if output_path is not None else quantized_path(checkpoint_path)
    report_path = Path(report_path) if report_path is not None else checkpoint_path.parent / REPORT_NAME
    if output_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output path must not overwrite the source checkpoint")
    if report_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Report path must not overwrite the source checkpoint")

    fp32_encoder = load_fp32_encoder(checkpoint_path)
    modules = quantized_module_names(fp32_encoder)
    int8_encoder = quantize_dynamic_encoder(fp32_encoder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "source_checkpoint": str(checkpoint_path),
            "quantized_modules": modules,
            "encoder": int8_encoder.state_dict(),
        },
        output_path,
    )

    report = {
        "source_checkpoint": str(checkpoint_path),
        "output_artifact": str(output_path),
        "source_bytes": checkpoint_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "embedding_input_seed": RANDOM_SEED,
        "max_embedding_absolute_error": max_embedding_absolute_error(fp32_encoder, int8_encoder),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path, report_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="FP32 training checkpoint")
    parser.add_argument("--output", type=Path, help="Encoder-only dynamic INT8 artifact path")
    parser.add_argument("--report", type=Path, help="JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, report_path, report = export_quantized_encoder(
        args.checkpoint, args.output, args.report
    )
    print(f"Saved quantized encoder: {output_path}")
    print(f"Saved report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
