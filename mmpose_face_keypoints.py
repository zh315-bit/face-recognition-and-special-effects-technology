"""Locate face landmarks in MMDetection visualization images with MMPose."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT = "outputs/mmdet_coco_celeba100/vis"
DEFAULT_OUTPUT = "outputs/mmpose_face_keypoints"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for face landmark inference."""
    parser = argparse.ArgumentParser(
        description="Use a pretrained MMPose face model to locate facial landmarks."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input image file or directory. Defaults to previous CelebA detection results.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Directory for landmark visualizations and prediction JSON files.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu.",
    )
    return parser


def run_inference(input_path: str, output_path: str, device: str = "cuda:0") -> None:
    """Run pretrained MMPose face landmark inference and save artifacts."""
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as error:
        raise RuntimeError("MMPose is not installed. Install it after PyTorch is ready.") from error

    input_path_object = Path(input_path)
    if not input_path_object.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path_object}")

    output_path_object = Path(output_path)
    visualization_directory = output_path_object / "vis"
    prediction_directory = output_path_object / "preds"
    output_path_object.mkdir(parents=True, exist_ok=True)

    # The MMPose face alias downloads a pretrained face-landmark model on first use.
    inferencer = MMPoseInferencer(pose2d="face", device=device)
    results = inferencer(
        str(input_path_object),
        vis_out_dir=str(visualization_directory),
        pred_out_dir=str(prediction_directory),
        show=False,
    )
    for _ in results:
        pass


def main() -> int:
    options = build_parser().parse_args()
    try:
        run_inference(options.input, options.output, options.device)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Face landmark inference failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
