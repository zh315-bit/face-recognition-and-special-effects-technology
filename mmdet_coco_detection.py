"""Use an MMDetection COCO checkpoint to detect and visualize objects."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


DEFAULT_MODEL = "rtmdet_tiny_8xb32-300e_coco"
DEFAULT_OUTPUT = "outputs/mmdet_coco"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="使用 MMDetection COCO 预训练模型进行目标检测并保存结果。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入图片路径，或包含图片的目录。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"结果输出目录，默认: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MMDetection 模型名称或配置文件路径，默认: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="可选的本地 checkpoint (.pth) 路径；不指定时按模型名称自动下载。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="推理设备，例如 cpu、cuda:0，默认使用 cpu。",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="置信度阈值，默认: 0.3。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="当输入为目录时，只检测按文件名排序后的前 N 张图片。",
    )
    parser.add_argument(
        "--no-predictions",
        dest="save_predictions",
        action="store_false",
        help="只保存可视化图片，不保存 JSON 检测结果。",
    )
    parser.set_defaults(save_predictions=True)
    return parser


def run_inference(
    input_path: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    weights: str | None = None,
    device: str = "cpu",
    score_thr: float = 0.3,
    save_predictions: bool = True,
    max_images: int | None = None,
) -> None:
    """运行 MMDetection 推理，并保存 vis/ 与 preds/ 结果。"""
    try:
        from mmdet.apis import DetInferencer
    except ImportError as error:
        raise RuntimeError(
            "未安装 MMDetection。请先安装 PyTorch、MMEngine、MMCV 和 MMDetection。"
        ) from error

    input_path_object = Path(input_path)
    if not input_path_object.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path_object}")
    if not 0 <= score_thr <= 1:
        raise ValueError("--score-thr 必须在 0 到 1 之间。")
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images 必须是正整数。")

    output_path_object = Path(output_path)
    output_path_object.mkdir(parents=True, exist_ok=True)

    # 官方文档支持使用模型名称自动下载权重，也支持传入本地配置和 checkpoint。
    inferencer = DetInferencer(
        model=model,
        weights=weights,
        device=device,
    )
    inputs: str | list[str] = str(input_path_object)
    if input_path_object.is_dir() and max_images is not None:
        image_paths = sorted(
            [
                path
                for path in input_path_object.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ],
            key=lambda path: (
                0,
                int(path.stem),
            ) if path.stem.isdigit() else (1, path.name),
        )
        inputs = [str(path) for path in image_paths[:max_images]]
        if not inputs:
            raise ValueError(f"目录中未找到可读取的图片: {input_path_object}")

    inferencer(
        inputs,
        out_dir=str(output_path_object),
        pred_score_thr=score_thr,
        no_save_vis=False,
        no_save_pred=not save_predictions,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """命令行入口；返回 0 表示成功，1 表示参数或运行失败。"""
    options = build_parser().parse_args(arguments)
    try:
        run_inference(
            input_path=options.input,
            output_path=options.output,
            model=options.model,
            weights=options.weights,
            device=options.device,
            score_thr=options.score_thr,
            save_predictions=options.save_predictions,
            max_images=options.max_images,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"检测失败: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
