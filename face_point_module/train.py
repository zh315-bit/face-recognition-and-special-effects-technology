from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HRNet on the 300W face landmark dataset.")
    parser.add_argument("--config", default="configs/hrnet_300w.py", help="Training config path.")
    parser.add_argument(
        "--work-dir",
        default="artifacts/work_dirs/hrnet_300w",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare MMPose annotations before training.",
    )
    return parser


def require_training_dependencies(import_fn: Callable[[str], object] = importlib.import_module) -> None:
    missing_packages = []
    for package_name in ("mmengine", "mmcv", "mmpose"):
        try:
            import_fn(package_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        packages = ", ".join(missing_packages)
        raise RuntimeError(
            "Missing training dependencies: "
            f"{packages}. Install MMPose and its MMCV/MMEngine dependencies before training."
        )


def _load_prepare_dataset(project_root: Path):
    module_path = project_root / "prepare_300w.py"
    spec = importlib.util.spec_from_file_location("prepare_300w", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dataset preparation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_dataset


def _launch_training(config_path: Path, work_dir: Path) -> None:
    from mmengine.config import Config
    from mmengine.runner import Runner

    config = Config.fromfile(config_path)
    config.work_dir = str(work_dir)
    Runner.from_cfg(config).train()


def run_training(
    project_root: Path | str,
    config_path: Path | str,
    work_dir: Path | str,
    prepare: bool,
) -> None:
    root = Path(project_root)
    config = Path(config_path)
    output_dir = Path(work_dir)

    if prepare:
        prepare_dataset = _load_prepare_dataset(root)
        prepare_dataset(root, root / "artifacts" / "annotations")

    require_training_dependencies()
    _launch_training(config, output_dir)


def main() -> int:
    options = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(options.config)
    work_dir = Path(options.work_dir)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not work_dir.is_absolute():
        work_dir = project_root / work_dir
    run_training(project_root, config_path, work_dir, options.prepare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
