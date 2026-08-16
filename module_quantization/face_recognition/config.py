"""Configuration helpers for the face-recognition project."""

from pathlib import Path


def discover_recordio_files(root: Path) -> tuple[Path, Path]:
    """Return the first recursively discovered RecordIO data and index pair."""
    root = Path(root)
    for rec_path in sorted(root.rglob("*.rec")):
        idx_path = rec_path.with_suffix(".idx")
        if idx_path.is_file():
            return rec_path, idx_path
    raise FileNotFoundError(f"No matching .rec/.idx files below {root}")


def read_num_classes(recordio_path: Path) -> int:
    """Read the InsightFace class count from the sibling property file."""
    property_path = Path(recordio_path).with_name("property")
    try:
        value = property_path.read_text(encoding="ascii").strip().split(",", 1)[0]
        count = int(value)
    except (OSError, ValueError, IndexError) as error:
        raise ValueError(f"Invalid or missing dataset property file: {property_path}") from error
    if count < 2:
        raise ValueError(f"Dataset must contain at least two identities: {property_path}")
    return count
