"""Pure LFW protocol parsing and verification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    left: tuple[str, int]
    right: tuple[str, int]
    same: bool


def _parse_int(value: str, field: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if result < 1:
        raise ValueError(f"{field} must be positive: {value!r}")
    return result


def parse_pairs(pairs_path: Path) -> list[PairRecord]:
    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("pairs.txt is empty")
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError("pairs.txt header must contain fold count and pairs per fold")
    folds = _parse_int(header[0], "fold count")
    pairs_per_fold = _parse_int(header[1], "pairs per fold")
    expected = folds * pairs_per_fold * 2
    records: list[PairRecord] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) == 3:
            name = fields[0]
            records.append(PairRecord((name, _parse_int(fields[1], "image index")), (name, _parse_int(fields[2], "image index")), True))
        elif len(fields) == 4:
            records.append(PairRecord((fields[0], _parse_int(fields[1], "image index")), (fields[2], _parse_int(fields[3], "image index")), False))
        else:
            raise ValueError(f"Invalid pairs.txt record on line {line_number}: {line!r}")
    if len(records) != expected:
        raise ValueError(f"Expected {expected} pair records, found {len(records)}")
    return records


def build_image_path(dataset_root: Path, image: tuple[str, int]) -> Path:
    identity, index = image
    return dataset_root / "lfw" / identity / f"{identity}_{index:04d}.jpg"


def limit_pairs(records: list[PairRecord], max_pairs: int | None) -> list[PairRecord]:
    if max_pairs is None or max_pairs >= len(records):
        return records
    if max_pairs < 2:
        raise ValueError("max_pairs must be at least 2")
    same = [record for record in records if record.same]
    different = [record for record in records if not record.same]
    same_count = max_pairs // 2
    different_count = max_pairs - same_count
    if same_count > len(same) or different_count > len(different):
        raise ValueError("max_pairs exceeds available same/different records")
    selected: list[PairRecord] = []
    same_index = different_index = 0
    while same_index < same_count or different_index < different_count:
        if same_index < same_count:
            selected.append(same[same_index])
            same_index += 1
        if different_index < different_count:
            selected.append(different[different_index])
            different_index += 1
    return selected


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector")
    return float(np.dot(left / left_norm, right / right_norm))


def accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("scores and labels must have the same non-empty shape")
    return float(np.mean((scores >= threshold) == labels))


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("scores and labels must have the same non-empty shape")
    unique = np.unique(scores)
    candidates = np.concatenate(([unique[0] - 1e-7], (unique[:-1] + unique[1:]) / 2, [unique[-1] + 1e-7]))
    accuracies = np.array([accuracy_at_threshold(scores, labels, value) for value in candidates])
    return float(candidates[int(np.argmax(accuracies))])


def split_folds(scores: np.ndarray, labels: np.ndarray, fold_count: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or scores.size == 0 or fold_count < 2 or scores.size % fold_count:
        raise ValueError("scores must be non-empty, divisible by fold_count, and match labels")
    indices = np.array_split(np.arange(scores.size), fold_count)
    folds = []
    for test_indices in indices:
        train_indices = np.setdiff1d(np.arange(scores.size), test_indices)
        folds.append((scores[train_indices], scores[test_indices], labels[train_indices], labels[test_indices]))
    return folds


def cross_validated_accuracy(scores: Iterable[float], labels: Iterable[bool], fold_count: int) -> dict:
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=bool)
    fold_results = []
    for train_scores, test_scores, train_labels, test_labels in split_folds(score_array, label_array, fold_count):
        threshold = select_threshold(train_scores, train_labels)
        fold_results.append({"threshold": threshold, "accuracy": accuracy_at_threshold(test_scores, test_labels, threshold)})
    accuracies = np.array([item["accuracy"] for item in fold_results])
    return {"folds": fold_results, "accuracy": float(np.mean(accuracies)), "std": float(np.std(accuracies))}
