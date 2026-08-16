"""基于 68 点人脸关键点的仿射（相似变换）人脸对齐工具。

典型用法::

    from align_face import align_face

    aligned, matrix = align_face(image, keypoints68, output_size=(112, 112))

其中 ``keypoints68`` 是形状 ``(68, 2)`` 或 ``(68, 3)`` 的 numpy 数组，
坐标为原图像素坐标（MMPose ``inference_topdown`` 的输出即为此坐标系）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# iBUG 68 点关键点索引
EYE_LEFT_IDX = list(range(36, 42))   # 图左眼（人脸右眼）
EYE_RIGHT_IDX = list(range(42, 48))  # 图右眼（人脸左眼）
NOSE_TIP_IDX = 33                     # 鼻尖
MOUTH_LEFT_IDX = 48                   # 图左嘴角
MOUTH_RIGHT_IDX = 54                  # 图右嘴角

# ArcFace/InsightFace 在 112x112 上的标准 5 点参考坐标
# 顺序: 左眼、右眼、鼻尖、左嘴角、右嘴角
ARC_FACE_REFERENCE = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AlignmentQuality:
    """Quality measurements used to decide whether an aligned crop is usable."""

    is_valid: bool
    reason: str | None
    eye_distance: float
    reprojection_error: float
    keypoint_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_keypoints(keypoints68: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints68, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape not in {(68, 2), (68, 3)}:
        raise ValueError("keypoints68 must have shape (68, 2) or (68, 3)")
    if not np.isfinite(keypoints[:, :2]).all():
        raise ValueError("keypoints68 coordinates must be finite")
    return keypoints


def extract_alignment_points(keypoints68: np.ndarray) -> np.ndarray:
    """从 68 点关键点抽取用于对齐的 5 点，返回形状 ``(5, 2)`` 的 float32 数组。"""
    keypoints = _validate_keypoints(keypoints68)[:, :2]

    return np.stack(
        [
            keypoints[EYE_LEFT_IDX].mean(axis=0),
            keypoints[EYE_RIGHT_IDX].mean(axis=0),
            keypoints[NOSE_TIP_IDX],
            keypoints[MOUTH_LEFT_IDX],
            keypoints[MOUTH_RIGHT_IDX],
        ]
    )


def transform_keypoints(keypoints: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """用 ``2x3`` 变换矩阵把关键点变换到对齐后坐标系，返回与输入相同的形状。"""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] not in {2, 3}:
        raise ValueError("keypoints must have shape (N, 2) or (N, 3)")
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 2x3 affine matrix")

    transformed = keypoints.copy()
    transformed[:, :2] = cv2.transform(keypoints[None, :, :2], matrix)[0]
    return transformed


def assess_alignment(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    keypoint_scores: np.ndarray | None = None,
    min_eye_distance: float = 8.0,
    min_keypoint_score: float = 0.3,
    max_reprojection_error: float = 6.0,
) -> AlignmentQuality:
    """Measure landmark geometry and transformation fit before saving a crop."""
    eye_distance = float(np.linalg.norm(source[0] - source[1]))
    projected = cv2.transform(source[None, :, :2], matrix)[0]
    reprojection_error = float(np.linalg.norm(projected - target, axis=1).mean())
    selected_score = None

    if keypoint_scores is not None:
        scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
        if scores.shape != (68,) or not np.isfinite(scores).all():
            raise ValueError("keypoint_scores must have shape (68,) with finite values")
        selected_indices = EYE_LEFT_IDX + EYE_RIGHT_IDX + [NOSE_TIP_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX]
        selected_score = float(scores[selected_indices].mean())

    if eye_distance < min_eye_distance:
        reason = "eye_distance_too_small"
    elif selected_score is not None and selected_score < min_keypoint_score:
        reason = "keypoint_score_too_low"
    elif reprojection_error > max_reprojection_error:
        reason = "reprojection_error_too_high"
    else:
        reason = None
    return AlignmentQuality(reason is None, reason, eye_distance, reprojection_error, selected_score)


def align_face(
    image: np.ndarray,
    keypoints68: np.ndarray,
    output_size=(112, 112),
    reference=None,
    keypoint_scores: np.ndarray | None = None,
    min_keypoint_score: float = 0.3,
    return_quality: bool = False,
):
    """按关键点做人脸对齐。

    Args:
        image: BGR 图像，形状 ``(H, W, 3)``。
        keypoints68: 68 点关键点，形状 ``(68, 2)`` 或 ``(68, 3)``，原图像素坐标。
        output_size: 输出尺寸 ``(width, height)``，默认 ``(112, 112)``。
        reference: 目标参考 5 点，形状 ``(5, 2)``；默认使用 ArcFace 112x112 模板。

    Returns:
        tuple: ``(对齐后的图像, 2x3 相似变换矩阵)``。
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have shape (H, W, 3)")
    if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain two positive dimensions")
    source = extract_alignment_points(keypoints68)

    if reference is None:
        reference = ARC_FACE_REFERENCE
    else:
        reference = np.asarray(reference, dtype=np.float32)
        if reference.shape != (5, 2) or not np.isfinite(reference).all():
            raise ValueError("reference must have shape (5, 2) with finite values")

    # 输出尺寸不是 112x112 时，按比例缩放参考点
    if tuple(output_size) != (112, 112):
        scale = np.array(
            [output_size[0] / 112.0, output_size[1] / 112.0], dtype=np.float32
        )
        target = reference * scale
    else:
        target = reference

    # 相似变换（旋转 + 等比缩放 + 平移，4 自由度），RANSAC 抗离群点
    matrix, _ = cv2.estimateAffinePartial2D(source, target)
    if matrix is None:
        raise RuntimeError("仿射变换估计失败，请检查关键点坐标")

    quality = assess_alignment(
        source, target, matrix, keypoint_scores=keypoint_scores, min_keypoint_score=min_keypoint_score
    )

    aligned = cv2.warpAffine(
        image,
        matrix,
        tuple(output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if return_quality:
        return aligned, matrix, quality
    return aligned, matrix
