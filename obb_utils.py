"""
OBB (Oriented Bounding Box) utilities
======================================
Shared functions for OBB IoU computation and mask-to-OBB conversion.

OBB format: 8 floats (x1,y1,x2,y2,x3,y3,x4,y4) — four corner points.
"""

import cv2
import numpy as np
import torch


def mask_to_obb(mask: np.ndarray) -> np.ndarray | None:
    """Convert a boolean H×W mask to an OBB (8 floats: x1,y1,...,x4,y4).

    Uses cv2.minAreaRect to fit the minimum-area rotated rectangle
    to the mask contour.

    Returns:
        (8,) float32 array, or None if mask is empty.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    # Use the largest contour if there are multiple
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) == 0:
        return None

    rect = cv2.minAreaRect(largest)  # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)        # (4, 2) float32 corner points
    return box.reshape(-1).astype(np.float32)  # (8,)


def obb_to_aabb(obb: np.ndarray) -> np.ndarray:
    """Convert OBB (8,) to axis-aligned bounding box (4,) xyxy."""
    pts = obb.reshape(4, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def mask_to_aabb(mask: np.ndarray) -> np.ndarray | None:
    """Compute the tight axis-aligned bounding box of a boolean mask.

    Returns (4,) float32 xyxy, or None if mask is empty.
    Unlike `obb_to_aabb(mask_to_obb(mask))`, this gives the *tight* AABB
    which better matches HBB ground-truth annotations.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32,
    )


def obb_iou(obb_a: np.ndarray, obb_b: np.ndarray) -> float:
    """Compute IoU between two OBBs, each (8,) = x1,y1,...,x4,y4.

    Uses cv2.intersectConvexConvex for the intersection area.
    """
    poly_a = cv2.convexHull(obb_a.reshape(4, 2).astype(np.float32))
    poly_b = cv2.convexHull(obb_b.reshape(4, 2).astype(np.float32))

    inter_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)

    area_a = cv2.contourArea(poly_a)
    area_b = cv2.contourArea(poly_b)
    union = area_a + area_b - inter_area

    return float(inter_area / max(union, 1e-6))


def obb_iou_matrix_np(obbs_a: np.ndarray, obbs_b: np.ndarray) -> np.ndarray:
    """Compute M×N OBB IoU matrix.

    Args:
        obbs_a: (M, 8) float32
        obbs_b: (N, 8) float32

    Returns:
        (M, N) float32 IoU matrix.
    """
    M, N = len(obbs_a), len(obbs_b)
    iou_mat = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            iou_mat[i, j] = obb_iou(obbs_a[i], obbs_b[j])
    return iou_mat


def obb_iou_matrix(obbs_a: torch.Tensor, obbs_b: torch.Tensor) -> torch.Tensor:
    """Compute M×N OBB IoU matrix (torch interface).

    Args:
        obbs_a: (M, 8) float tensor
        obbs_b: (N, 8) float tensor

    Returns:
        (M, N) float tensor IoU matrix.
    """
    a_np = obbs_a.cpu().numpy().astype(np.float32)
    b_np = obbs_b.cpu().numpy().astype(np.float32)
    return torch.from_numpy(obb_iou_matrix_np(a_np, b_np))


def obb_iou_1_vs_n(obb: np.ndarray, obbs: np.ndarray) -> np.ndarray:
    """IoU of one OBB (8,) against N OBBs (N, 8)."""
    N = len(obbs)
    ious = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ious[i] = obb_iou(obb, obbs[i])
    return ious
