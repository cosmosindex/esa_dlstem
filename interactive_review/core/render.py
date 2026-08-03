"""Shared drawing primitives and the clickable annotation canvas.

The per-track views this module used to hold — a six-crop strip, a coverage bar,
a whole-frame "did the ground truth miss this object" view — are gone with the
track-level workflow that needed them: whether an object belongs in the ground
truth is now decided by ``tools/merge_det_to_mot.py``, not by a human. What
remains is what the video-level review still draws with; the views themselves
live in :mod:`.vrender`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .paths import frame_path, sequence_by_id

# BGR-free: everything here works in RGB, which is what Gradio expects.
COLOR_TARGET = (220, 40, 40)      # the object under the cursor
COLOR_MOT_GT = (40, 200, 90)      # shipped with the dataset
COLOR_OTHER = (250, 190, 40)      # recovered by the merge
COLOR_REFINED = (60, 220, 220)    # touched by SAM 3 or by hand

CATEGORY_SHORT = {"airplane": "A", "ship": "S", "train": "T", "car": "C"}


def _read_frame(seq_id: str, frame_id: int) -> np.ndarray | None:
    seq = sequence_by_id(seq_id)
    path = frame_path(seq, frame_id)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else img[..., ::-1].copy()


def _draw_box(img: np.ndarray, box: np.ndarray, color, thickness: int = 2,
              label: str | None = None, ring_below: float = 0.0) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if ring_below and max(x2 - x1, y2 - y1) < ring_below:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), int(ring_below), color, 1)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_view(seq_id: str, frame_id: int, draft, mask: np.ndarray | None = None,
                  max_side: int = 1100, zoom_box: list[float] | None = None,
                  decisions=None) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Clickable canvas for interactive annotation.

    Returns ``(image, scale, origin)``. A click at ``(cx, cy)`` on the returned
    image maps back to full-resolution pixels as
    ``(cx / scale + origin[0], cy / scale + origin[1])`` — the UI needs this
    because a 5 px object is unclickable at full-sequence scale, so the canvas
    can be zoomed to a region.
    """
    img = _read_frame(seq_id, frame_id)
    if img is None:
        return _placeholder(f"frame {frame_id} not readable"), 1.0, (0, 0)

    ox, oy = 0, 0
    if zoom_box is not None:
        h, w = img.shape[:2]
        x1 = int(max(0, min(zoom_box[0], w - 2)))
        y1 = int(max(0, min(zoom_box[1], h - 2)))
        x2 = int(max(x1 + 1, min(zoom_box[2], w)))
        y2 = int(max(y1 + 1, min(zoom_box[3], h)))
        img = img[y1:y2, x1:x2].copy()
        ox, oy = x1, y1
    else:
        img = img.copy()

    # Existing ground truth stays visible so a new object is not drafted on top
    # of one that is already annotated — the *completed* ground truth, since on
    # a merged sequence the raw one is missing exactly the static objects a
    # reviewer would otherwise annotate a second time.
    from .gtsource import frame_objects

    for o in frame_objects(seq_id, frame_id, decisions):
        _draw_box(img, o.box - np.array([ox, oy, ox, oy]), COLOR_MOT_GT, 1)

    if mask is not None:
        sub = mask[oy:oy + img.shape[0], ox:ox + img.shape[1]]
        if sub.shape[:2] == img.shape[:2] and sub.any():
            tint = img.astype(np.float32)
            tint[sub] = tint[sub] * 0.55 + np.array(COLOR_REFINED, np.float32) * 0.45
            img = tint.astype(np.uint8)
            edges = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_GRADIENT,
                                     np.ones((3, 3), np.uint8))
            img[edges > 0] = COLOR_REFINED

    if draft is not None:
        if draft.seed_box is not None:
            _draw_box(img, np.asarray(draft.seed_box, float) - np.array([ox, oy, ox, oy]),
                      COLOR_TARGET, 2, label=f"{draft.category}")
        if draft.box is not None:
            _draw_box(img, np.asarray(draft.box, float) - np.array([ox, oy, ox, oy]),
                      COLOR_OTHER, 1)
        for (px, py), lab in zip(draft.points, draft.labels):
            c = (60, 230, 90) if lab == 1 else (240, 60, 60)
            cv2.drawMarker(img, (int(px - ox), int(py - oy)), c,
                           cv2.MARKER_CROSS if lab == 1 else cv2.MARKER_TILTED_CROSS,
                           14, 2)
        for (px, py) in draft.box_corners:
            cv2.drawMarker(img, (int(px - ox), int(py - oy)), COLOR_OTHER,
                           cv2.MARKER_SQUARE, 12, 2)

    h, w = img.shape[:2]
    # Zoomed in, upscaling is the point: a 10 px object has to become clickable,
    # and a 1-in-8 mis-click puts a positive prompt on the background.
    scale = (max_side / max(h, w)) if zoom_box is not None \
        else min(1.0, max_side / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                         interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    return img, scale, (ox, oy)



def _placeholder(text: str, w: int = 640, h: int = 200) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _tile_placeholder(tile: int, text: str) -> np.ndarray:
    img = np.full((tile, tile, 3), 30, np.uint8)
    cv2.putText(img, text, (6, tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 180, 180), 1, cv2.LINE_AA)
    return img
