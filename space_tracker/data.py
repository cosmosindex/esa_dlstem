"""Per-frame loaders that bridge the manifest to on-disk imagery + GT.

The manifest stores image / GT paths *relative* to each dataset's own root
(so the manifest is portable across machines). At runtime the user supplies
the absolute root for each dataset; this module does the resolution and
yields per-frame ``(image_path, gt_box, gt_obb)`` records.

GT formats handled:

* ``obb_8pt``           — OOTB; 8-corner OBB per line.
* ``xywh_with_none``    — SatSOT; ``x,y,w,h`` per line, ``none`` for absent target.
* ``xywh_with_state``   — SV248S; ``<seq>.rect`` (xywh) + ``<seq>.state``
                          (0=visible / 1=invisible / 2=occluded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .manifest import SequenceRecord


@dataclass
class Frame:
    """One frame of GT + the path to the image."""
    frame_id: int
    image_path: Path
    visible: bool                # False → target absent / fully invisible this frame
    gt_box_xyxy: np.ndarray | None      # axis-aligned bbox (x1, y1, x2, y2)
    gt_obb_8pt:  np.ndarray | None      # OBB corners (8,) — only for OOTB
    state: int | None                   # SV248S only: 0 / 1 / 2


def resolve(seq: SequenceRecord, dataset_root: str | Path) -> tuple[Path, Path]:
    """(image_dir_abs, gt_path_abs) for a sequence."""
    root = Path(dataset_root)
    return root / seq.image_dir, root / seq.gt_path


def _list_images(image_dir: Path, image_glob: str) -> list[Path]:
    """Return per-frame image paths sorted in capture order."""
    files = sorted(image_dir.glob(image_glob))
    return files


# ---------------------------------------------------------------- OOTB --

def _iter_ootb(seq: SequenceRecord, dataset_root: Path) -> Iterator[Frame]:
    img_dir, gt_path = resolve(seq, dataset_root)
    frames = sorted(img_dir.glob("*.jpg"))
    with open(gt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = min(len(frames), len(lines))
    for i in range(n):
        vals = re.split(r"[,\t ]+", lines[i])
        try:
            obb = np.array([float(v) for v in vals[:8]], dtype=np.float32)
        except ValueError:
            obb = None
        if obb is None or np.isnan(obb).any() or obb.size < 8:
            yield Frame(i, frames[i], visible=False,
                        gt_box_xyxy=None, gt_obb_8pt=None, state=None)
            continue
        xs, ys = obb[0::2], obb[1::2]
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        yield Frame(i, frames[i], visible=True,
                    gt_box_xyxy=box, gt_obb_8pt=obb, state=None)


# ------------------------------------------------------------- SatSOT --

def _iter_satsot(seq: SequenceRecord, dataset_root: Path) -> Iterator[Frame]:
    img_dir, gt_path = resolve(seq, dataset_root)
    frames = sorted(img_dir.iterdir())
    frames = [p for p in frames if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")]
    with open(gt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = min(len(frames), len(lines))
    for i in range(n):
        line = lines[i]
        if "none" in line.lower():
            yield Frame(i, frames[i], visible=False,
                        gt_box_xyxy=None, gt_obb_8pt=None, state=None)
            continue
        vals = re.split(r"[,\t ]+", line)
        try:
            x, y, w, h = (float(v) for v in vals[:4])
        except ValueError:
            yield Frame(i, frames[i], visible=False,
                        gt_box_xyxy=None, gt_obb_8pt=None, state=None)
            continue
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        yield Frame(i, frames[i], visible=True,
                    gt_box_xyxy=box, gt_obb_8pt=None, state=None)


# ------------------------------------------------------------- SV248S --

def _iter_sv248s(seq: SequenceRecord, dataset_root: Path) -> Iterator[Frame]:
    img_dir, rect_path = resolve(seq, dataset_root)
    state_path = rect_path.with_suffix(".state")
    frames = sorted(img_dir.glob("*.tiff"))

    rects: list[tuple[float, float, float, float]] = []
    with open(rect_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = re.split(r"[,\t ]+", line)
            try:
                rects.append(tuple(float(v) for v in vals[:4]))   # type: ignore[arg-type]
            except ValueError:
                rects.append((0.0, 0.0, 0.0, 0.0))
    states: list[int] = []
    if state_path.exists():
        with open(state_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    states.append(int(line))
                except ValueError:
                    states.append(0)
    else:
        states = [0] * len(rects)

    n = min(len(frames), len(rects), len(states))
    for i in range(n):
        st = states[i]
        if st == 1:   # invisible
            yield Frame(i, frames[i], visible=False,
                        gt_box_xyxy=None, gt_obb_8pt=None, state=st)
            continue
        x, y, w, h = rects[i]
        if w <= 0 or h <= 0:
            yield Frame(i, frames[i], visible=False,
                        gt_box_xyxy=None, gt_obb_8pt=None, state=st)
            continue
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        yield Frame(i, frames[i], visible=True,
                    gt_box_xyxy=box, gt_obb_8pt=None, state=st)


# ---------------------------------------------------------------------

def iter_frames(
    seq: SequenceRecord,
    dataset_roots: dict[str, str | Path],
) -> Iterator[Frame]:
    """Yield frames for ``seq`` using the user-provided dataset root."""
    if seq.dataset not in dataset_roots:
        raise KeyError(
            f"dataset_roots is missing an entry for '{seq.dataset}' "
            f"(needed by sequence {seq.id})"
        )
    root = Path(dataset_roots[seq.dataset])
    if seq.dataset == "ootb":
        yield from _iter_ootb(seq, root)
    elif seq.dataset == "satsot":
        yield from _iter_satsot(seq, root)
    elif seq.dataset == "sv248s":
        yield from _iter_sv248s(seq, root)
    else:
        raise ValueError(f"Unknown dataset {seq.dataset!r} for sequence {seq.id}")
