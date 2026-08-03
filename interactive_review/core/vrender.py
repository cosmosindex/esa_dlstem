"""The three views a video-level review needs, and why there are three.

``frame_view``
    The whole frame with every box drawn. This is navigation, not inspection:
    at 1600x1200 downscaled to fit a browser, a 10 px aircraft is 6 px and its
    box is a dot. It answers "where is everything and is anything obviously
    missing", never "is this box tight".

``grid_view``
    One tile per object on the current frame, each cropped to its own box plus
    context and upscaled to a constant tile size. **This is the inspection
    surface.** A frame's objects differ in size by 20x, so only a per-object
    zoom puts them all at a judgeable scale at once; scanning a wall of tiles
    finds the wrong box in one pass instead of 40 clicks.

``fix_view``
    One object, 5-10x, for correcting its box by hand. The reviewer is here
    precisely because the error is a few pixels wide.

Colour is provenance, not category: after the merge writes one file, nothing
else distinguishes a box that shipped with the dataset from one recovered out
of detection XML, and the newest annotation is where a reviewer's attention is
worth most.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gtsource import (DRAWN, FILLED, RECOVERED, REVIEWED, Obj, frame_objects,
                       raw_geometry)
from .paths import sequence_by_id
from .render import (CATEGORY_SHORT, _draw_box, _placeholder, _read_frame,
                     _tile_placeholder)

COLOR_BY_PROVENANCE = {
    "original": (40, 200, 90),     # green  — shipped with the dataset
    RECOVERED:  (250, 190, 40),    # amber  — restored from detection XML
    FILLED:     (250, 130, 40),    # orange — interpolated, then SAM 3
    REVIEWED:   (60, 220, 220),    # cyan   — corrected by hand
    DRAWN:      (200, 90, 240),    # violet — annotated from nothing
}
COLOR_SELECTED = (240, 40, 40)
#: The pre-SAM 3 box, drawn alongside on request so the two standards can be
#: compared where the difference actually matters — on the imagery.
COLOR_RAW = (235, 235, 235)

LEGEND = ("🟩 dataset &nbsp;·&nbsp; 🟧 recovered from detection XML "
          "&nbsp;·&nbsp; 🟠 hole-filled &nbsp;·&nbsp; 🟦 corrected by hand "
          "&nbsp;·&nbsp; 🟪 drawn by hand &nbsp;·&nbsp; 🟥 selected "
          "&nbsp;·&nbsp; ⬜ original geometry, before SAM 3")


def label_of(o: Obj) -> str:
    return f"{CATEGORY_SHORT.get(o.category, '?')}{o.track_id}"


def frame_view(seq_id: str, frame_id: int, decisions=None,
               selected: str | None = None, max_side: int = 1100,
               ring_below: float = 14.0, labels: bool = True,
               show_raw: bool = False) -> tuple[np.ndarray, float]:
    """Whole frame, every object drawn, coloured by provenance.

    Returns ``(image, scale)`` so a click can be mapped back to full-resolution
    pixels as ``(cx / scale, cy / scale)``.
    """
    img = _read_frame(seq_id, frame_id)
    if img is None:
        return _placeholder(f"frame {frame_id} not readable"), 1.0

    deleted = decisions.deleted(seq_id) if decisions else set()
    raw = raw_geometry(seq_id) if show_raw else {}
    for o in frame_objects(seq_id, frame_id, decisions):
        if o.key in deleted:
            continue
        before = raw.get(o.key, {}).get(frame_id)
        if before is not None:
            _draw_box(img, np.asarray(before, float), COLOR_RAW, 1)
        chosen = o.key == selected
        _draw_box(img, o.box,
                  COLOR_SELECTED if chosen else COLOR_BY_PROVENANCE.get(
                      o.provenance, (200, 200, 200)),
                  2 if chosen else 1,
                  label=label_of(o) if labels else None,
                  ring_below=ring_below * (1.5 if chosen else 1.0))

    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def grid_objects(seq_id: str, frame_id: int, decisions=None,
                 categories: tuple[str, ...] | None = None,
                 provenances: tuple[str, ...] | None = None) -> list[Obj]:
    """Objects to tile, ordered so the same object keeps its place across frames.

    Stable ordering matters more than it sounds: the reviewer steps through
    frames watching one tile, and sorting by anything frame-dependent (size,
    position) would shuffle the wall under them.
    """
    deleted = decisions.deleted(seq_id) if decisions else set()
    objs = [o for o in frame_objects(seq_id, frame_id, decisions)
            if o.key not in deleted
            and (categories is None or o.category in categories)
            and (provenances is None or o.provenance in provenances)]
    return sorted(objs, key=lambda o: (o.category, o.track_id))


def grid_view(seq_id: str, frame_id: int, decisions=None, tile: int = 150,
              context: float = 2.5, columns: int = 8,
              selected: str | None = None,
              categories: tuple[str, ...] | None = None,
              provenances: tuple[str, ...] | None = None,
              max_tiles: int = 96,
              show_raw: bool = False) -> tuple[np.ndarray, list[Obj]]:
    """A wall of per-object crops for one frame, all at the same tile size.

    Returns ``(image, objects)``; ``objects`` is in tile order, so a click at
    tile *i* identifies ``objects[i]`` — the UI needs that to open the fix view.

    ``max_tiles`` bounds a frame with hundreds of cars. It is reported in the UI
    rather than applied silently: a wall that quietly showed 96 of 300 objects
    would read as "all checked".
    """
    objs = grid_objects(seq_id, frame_id, decisions, categories, provenances)[:max_tiles]
    if not objs:
        return _placeholder("no objects on this frame"), []

    img = _read_frame(seq_id, frame_id)
    if img is None:
        return _placeholder(f"frame {frame_id} not readable"), []
    H, W = img.shape[:2]
    raw = raw_geometry(seq_id) if show_raw else {}

    tiles = []
    for o in objs:
        x1, y1, x2, y2 = (float(v) for v in o.box)
        # A minimum half-window stops a 5 px object from becoming a wall of
        # interpolation mush, and keeps every tile at a comparable zoom.
        half = max((x2 - x1) * (1 + context) / 2, (y2 - y1) * (1 + context) / 2, 18.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = int(round(cx - half)), int(round(cy - half))
        crop = img[max(oy, 0):min(int(round(cy + half)), H),
                   max(ox, 0):min(int(round(cx + half)), W)]
        if crop.size == 0:
            tiles.append(_tile_placeholder(tile, label_of(o)))
            continue
        crop = crop.copy()
        # Pad rather than let an edge object render at a different zoom than
        # its neighbours — comparing tiles is the whole point of the wall.
        pad_l, pad_t = max(-ox, 0), max(-oy, 0)
        pad_r = max(int(round(cx + half)) - W, 0)
        pad_b = max(int(round(cy + half)) - H, 0)
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                      cv2.BORDER_CONSTANT, value=(30, 30, 30))
        ch, cw = crop.shape[:2]
        s = tile / max(ch, cw)
        crop = cv2.resize(crop, (max(int(cw * s), 1), max(int(ch * s), 1)),
                          interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
        crop = cv2.copyMakeBorder(crop, 0, tile - crop.shape[0], 0, tile - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))

        to_tile = lambda b: (np.asarray(b, float) - np.array([ox, oy, ox, oy])) * s
        before = raw.get(o.key, {}).get(frame_id)
        if before is not None:
            _draw_box(crop, to_tile(before), COLOR_RAW, 1)
        box_in_tile = to_tile([x1, y1, x2, y2])
        colour = (COLOR_SELECTED if o.key == selected
                  else COLOR_BY_PROVENANCE.get(o.provenance, (200, 200, 200)))
        _draw_box(crop, box_in_tile, colour, 1)
        cv2.rectangle(crop, (0, 0), (tile - 1, tile - 1), colour,
                      3 if o.key == selected else 1)
        cv2.putText(crop, f"{label_of(o)} {o.size:.0f}px", (4, tile - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        tiles.append(crop)

    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i:i + columns]
        while len(row) < columns:
            row.append(np.full((tile, tile, 3), 20, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows), objs


def tile_at(x: float, y: float, n: int, tile: int = 150, columns: int = 8) -> int | None:
    """Index of the tile a click landed on, or ``None`` past the last one."""
    col, row = int(x // tile), int(y // tile)
    if col < 0 or col >= columns or row < 0:
        return None
    idx = row * columns + col
    return idx if idx < n else None


def fix_view(seq_id: str, frame_id: int, box: list[float] | None,
             corners: list[tuple[float, float]] | None = None,
             drawn: list[float] | None = None, pad: float = 3.0,
             max_side: int = 900, raw_box: list[float] | None = None
             ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """One object, zoomed, for correcting its box by hand.

    Returns ``(image, scale, origin)``; a click at ``(cx, cy)`` maps back with
    ``(cx / scale + origin[0], cy / scale + origin[1])``.
    """
    img = _read_frame(seq_id, frame_id)
    if img is None:
        return _placeholder(f"frame {frame_id} not readable"), 1.0, (0, 0)

    H, W = img.shape[:2]
    if box is None:
        ox = oy = 0
    else:
        x1, y1, x2, y2 = (float(v) for v in box)
        half = max((x2 - x1) * (1 + pad) / 2, (y2 - y1) * (1 + pad) / 2, 40.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox = int(max(0, min(cx - half, W - 1)))
        oy = int(max(0, min(cy - half, H - 1)))
        img = img[oy:int(min(cy + half, H)), ox:int(min(cx + half, W))].copy()

    off = np.array([ox, oy, ox, oy], float)
    if raw_box is not None:
        _draw_box(img, np.asarray(raw_box, float) - off, COLOR_RAW, 1,
                  label="before SAM 3")
    if box is not None:
        _draw_box(img, np.asarray(box, float) - off, COLOR_SELECTED, 1, label="current")
    if drawn is not None:
        _draw_box(img, np.asarray(drawn, float) - off, COLOR_BY_PROVENANCE[REVIEWED],
                  1, label="replacement")
    for (px, py) in (corners or []):
        cv2.drawMarker(img, (int(px - ox), int(py - oy)), COLOR_BY_PROVENANCE[REVIEWED],
                       cv2.MARKER_SQUARE, 12, 2)

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return _placeholder("crop is empty"), 1.0, (0, 0)
    scale = max_side / max(h, w)
    img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                     interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    return img, scale, (ox, oy)


def size_outliers(seq_id: str, key: str, decisions=None, z: float = 3.0
                  ) -> list[tuple[int, float]]:
    """Frames where a track's box area departs from its own norm.

    Catches the failure this review exists to find — a box that grew or shrank
    on a handful of frames — without a human stepping through 300 of them. Uses
    a median/MAD rule, since one bad box would drag a mean-based one with it.
    """
    from .gtsource import load_frames

    fixes = (decisions.boxes(seq_id).get(key, {}) if decisions else {})
    areas: list[tuple[int, float]] = []
    for fid, objs in load_frames(seq_id).items():
        for o in objs:
            if o.key != key:
                continue
            b = fixes.get(fid, o.box)
            areas.append((fid, max((b[2] - b[0]) * (b[3] - b[1]), 0.0)))
    if len(areas) < 8:
        return []
    a = np.array([v for _, v in areas])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) or 1e-9
    return sorted((fid, v / max(med, 1e-9)) for fid, v in areas
                  if abs(v - med) / (1.4826 * mad) > z)
