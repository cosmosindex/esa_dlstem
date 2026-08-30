"""Annotation-tool figure: the surfaces a reviewer actually looks at.

Every panel is rendered by the tool's own code (``interactive_review.core``),
not redrawn for the paper, so the box colours, the tile zoom and the geometry
are exactly what the reviewer sees:

(a) whole frame      navigation and "is anything missing"
(b) object grid      one tile per object at a constant zoom -- the inspection
                     surface, where a misplaced box is obvious
(c) SAM 3 geometry   one tile with the box before and after the SAM 3 pass, and
                     one hand-drawn 5 px car from the same frame
(d) playback         a fixed window over time, where drift and flicker appear

Paths default to anonymised placeholders and are overridden at run time by
SPACE_TRACKER_DATA (source datasets) and WORK_ROOT (merged ground truth).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from interactive_review.core import vrender  # noqa: E402
from interactive_review.core.gtsource import visible_objects  # noqa: E402
from interactive_review.core.paths import DEFAULT_OVERRIDES  # noqa: E402
from interactive_review.core.render import _read_frame  # noqa: E402
from interactive_review.core.vdecisions import SequenceDecisions  # noqa: E402
from interactive_review.core.vrender import frame_view, grid_view  # noqa: E402
from plot_style import apply_neurips_style  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "wacv-2027-author-kit-template" / "plots"

#: One released SAT-MTB sequence carries all five provenances worth showing at
#: once: three aircraft the source ships, nineteen it annotates for detection
#: but omits from its MOT ground truth, and a 5 px car drawn by hand.
SEQ = "satmtb/airplane/34"
FRAME = 139
#: The moving aircraft, for the playback strip.
TRACK = "airplane:3"
STRIP_FRAMES = [1, 71, 142, 212, 283]
#: The tile shown before/after the SAM 3 geometry pass, and the drawn car.
SAM_KEY = "airplane:5"
DRAWN_KEY = "car:9007"

RGB = lambda c: tuple(v / 255.0 for v in c)  # noqa: E731
#: Generic wording: the figure describes the tool, not this benchmark's
#: curation, so the labels name what produced a box rather than which of our
#: source datasets it came out of.
PROV_LEGEND = [
    ("original", "arrived with the data"),
    (vrender.RECOVERED, "recovered by a batch pass"),
    (vrender.FILLED, "hole-filled"),
    (vrender.REVIEWED, "corrected by hand"),
    (vrender.DRAWN, "drawn by hand"),
]


def _thick(factor: int = 2):
    """Draw the tool's 1 px boxes thicker, so they survive print reduction."""
    original = vrender._draw_box

    def patched(img, box, color, thickness=2, **kw):
        return original(img, box, color, max(1, thickness) * factor, **kw)

    vrender._draw_box = patched
    return original


def _panel(ax, img, label, title=None, label_right=False):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.45"); sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8, pad=2.5)
    ax.text(0.988 if label_right else 0.012, 0.985, label, transform=ax.transAxes,
            ha="right" if label_right else "left", va="top",
            fontsize=8.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.18", fc="black", ec="none", alpha=0.62))


def _tile(seq, frame, key, dec, side=260, context=2.5, raw=False):
    """One object cropped and upscaled the way the grid does it."""
    obj = next(o for o in visible_objects(seq, frame, dec) if o.key == key)
    img = _read_frame(seq, frame)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in obj.box)
    half = max((x2 - x1) * (1 + context) / 2, (y2 - y1) * (1 + context) / 2, 18.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox, oy = int(round(cx - half)), int(round(cy - half))
    crop = img[max(oy, 0):min(int(round(cy + half)), h),
               max(ox, 0):min(int(round(cx + half)), w)].copy()
    s = side / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                      interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    to_tile = lambda b: (np.asarray(b, float) - np.array([ox, oy, ox, oy])) * s  # noqa: E731
    if raw:
        before = vrender.raw_geometry(seq).get(key, {}).get(frame)
        if before is not None:
            vrender._draw_box(crop, to_tile(before), vrender.COLOR_RAW, 1)
    vrender._draw_box(crop, to_tile(obj.box),
                      vrender.COLOR_BY_PROVENANCE[obj.provenance], 1)
    return crop


def _strip(seq, frames, key, dec, side=300, half=150, gap=6):
    """A fixed window over time: the background holds still, the target does not."""
    from interactive_review.core.gtsource import load_frames
    loaded = load_frames(seq)
    centres = []
    for f in frames:
        b = next(o for o in loaded[f] if o.key == key).box
        centres.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    cx, cy = np.asarray(centres, float).mean(0)
    ox, oy = int(cx - half), int(cy - half)

    tiles = []
    for f in frames:
        img = _read_frame(seq, f)
        crop = img[oy:oy + 2 * half, ox:ox + 2 * half].copy()
        for o in visible_objects(seq, f, dec):
            b = np.asarray(o.box, float) - np.array([ox, oy, ox, oy])
            if b[2] < 0 or b[3] < 0 or b[0] > 2 * half or b[1] > 2 * half:
                continue
            vrender._draw_box(crop, b, vrender.COLOR_BY_PROVENANCE[o.provenance], 1)
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"t = {f}", (7, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(crop)
    sep = np.full((side, gap, 3), 255, np.uint8)
    return np.hstack([x for t in tiles for x in (t, sep)][:-1])


def main():
    apply_neurips_style(base_size=9.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _thick(2)
    dec = SequenceDecisions(DEFAULT_OVERRIDES)

    grid, objs = grid_view(SEQ, FRAME, dec, tile=190, columns=6)
    frame, _ = frame_view(SEQ, FRAME, dec, max_side=1300, labels=False)
    sam_tile = _tile(SEQ, FRAME, SAM_KEY, dec, raw=True)
    car_tile = _tile(SEQ, FRAME, DRAWN_KEY, dec)
    strip = _strip(SEQ, STRIP_FRAMES, TRACK, dec)

    # Lay the panels out in inches, so no panel is letterboxed inside its axes:
    # the whole frame is square, the grid is 6x4 tiles, the right column is two
    # stacked square tiles, and the playback strip keeps its own aspect.
    W = 7.0
    margin, gap, tile_gap = 0.05, 0.07, 0.11
    title_h, row_gap, legend_h = 0.17, 0.30, 0.07

    usable = W - 2 * margin - 2 * gap
    frame_ar = frame.shape[1] / frame.shape[0]
    grid_ar = grid.shape[1] / grid.shape[0]
    h1 = (usable + tile_gap / 2) / (frame_ar + grid_ar + 0.5)
    w_frame, w_grid = frame_ar * h1, grid_ar * h1
    w_col = (h1 - tile_gap) / 2

    # The strip takes the left of the bottom band and the provenance legend the
    # right, rather than a full-width strip with a legend line under it: that
    # layout wasted 0.6 in of page height on a figure that already fills half a
    # column of float space.
    strip_ar = strip.shape[1] / strip.shape[0]
    w_strip = 0.615 * (W - 2 * margin)
    h_strip = w_strip / strip_ar

    H = title_h + h1 + row_gap + h_strip + legend_h
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]  # noqa: E731

    y_strip = legend_h
    y1 = y_strip + h_strip + row_gap

    _panel(fig.add_axes(box(margin, y1, w_frame, h1)), frame, "(a)",
           "whole frame — what is missing")
    _panel(fig.add_axes(box(margin + w_frame + gap, y1, w_grid, h1)), grid, "(b)",
           "object grid — is every box right")

    x_col = margin + w_frame + w_grid + 2 * gap
    _panel(fig.add_axes(box(x_col, y1 + w_col + tile_gap, w_col, w_col)),
           sam_tile, "(c)", "SAM 3 geometry")
    _panel(fig.add_axes(box(x_col, y1, w_col, w_col)), car_tile, "(d)",
           "drawn by hand, $5$ px")

    _panel(fig.add_axes(box(margin, y_strip, w_strip, h_strip)), strip, "(e)",
           "playback — where drift, flicker and duplicate identities appear",
           label_right=True)

    # Provenance legend, in the tool's own colours.
    entries = [(vrender.COLOR_BY_PROVENANCE[p], t) for p, t in PROV_LEGEND]
    entries.append((vrender.COLOR_RAW, "before the SAM 3 snap"))
    x_leg = margin + w_strip + 0.24
    fig.text(x_leg / W, (y_strip + h_strip - 0.02) / H, "box colour = provenance",
             ha="left", va="top", fontsize=8.0, fontweight="bold")
    step = (h_strip - 0.26) / (len(entries) - 1)
    for i, (colour, text) in enumerate(entries):
        y = y_strip + h_strip - 0.26 - i * step
        fig.add_artist(Rectangle((x_leg / W, (y - 0.035) / H), 0.115 / W, 0.075 / H,
                                 transform=fig.transFigure, fc=RGB(colour),
                                 ec="0.35", lw=0.4))
        fig.text((x_leg + 0.175) / W, y / H, text, ha="left", va="center",
                 fontsize=7.6)

    out = OUT_DIR / "annotation_tool_views.pdf"
    fig.savefig(out)
    fig.savefig(OUT_DIR / "annotation_tool_views.png", dpi=240)
    plt.close(fig)
    print(f"saved: {out}  (grid tiles: {len(objs)}, strip frames: {len(STRIP_FRAMES)})")


if __name__ == "__main__":
    main()
