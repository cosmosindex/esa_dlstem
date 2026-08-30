"""Space-Tracker dataset showcase: 2x4 teaser figure for the paper.

Columns 1-3  Space-Tracker-SOT.  Rows are attribute groups (similar object,
             occlusion), the panel title is the target class, and the green box
             is the single annotated target.
Column 4     Space-Tracker-MOT.  Every target in the scene is annotated and the
             box colour encodes the class (car / ship / airplane).

Paths default to anonymised placeholders and are overridden at run time by
SPACE_TRACKER_DATA_ROOT (source SOT datasets) and SPACE_TRACKER_RELEASE
(the released benchmark, used for the MOT panels).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_neurips_style  # noqa: E402

DATA_ROOT = Path(os.environ.get("SPACE_TRACKER_DATA_ROOT", "/data/anon/data/trafic"))
RELEASE = Path(os.environ.get("SPACE_TRACKER_RELEASE", "/data/anon/release/space_tracker"))
OUT_DIR = Path(__file__).resolve().parents[1] / "wacv-2027-author-kit-template" / "plots"

GT_GREEN = "#21d04a"
CLASS_COLOR = {1: "#21d04a", 2: "#ff9a1f", 3: "#2ec4ff", 4: "#ff5cf0"}
LINE_W = 1.0
MOT_LINE_W = 0.6
DISPLAY_PX = 420  # per-cell square size in pixels


# --------------------------------------------------------------------- SOT ---
def _satsot_gt(seq: str, fid_zero: int) -> np.ndarray:
    line = (DATA_ROOT / "SatSOT" / seq / "groundtruth.txt").read_text().splitlines()[fid_zero]
    return np.fromstring(line, sep=",", dtype=np.float32)  # x,y,w,h


def _satsot_img(seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "SatSOT" / seq / "img" / f"{fid_zero + 1:04d}.jpg"
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _sv248s_gt(vid: str, seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "SV248S" / vid / "annotations" / f"{seq}.rect"
    return np.fromstring(p.read_text().splitlines()[fid_zero], sep=",", dtype=np.float32)


def _sv248s_img(vid: str, seq: str, fid_zero: int) -> np.ndarray:
    frames = sorted((DATA_ROOT / "SV248S" / vid / "sequences" / seq).glob("*.tiff"))
    return cv2.cvtColor(cv2.imread(str(frames[fid_zero])), cv2.COLOR_BGR2RGB)


def _ootb_gt(seq: str, fid_zero: int) -> np.ndarray:
    line = (DATA_ROOT / "OOTB" / seq / "groundtruth.txt").read_text().splitlines()[fid_zero]
    return np.array(line.split(), dtype=np.float32)  # x1,y1,...,x4,y4


def _ootb_img(seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "OOTB" / seq / "img" / f"{fid_zero + 1:04d}.jpg"
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _sot_scene(loader, gt_loader, *args, kind: str):
    img = loader(*args)
    gt = gt_loader(*args)
    if kind == "aabb":
        cx, cy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
        gw, gh = float(gt[2]), float(gt[3])
    else:  # obb, 8 points
        xs, ys = gt[0::2], gt[1::2]
        cx, cy = float(xs.mean()), float(ys.mean())
        gw, gh = float(xs.max() - xs.min()), float(ys.max() - ys.min())
    return dict(task="sot", img=img, gt=gt, centre=(cx, cy, gw, gh), kind=kind)


# --------------------------------------------------------------------- MOT ---
def _mot_scene(category: str, seq: str, frame: int, x0: int, y0: int, size: int):
    """One released MOT frame, cropped to `size` px square at (x0, y0)."""
    seq_dir = RELEASE / "mot" / category / seq
    frames = sorted((seq_dir / "img1").iterdir())
    img = cv2.cvtColor(cv2.imread(str(frames[frame - 1])), cv2.COLOR_BGR2RGB)
    boxes = []
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        f = line.split(",")
        if len(f) < 8 or int(f[0]) != frame:
            continue
        x, y, w, h = (float(v) for v in f[2:6])
        boxes.append((x, y, w, h, int(float(f[7]))))
    return dict(task="mot", img=img, boxes=boxes, crop=(x0, y0, size))


# ------------------------------------------------------------------ layout ---
# Every sequence below is in the released benchmark: the 32\u2009px filter removed
# most large targets, so e.g. satsot/train_05 and train_02 (used in an earlier
# draft) are gone and satsot/train_08 is the only released train sequence.
# Panels are titled "class (attribute)" and together cover all five SOT classes.
SOT_SCENES = [
    ("Car (SOB)",       _sot_scene(_satsot_img, _satsot_gt, "car_48", 125, kind="aabb")),
    ("Train (BC)",      _sot_scene(_satsot_img, _satsot_gt, "train_08", 60, kind="aabb")),
    ("Plane (SOB)",     _sot_scene(_sv248s_img, _sv248s_gt, "04", "000000", 714, kind="aabb")),
    ("Car (OCC)",       _sot_scene(_sv248s_img, _sv248s_gt, "05", "000027", 299, kind="aabb")),
    ("Car-large (OCC)", _sot_scene(_sv248s_img, _sv248s_gt, "05", "000006", 100, kind="aabb")),
    ("Ship (OCC)",      _sot_scene(_ootb_img,   _ootb_gt,   "ship_14", 194, kind="obb")),
]

# Both MOT panels are urban waterfronts carrying cars and ships at once; the
# panel title is the location rather than a class (see the figure caption).
MOT_SCENES = [
    ("London, The City",   _mot_scene("mixed", "mixed_sdmcar_0030", 95, 840, 0, 1080)),
    ("London, South Bank", _mot_scene("mixed", "mixed_sdmcar_0054", 104, 560, 0, 1080)),
]


def _crop_sot(img: np.ndarray, cx: float, cy: float, gw: float, gh: float):
    H, W = img.shape[:2]
    target = max(256, int(round(max(gw, gh) * 4.0)))
    target = max(min(target, min(H, W)), 220)
    target = min(target, min(H, W))
    half = target / 2
    x0 = int(round(max(0, min(cx - half, W - target))))
    y0 = int(round(max(0, min(cy - half, H - target))))
    sub = img[y0:y0 + target, x0:x0 + target]
    scale = DISPLAY_PX / sub.shape[0]
    return cv2.resize(sub, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA), x0, y0, scale


def _draw_sot(ax, scene):
    crop, x0, y0, s = _crop_sot(scene["img"], *scene["centre"])
    ax.imshow(crop)
    gt = scene["gt"]
    if scene["kind"] == "aabb":
        x, y, w, h = gt
        ax.add_patch(Rectangle(((x - x0) * s, (y - y0) * s), w * s, h * s,
                               fill=False, edgecolor=GT_GREEN, linewidth=LINE_W))
    else:
        pts = np.stack([(gt[0::2] - x0) * s, (gt[1::2] - y0) * s], axis=1)
        ax.add_patch(Polygon(pts, closed=True, fill=False,
                             edgecolor=GT_GREEN, linewidth=LINE_W))


def _draw_mot(ax, scene):
    x0, y0, size = scene["crop"]
    H, W = scene["img"].shape[:2]
    x0 = max(0, min(x0, W - size))
    y0 = max(0, min(y0, H - size))
    sub = scene["img"][y0:y0 + size, x0:x0 + size]
    s = DISPLAY_PX / size
    ax.imshow(cv2.resize(sub, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA))
    n = 0
    for x, y, w, h, cls in scene["boxes"]:
        if x + w < x0 or x > x0 + size or y + h < y0 or y > y0 + size:
            continue
        ax.add_patch(Rectangle(((x - x0) * s, (y - y0) * s), w * s, h * s,
                               fill=False, edgecolor=CLASS_COLOR.get(cls, "#ffffff"),
                               linewidth=MOT_LINE_W))
        n += 1
    return n


def main():
    apply_neurips_style(base_size=10.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Explicit axes placement: three tight SOT columns, a wider gap, then MOT.
    fig_w = 7.0
    left, right = 0.008, 0.008
    gap_s, gap_big = 0.008, 0.034
    panel_w = (1.0 - left - right - 2 * gap_s - gap_big) / 4.0

    bottom, row_gap, top = 0.014, 0.060, 0.130
    panel_h = (1.0 - bottom - row_gap - top) / 2.0
    fig_h = panel_w * fig_w / panel_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    xs = [left,
          left + panel_w + gap_s,
          left + 2 * (panel_w + gap_s),
          left + 3 * panel_w + 2 * gap_s + gap_big]
    ys = [bottom + panel_h + row_gap, bottom]  # row 0 on top

    counts = []
    for r in range(2):
        for c in range(4):
            ax = fig.add_axes([xs[c], ys[r], panel_w, panel_h])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c < 3:
                title, scene = SOT_SCENES[r * 3 + c]
                _draw_sot(ax, scene)
            else:
                title, scene = MOT_SCENES[r]
                counts.append(_draw_mot(ax, scene))
            ax.set_title(title, fontsize=9, pad=2.5)

    sot_mid = left + (3 * panel_w + 2 * gap_s) / 2.0
    mot_mid = xs[3] + panel_w / 2.0
    top_of_panels = bottom + 2 * panel_h + row_gap
    rule_y = top_of_panels + 0.062
    text_y = rule_y + 0.012
    for xc, txt in ((sot_mid, "Space-Tracker-SOT"), (mot_mid, "Space-Tracker-MOT")):
        fig.text(xc, text_y, txt, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold")
    for xc, half in ((sot_mid, (3 * panel_w + 2 * gap_s) / 2.0), (mot_mid, panel_w / 2.0)):
        fig.add_artist(plt.Line2D([xc - half, xc + half], [rule_y] * 2,
                                  color="0.35", linewidth=0.7,
                                  transform=fig.transFigure))

    out_pdf = OUT_DIR / "dataset_showcase.pdf"
    fig.savefig(out_pdf)
    fig.savefig(OUT_DIR / "dataset_showcase.png", dpi=220)
    plt.close(fig)
    print(f"saved: {out_pdf}  (MOT boxes drawn: {counts})")


if __name__ == "__main__":
    main()
