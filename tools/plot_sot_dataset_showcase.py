"""SOT dataset showcase: 2x3 grid of representative scenes for the paper.

Rows = attribute groups (SOB, Occlusion).
Cols = object classes (Car, Train, Plane/Ship).
Green box = ground-truth (axis-aligned for SatSOT/SV248S, oriented for OOTB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_neurips_style  # noqa: E402

DATA_ROOT = Path("/data/ESA_DLSTEM_2025/data/trafic")
OUT_DIR = Path(__file__).resolve().parents[1] / "Formatting Instructions For NeurIPS 2026" / "plots"

GT_GREEN = "#21d04a"
LINE_W = 1.0
DISPLAY_PX = 400  # final per-cell square size in pixels


def _satsot_gt(seq: str, fid_zero: int) -> np.ndarray:
    line = (DATA_ROOT / "SatSOT" / seq / "groundtruth.txt").read_text().splitlines()[fid_zero]
    return np.fromstring(line, sep=",", dtype=np.float32)  # x,y,w,h


def _satsot_img(seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "SatSOT" / seq / "img" / f"{fid_zero + 1:04d}.jpg"
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _sv248s_gt(vid: str, seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "SV248S" / vid / "annotations" / f"{seq}.rect"
    line = p.read_text().splitlines()[fid_zero]
    return np.fromstring(line, sep=",", dtype=np.float32)


def _sv248s_img(vid: str, seq: str, fid_zero: int) -> np.ndarray:
    seq_dir = DATA_ROOT / "SV248S" / vid / "sequences" / seq
    frames = sorted(seq_dir.glob("*.tiff"))
    return cv2.cvtColor(cv2.imread(str(frames[fid_zero])), cv2.COLOR_BGR2RGB)


def _ootb_gt(seq: str, fid_zero: int) -> np.ndarray:
    line = (DATA_ROOT / "OOTB" / seq / "groundtruth.txt").read_text().splitlines()[fid_zero]
    return np.array(line.split(), dtype=np.float32)  # x1,y1,...,x4,y4


def _ootb_img(seq: str, fid_zero: int) -> np.ndarray:
    p = DATA_ROOT / "OOTB" / seq / "img" / f"{fid_zero + 1:04d}.jpg"
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _make_scene(loader, gt_loader, *args, kind: str):
    img = loader(*args)
    gt = gt_loader(*args)
    if kind == "aabb":
        cx = gt[0] + gt[2] / 2
        cy = gt[1] + gt[3] / 2
        gw, gh = float(gt[2]), float(gt[3])
    else:  # obb 8 points
        xs, ys = gt[0::2], gt[1::2]
        cx, cy = float(xs.mean()), float(ys.mean())
        gw = float(xs.max() - xs.min())
        gh = float(ys.max() - ys.min())
    return img, gt, (cx, cy, gw, gh), kind


SCENES = [
    # (row_label, col_label, scene)
    ("SOB", "Car",   _make_scene(_satsot_img, _satsot_gt, "car_48", 125, kind="aabb")),
    ("SOB", "Train", _make_scene(_satsot_img, _satsot_gt, "train_05", 26,  kind="aabb")),
    ("SOB", "Plane", _make_scene(_sv248s_img, _sv248s_gt, "04", "000000", 714, kind="aabb")),
    ("Occ", "Car",   _make_scene(_sv248s_img, _sv248s_gt, "05", "000027", 299, kind="aabb")),
    ("Occ", "Train", _make_scene(_satsot_img, _satsot_gt, "train_02", 13,  kind="aabb")),
    ("Occ", "Ship",  _make_scene(_ootb_img,   _ootb_gt,   "ship_14", 194, kind="obb")),
]


def _crop_square(img: np.ndarray, cx: float, cy: float, gw: float, gh: float):
    H, W = img.shape[:2]
    target = max(256, int(round(max(gw, gh) * 4.0)))
    target = min(target, min(H, W))            # never larger than image
    target = max(target, 220)
    target = min(target, min(H, W))
    # Place the square centered on (cx, cy) but shift it inside the image
    # bounds so we never need white padding.
    half = target / 2
    x0 = max(0, min(cx - half, W - target))
    y0 = max(0, min(cy - half, H - target))
    x0 = int(round(x0)); y0 = int(round(y0))
    sub = img[y0:y0 + target, x0:x0 + target]
    # Resize crop to a fixed square so every panel is pixel-identical in size.
    scale = DISPLAY_PX / sub.shape[0]
    sub = cv2.resize(sub, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA)
    return sub, x0, y0, scale


def main():
    apply_neurips_style(base_size=10.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 5.2))
    grid = ImageGrid(
        fig, [0.05, 0.01, 0.94, 0.94],  # [left, bottom, width, height] in figure fraction
        nrows_ncols=(2, 3),
        axes_pad=(0.06, 0.22),  # (horizontal, vertical) in inches
        share_all=True,
    )

    axes = list(grid)
    for ax, (row, col, (img, gt, (cx, cy, gw, gh), kind)) in zip(axes, SCENES):
        crop, x0, y0, s = _crop_square(img, cx, cy, gw, gh)
        ax.imshow(crop)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if kind == "aabb":
            x, y, w, h = gt
            ax.add_patch(Rectangle(((x - x0) * s, (y - y0) * s), w * s, h * s,
                                   fill=False, edgecolor=GT_GREEN, linewidth=LINE_W))
        else:
            pts = np.stack([(gt[0::2] - x0) * s, (gt[1::2] - y0) * s], axis=1)
            ax.add_patch(Polygon(pts, closed=True, fill=False,
                                 edgecolor=GT_GREEN, linewidth=LINE_W))

        ax.set_title(col, fontsize=10, pad=3)

    # Row labels on the left of column 0 (axes are flat: 0,1,2 = row0; 3,4,5 = row1)
    for r, label in enumerate(["Similar Object", "Occlusion"]):
        axes[r * 3].set_ylabel(label, fontsize=11, labelpad=6)
        axes[r * 3].yaxis.set_label_coords(-0.04, 0.5)

    out_pdf = OUT_DIR / "sot_dataset_showcase.pdf"
    out_png = OUT_DIR / "sot_dataset_showcase.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved: {out_pdf}\nsaved: {out_png}")


if __name__ == "__main__":
    main()
