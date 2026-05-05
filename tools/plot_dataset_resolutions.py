"""Scan SOT + MOT benchmark datasets and plot per-sequence (W, H) on log-log axes.

Output: a 1x2 figure (Left: SOT, Right: MOT) with shared axes and iso-megapixel
reference curves. One marker per sequence; color = dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plot_style import apply_neurips_style

apply_neurips_style()

DATA_ROOT = Path("/data/ESA_DLSTEM_2025/data/trafic")
OUT_DIR = Path("/home/ziwen/code/esa_dlstem/docs/figures")
CACHE_PATH = OUT_DIR / "dataset_resolutions_cache.json"


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _is_real_file(p: Path) -> bool:
    return p.is_file() and not p.name.startswith(".") and not p.name.startswith("._")


def _is_real_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.startswith(".")


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if _is_real_file(p) and p.suffix.lower() in _IMG_EXTS)


def img_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def video_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def scan_seq_dir_with_img_subdir(root: Path, img_subdir: str = "img") -> list[tuple[int, int]]:
    out = []
    for seq in sorted(p for p in root.iterdir() if _is_real_dir(p)):
        frames = _list_images(seq / img_subdir)
        if frames:
            out.append(img_size(frames[0]))
    return out


def scan_ootb() -> list[tuple[int, int]]:
    root = DATA_ROOT / "OOTB"
    out = []
    for seq in sorted(p for p in root.iterdir() if _is_real_dir(p) and p.name != "anno"):
        frames = _list_images(seq / "img")
        if frames:
            out.append(img_size(frames[0]))
    return out


def scan_satsot() -> list[tuple[int, int]]:
    root = DATA_ROOT / "SatSOT"
    out = []
    for seq in sorted(p for p in root.iterdir() if _is_real_dir(p)):
        frames = _list_images(seq / "img")
        if not frames:
            frames = _list_images(seq)
        if frames:
            out.append(img_size(frames[0]))
    return out


def scan_sv248s() -> list[tuple[int, int]]:
    """SV248S: 6 base videos, each with many target sub-sequences (crops). Walk leaves."""
    root = DATA_ROOT / "SV248S"
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden dirs in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        imgs = [f for f in filenames
                if f.lower().endswith((".tif", ".tiff", ".jpg", ".png"))
                and not f.startswith(".") and not f.startswith("._")]
        if imgs and not dirnames:
            out.append(img_size(Path(dirpath) / sorted(imgs)[0]))
    return out


def scan_air_mot() -> list[tuple[int, int]]:
    return scan_seq_dir_with_img_subdir(DATA_ROOT / "AIR-MOT-100", "img")


def scan_rscardata() -> list[tuple[int, int]]:
    out = []
    for split in ("train", "test1024"):
        split_dir = DATA_ROOT / "RsCarData" / "images" / split
        if not split_dir.is_dir():
            continue
        for seq in sorted(p for p in split_dir.iterdir() if _is_real_dir(p)):
            frames = _list_images(seq / "img1")
            if frames:
                out.append(img_size(frames[0]))
    return out


def scan_satmtb_mot() -> list[tuple[int, int]]:
    root = DATA_ROOT / "SAT-MTB" / "SAT-MTB_Dataset"
    out = []
    if not root.is_dir():
        return out
    for cls_dir in sorted(p for p in root.iterdir()
                          if _is_real_dir(p) and p.name in {"airplane", "car", "ship", "train"}):
        for seq in sorted(p for p in cls_dir.iterdir() if _is_real_dir(p)):
            frames = _list_images(seq / "img")
            if frames:
                out.append(img_size(frames[0]))
    return out


def scan_sdm_car() -> list[tuple[int, int]]:
    out = []
    for split in ("train", "test", "validation"):
        split_dir = DATA_ROOT / "SDM-Car" / split
        if not split_dir.is_dir():
            continue
        for vid in sorted(split_dir.glob("*.avi")):
            if vid.name.startswith("."):
                continue
            try:
                out.append(video_size(vid))
            except Exception:
                pass
    return out


def scan_viso_mot() -> list[tuple[int, int]]:
    root = DATA_ROOT / "VISO" / "mot"
    out = []
    if not root.is_dir():
        return out
    for cls_dir in sorted(p for p in root.iterdir() if _is_real_dir(p)):
        for seq in sorted(p for p in cls_dir.iterdir() if _is_real_dir(p)):
            frames = _list_images(seq / "img")
            if frames:
                out.append(img_size(frames[0]))
    return out


SOT_SCANNERS = {
    "OOTB": scan_ootb,
    "SatSOT": scan_satsot,
    "SV248S": scan_sv248s,
}
MOT_SCANNERS = {
    "AIR-MOT-100": scan_air_mot,
    "RsCarData": scan_rscardata,
    "SAT-MTB": scan_satmtb_mot,
    "SDM-Car": scan_sdm_car,
    "VISO-MOT": scan_viso_mot,
}


def gather(use_cache: bool = True) -> dict[str, list[tuple[int, int]]]:
    if use_cache and CACHE_PATH.exists():
        print(f"[cache] loading {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            return {k: [tuple(p) for p in v] for k, v in json.load(f).items()}
    data: dict[str, list[tuple[int, int]]] = {}
    for name, fn in {**SOT_SCANNERS, **MOT_SCANNERS}.items():
        print(f"[scan] {name} ...", flush=True)
        try:
            sizes = fn()
        except Exception as e:
            print(f"  ! {name} failed: {e}")
            sizes = []
        print(f"       {len(sizes)} sequences")
        data[name] = sizes
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)
    print(f"[cache] wrote {CACHE_PATH}")
    return data


def plot(data: dict[str, list[tuple[int, int]]], out_path: Path) -> None:
    sot_palette = {
        "OOTB":   ("#1f77b4", "o"),
        "SatSOT": ("#d62728", "^"),
        "SV248S": ("#2ca02c", "D"),
    }
    mot_palette = {
        "AIR-MOT-100": ("#9467bd", "o"),
        "RsCarData":   ("#8c564b", "s"),
        "SAT-MTB":     ("#e377c2", "^"),
        "SDM-Car":     ("#17becf", "D"),
        "VISO-MOT":    ("#bcbd22", "v"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True, sharey=True)
    ax_sot, ax_mot = axes

    # collect global range for iso-curves
    all_w = [w for d in data.values() for (w, _) in d]
    all_h = [h for d in data.values() for (_, h) in d]
    if not all_w:
        raise RuntimeError("no data scanned")
    w_min, w_max = min(all_w), max(all_w)
    h_min, h_max = min(all_h), max(all_h)
    pad = 1.15
    xlim = (w_min / pad, w_max * pad)
    ylim = (h_min / pad, h_max * pad)

    # iso pixel curves: H = K / W
    iso_levels = [
        (0.05e6, "0.05 MP"),
        (0.5e6,  "0.5 MP"),
        (2e6,    "2 MP"),
        (8e6,    "8 MP"),
    ]

    def draw_panel(ax, palette, title):
        # iso-curves first (background)
        ws = np.geomspace(xlim[0], xlim[1], 200)
        for K, lbl in iso_levels:
            hs = K / ws
            mask = (hs >= ylim[0]) & (hs <= ylim[1])
            ax.plot(ws[mask], hs[mask], color="0.75", linewidth=0.6, linestyle="--", zorder=0)
            # label near top-left where curve enters the panel
            idx = np.where(mask)[0]
            if len(idx):
                i = idx[0]
                ax.text(ws[i] * 1.05, hs[i] * 0.92, lbl,
                        color="0.45", fontsize=6, ha="left", va="top", zorder=0)
        for name, (color, marker) in palette.items():
            pts = data.get(name, [])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=14, c=color, marker=marker, alpha=0.6,
                       edgecolors="white", linewidths=0.3,
                       label=f"{name} (n={len(pts)})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # NB: axis labels are placed inline via fig.text() below to save space.
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.4)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85,
                  handletextpad=0.3, borderpad=0.3)

    draw_panel(ax_sot, sot_palette, "SOT benchmarks")
    draw_panel(ax_mot, mot_palette, "MOT benchmarks")

    # Reserve explicit margins for inline axis labels (skip tight_layout).
    # wspace tightened so the two panels sit closer together; just enough gutter
    # to hold the shared "Width (px)" label.
    fig.subplots_adjust(left=0.060, right=0.995, bottom=0.16, top=0.90, wspace=0.12)

    # --- Inline axis labels (space-saving) -----------------------------------
    # 1) "Width (px)" sits in the gutter between the two panels, on the same row
    #    as the bottom x-tick labels. One shared label replaces two per-panel ones.
    # 2) "Height (px)" is the rotated y-axis label, column-aligned with the y-tick
    #    labels (same vertical line as "10^3") rather than offset further left.
    left_bbox = ax_sot.get_position()
    right_bbox = ax_mot.get_position()

    fig.text(
        (left_bbox.x1 + right_bbox.x0) / 2,
        left_bbox.y0 - 0.055,
        "Width (px)",
        ha="center", va="center",
    )

    ax_sot.set_ylabel("Height (px)")
    # x in axes-fraction: negative shifts the label LEFT of the y-axis. We pick a
    # value that puts the rotated text on the same vertical column as the
    # "10^3" / "10^4" tick labels (which sit ~0.03 axes-frac left of the axis).
    ax_sot.yaxis.set_label_coords(-0.025, 0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] wrote {out_path} (and .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "dataset_resolutions.png"))
    args = ap.parse_args()
    data = gather(use_cache=not args.no_cache)
    for k, v in data.items():
        print(f"  {k:14s} {len(v):4d} seqs")
    plot(data, Path(args.out))


if __name__ == "__main__":
    main()
