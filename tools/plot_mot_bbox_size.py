"""Plot bbox-size distribution per class across all MOT datasets.

Datasets pooled (4 classes: car, airplane, ship, train):
    - AIR-MOT-100  : MOT csv per sequence (class column 8: 1=airplane, 2=ship)
    - VISO         : per-class subdirs `mot/{plane,ship,train}/<seq>/gt/gt.txt`
                     (NB: `mot/car/` is intentionally skipped because RsCarData
                      is the re-annotation of VISO's car split)
    - RsCarData    : `labeleddata20230227/<seq>/img1/*.xml` (new-mode test GT)
                     + `annotations/train_mot.json` (COCO MOT for train)
    - SDM-Car      : `{train,val,test}/<seq>-gt.csv` (single class car)
    - SAT-MTB      : `SAT-MTB_Dataset/{car,airplane,ship,train}/<seq>/mot/*.txt`
                     (class id 0=car, 1=airplane, 2=ship, 3=train)

For each box we record (dataset, class, w, h, sqrt_area). Output:
    - docs / bbox_stats / mot_bbox_size_ecdf.{png,pdf}   (ECDF small multiples)
    - docs / bbox_stats / mot_bbox_size_summary.csv      (per dataset×class quantiles + counts)

Run:
    python tools/plot_mot_bbox_size.py
"""
from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_neurips_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/data/ESA_DLSTEM_2025/data/trafic")
OUT_DIR = Path("/home/ziwen/code/esa_dlstem/docs/bbox_stats")
CACHE = Path("/tmp/mot_bbox_size_cache.pkl")  # delete to force re-parse

CLASSES = ["car", "airplane", "ship", "train"]
DATASETS = ["AIR-MOT", "VISO", "RsCarData", "SDM-Car", "SAT-MTB"]


# ---------------------------------------------------------------------------
# Parsers — each returns a list of (class_name, w, h) tuples
# ---------------------------------------------------------------------------

def parse_airmot(root: Path) -> list[tuple[str, float, float]]:
    """AIR-MOT-100: MOT csv `frame,id,x,y,w,h,conf,class,visibility`.

    class==1 → airplane, class==2 → ship. Empty gt.txt files are skipped.
    """
    cmap = {"1": "airplane", "2": "ship"}
    out: list[tuple[str, float, float]] = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.isdigit():
            continue
        gt = seq_dir / "gt" / "gt.txt"
        if not gt.exists() or gt.stat().st_size == 0:
            continue
        with open(gt) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                w, h = float(parts[4]), float(parts[5])
                cls = cmap.get(parts[7].strip())
                if cls is None or w <= 0 or h <= 0:
                    continue
                out.append((cls, w, h))
    return out


def parse_viso(root: Path) -> list[tuple[str, float, float]]:
    """VISO: skip `car/` (re-annotated by RsCarData).

    Plane/Ship use **space-delimited xyxy**: `frame id x1 y1 x2 y2 ...`
    Train uses **comma-delimited xywh**: `frame,id,x,y,w,h,...`
    """
    out: list[tuple[str, float, float]] = []
    name_to_class = {"plane": "airplane", "ship": "ship", "train": "train"}
    mot_root = root / "mot"
    for sub_name, cls in name_to_class.items():
        sub = mot_root / sub_name
        if not sub.is_dir():
            continue
        for seq_dir in sorted(sub.iterdir()):
            gt = seq_dir / "gt" / "gt.txt"
            if not gt.exists():
                continue
            with open(gt) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "," in line:  # comma-delimited xywh (train category)
                        parts = line.split(",")
                        if len(parts) < 6:
                            continue
                        w, h = float(parts[4]), float(parts[5])
                    else:  # space-delimited xyxy (plane / ship)
                        parts = line.split()
                        if len(parts) < 6:
                            continue
                        x1, y1, x2, y2 = (float(parts[i]) for i in range(2, 6))
                        w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    out.append((cls, w, h))
    return out


def parse_rscardata(root: Path) -> list[tuple[str, float, float]]:
    """RsCarData: train via COCO MOT json, test via XML (new-mode GT)."""
    out: list[tuple[str, float, float]] = []

    # Train — COCO `bbox = [x, y, w, h]`
    train_json = root / "annotations" / "train_mot.json"
    if train_json.exists():
        with open(train_json) as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                out.append(("car", float(w), float(h)))

    # Test — VOC-style xmin/ymin/xmax/ymax XML
    label_root = root / "labeleddata20230227"
    if label_root.is_dir():
        for xml_path in label_root.rglob("*.xml"):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                continue
            for obj in tree.getroot().findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.findtext("xmin", "0"))
                ymin = float(bb.findtext("ymin", "0"))
                xmax = float(bb.findtext("xmax", "0"))
                ymax = float(bb.findtext("ymax", "0"))
                w, h = xmax - xmin, ymax - ymin
                if w > 0 and h > 0:
                    out.append(("car", w, h))
    return out


def parse_sdmcar(root: Path) -> list[tuple[str, float, float]]:
    """SDM-Car: csv `frame,id,x,y,w,h,-1,-1,-1,-1`, single class `car`."""
    out: list[tuple[str, float, float]] = []
    for sub in ("train", "validation", "test"):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        for csv_path in sorted(sub_dir.glob("*-gt.csv")):
            with open(csv_path) as f:
                reader = csv.reader(f)
                for parts in reader:
                    if len(parts) < 6:
                        continue
                    w, h = float(parts[4]), float(parts[5])
                    if w > 0 and h > 0:
                        out.append(("car", w, h))
    return out


def parse_satmtb(root: Path) -> list[tuple[str, float, float]]:
    """SAT-MTB MOT: per-class subdirs, csv `frame,id,x,y,w,h,conf,cls,...`."""
    cmap = {0: "car", 1: "airplane", 2: "ship", 3: "train"}
    out: list[tuple[str, float, float]] = []
    base = root / "SAT-MTB_Dataset"
    for cat_name in ("airplane", "car", "ship", "train"):
        cat_dir = base / cat_name
        if not cat_dir.is_dir():
            continue
        for seq_dir in sorted(cat_dir.iterdir()):
            mot_dir = seq_dir / "mot"
            if not mot_dir.is_dir():
                continue
            mot_files = [p for p in mot_dir.iterdir() if p.is_file()]
            if not mot_files:
                continue
            for mp in mot_files:
                with open(mp) as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 8:
                            continue
                        try:
                            w, h = float(parts[4]), float(parts[5])
                            cls_id = int(parts[7])
                        except ValueError:
                            continue
                        cls = cmap.get(cls_id)
                        if cls is None or w <= 0 or h <= 0:
                            continue
                        out.append((cls, w, h))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect() -> pd.DataFrame:
    if CACHE.exists():
        print(f"  using cached parse → {CACHE}  (delete to force re-parse)")
        return pd.read_pickle(CACHE)
    rows: list[dict] = []
    for ds_name, parser, ds_root in [
        ("AIR-MOT",   parse_airmot,    ROOT / "AIR-MOT-100"),
        ("VISO",      parse_viso,      ROOT / "VISO"),
        ("RsCarData", parse_rscardata, ROOT / "RsCarData"),
        ("SDM-Car",   parse_sdmcar,    ROOT / "SDM-Car"),
        ("SAT-MTB",   parse_satmtb,    ROOT / "SAT-MTB"),
    ]:
        print(f"  parsing {ds_name} …", flush=True)
        for cls, w, h in parser(ds_root):
            rows.append({
                "dataset": ds_name,
                "class": cls,
                "w": w,
                "h": h,
                "area": w * h,
                "sqrt_area": np.sqrt(w * h),
            })
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE)
    print(f"  cached parse → {CACHE}")
    return df


def write_summary(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for (ds, cls), sub in df.groupby(["dataset", "class"]):
        rows.append({
            "dataset": ds,
            "class": cls,
            "n_boxes": len(sub),
            "sqrt_area_p05": float(np.percentile(sub["sqrt_area"], 5)),
            "sqrt_area_p25": float(np.percentile(sub["sqrt_area"], 25)),
            "sqrt_area_median": float(sub["sqrt_area"].median()),
            "sqrt_area_p75": float(np.percentile(sub["sqrt_area"], 75)),
            "sqrt_area_p95": float(np.percentile(sub["sqrt_area"], 95)),
            "avg_w": float(sub["w"].mean()),
            "avg_h": float(sub["h"].mean()),
        })
    out = pd.DataFrame(rows).sort_values(["class", "dataset"])
    out.to_csv(path, index=False, float_format="%.2f")


# ---------------------------------------------------------------------------
# Plotting — width × height scatter, color = class
# ---------------------------------------------------------------------------

CLASS_PALETTE = {
    "car":      "#C44E52",  # red
    "airplane": "#4C72B0",  # blue
    "ship":     "#55A868",  # green
    "train":    "#CCB974",  # yellow
}

SMALL_THR = 8  # px — boxes with w < 8 AND h < 8 are flagged as "small objects"


def plot_scatter(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    cap_per_class: int = 3000,
    seed: int = 42,
) -> None:
    apply_neurips_style(base_size=10.0)
    rng = np.random.default_rng(seed)

    counts = df.groupby("class").size().to_dict()

    # SR/NPR plot is 1421x548 px (ratio ~2.594) with data axes occupying the
    # top ~73% and a horizontal legend in the bottom ~22%. We mirror that exact
    # external aspect AND internal axes position so the two subfigures, when
    # placed side-by-side at width=0.49\linewidth, render with their x-axis
    # baselines on the same horizontal line.
    fig = plt.figure(figsize=(7.1, 2.74))  # 2.594:1 ratio
    # Data axes occupy fig y=[0.23, 0.96] → x-axis at 23% from bottom = 77%
    # from top, matching the SR/NPR plot's data-axis baseline.
    ax = fig.add_axes([0.075, 0.23, 0.905, 0.73])

    # Plot in CLASSES order so the smallest-count class (Train) ends up on top
    # of the densest class (Car) and stays visible.
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        n_total = counts.get(cls, 0)
        if n_total == 0:
            continue
        if n_total > cap_per_class:
            idx = rng.choice(n_total, cap_per_class, replace=False)
            sub = sub.iloc[idx]
        ax.scatter(
            sub["w"].to_numpy(), sub["h"].to_numpy(),
            c=CLASS_PALETTE[cls], s=5, alpha=0.45, lw=0,
            label=f"{cls.capitalize()}  (n = {n_total:,})",
        )

    # "Small object" zone: w < 8 AND h < 8 px.
    ax.axvline(SMALL_THR, color="0.35", lw=0.9, ls="--", alpha=0.8, zorder=2)
    ax.axhline(SMALL_THR, color="0.35", lw=0.9, ls="--", alpha=0.8, zorder=2)

    # Faint shading of the bottom-left small-object zone.
    ax.axvspan(0, SMALL_THR, ymin=0, ymax=1, color="0.5", alpha=0.05, zorder=0)
    ax.axhspan(0, SMALL_THR, xmin=0, xmax=1, color="0.5", alpha=0.05, zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")

    lim = (2.0, 1000.0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    # No set_aspect("equal"): the panel must match the SR/NPR plot's wide aspect
    # so the two subfigures are visually balanced side-by-side. Log axes still
    # span the same decades, so the dashed 8 px reference lines stay correct.

    ticks = [3, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])

    ax.set_xlabel("Width  (px)", labelpad=2)
    ax.set_ylabel("Height  (px)")

    ax.grid(True, which="both", lw=0.3, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Label the small-object threshold near each dashed line.
    ax.text(SMALL_THR, lim[1] * 0.92, "  8 px", fontsize=7, color="0.35",
            ha="left", va="top")
    ax.text(lim[1] * 0.92, SMALL_THR, "8 px", fontsize=7, color="0.35",
            ha="right", va="bottom")

    # Figure-level horizontal legend in the bottom band, mirroring the SR/NPR
    # plot. Anchored at fig coords so the saved PNG keeps the (7.1, 2.74)
    # aspect — IMPORTANT: do NOT pass bbox_inches="tight" to savefig, otherwise
    # matplotlib retrims the canvas and the ratio drifts.
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=len(CLASSES), frameon=True, fontsize=7,
        markerscale=2.5, handletextpad=0.3, borderpad=0.4,
        columnspacing=1.2,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("0.6")

    # Override rcParam savefig.bbox="tight" (set by apply_neurips_style); we want
    # the saved canvas to match figsize exactly so the aspect ratio is locked
    # and the data axes sit at the precise position we configured via add_axes.
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing datasets …")
    df = collect()

    # Quick sanity: per dataset×class counts
    counts = df.groupby(["dataset", "class"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=DATASETS, columns=CLASSES, fill_value=0)
    print("\nbox counts:\n" + counts.to_string())
    print(f"\ntotal boxes: {len(df):,}")

    summary_csv = OUT_DIR / "mot_bbox_size_summary.csv"
    write_summary(df, summary_csv)
    print(f"summary → {summary_csv}")

    out_png = OUT_DIR / "mot_bbox_size_scatter.png"
    out_pdf = OUT_DIR / "mot_bbox_size_scatter.pdf"
    plot_scatter(df, out_png, out_pdf)
    print(f"figure  → {out_png}")
    print(f"figure  → {out_pdf}")


if __name__ == "__main__":
    main()
