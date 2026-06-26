"""
Per-attribute SOT performance restricted to *tiny* frames (sqrt(area) < 8 px),
micro-averaged over qualifying frames.

Differences vs. ``sot_unified_attribute_table.py`` and
``sot_unique_attribute_table.py`` (the per-sequence variants):

  1. **Frame-level tiny filter.** A frame is *tiny* iff its GT bbox
     ``sqrt(area) < 8`` px. SV248S frames flagged invisible (state==1) are
     excluded.
  2. **Micro averaging.** Within each (dataset, attribute) group we pool
     all tiny frames across all sequences carrying the attribute, then
     compute single rates:
       - SR  = mean over 21 IoU thresholds of frac(tiny frames with IoU >= t)
       - PR  = mean over 51 CLE thresholds [0, 50] px of frac(CLE <= t)
       - P@5 = frac(tiny frames with CLE <= 5 px)
  3. For *unified* attributes (BC/IV/ROT/OCC/SOB/DEF), we then arithmetic-mean
     these per-(dataset, attribute) micro rates across the annotating datasets.
     For dataset-unique attributes (incl. occlusion sub-types), the single
     annotating dataset's rate is reported directly.

Outputs (under ``--out``):
  - ``per_dataset_attr_tiny.csv`` — (tracker, dataset, attr, n_seq, n_tiny_frames, SR, PR, P@5)
  - ``unified_attr_tiny.csv``    — (tracker, attr, n_datasets, SR, PR, P@5)
  - ``unique_attr_tiny.csv``     — (tracker, attr, dataset, n_seq, n_tiny_frames, SR, PR, P@5)

Usage::

    micromamba run -n esa_dlstem python tools/sot_attribute_table_tiny.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ootb import OOTBDataset
from datasets.satsot import SatSOTDataset
from datasets.sv248s import SV248SDataset


# --- Curve thresholds (consistent with main table; micro-pooled here) ---
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)

TINY_SQRT_AREA_THRESH = 8.0


# --- Attribute taxonomy (same as the per-sequence tables) ---
UNIFIED_ATTR_MAP: dict[str, dict[str, list[str]]] = {
    "BC":  {"satsot": ["BC"],            "sv248s": [],                   "ootb": ["BC"]},
    "IV":  {"satsot": ["IV"],            "sv248s": ["IV"],               "ootb": ["IV"]},
    "ROT": {"satsot": ["ROT"],           "sv248s": ["IPR"],              "ootb": ["IPR"]},
    "OCC": {"satsot": ["POC", "FOC"],    "sv248s": ["STO", "LTO", "CO"], "ootb": ["PO", "FO"]},
    "SOB": {"satsot": ["SOB"],           "sv248s": ["DS"],               "ootb": ["SA"]},
    "DEF": {"satsot": ["DEF"],           "sv248s": [],                   "ootb": ["DEF"]},
}
ATTR_ORDER = ["BC", "IV", "ROT", "OCC", "SOB", "DEF"]
DATASET_ORDER = ["satsot", "sv248s", "ootb"]

# (display label, {dataset: [native attr names]}) for non-shared taxonomy
# rows + occlusion sub-types. POC/FOC pool SatSOT+OOTB (mirrors the OCC
# unification in the shared block); every other entry is single-dataset.
UNIQUE_ATTRS: list[tuple[str, dict[str, list[str]]]] = [
    ("ARC", {"satsot": ["ARC"]}),
    ("OON", {"ootb":   ["OON"]}),
    ("LQ",  {"satsot": ["LQ"]}),
    ("TO",  {"satsot": ["TO"]}),
    ("BJT", {"satsot": ["BJT"]}),
    ("BCH", {"sv248s": ["BCH"]}),
    ("ND",  {"sv248s": ["ND"]}),
    ("IBG", {"sv248s": ["BCL"]}),
    ("SM",  {"sv248s": ["SM"]}),
    ("LT",  {"ootb":   ["LT"]}),
    ("MB",  {"ootb":   ["MB"]}),
    ("IM",  {"ootb":   ["IM"]}),
    ("AM",  {"ootb":   ["AM"]}),
]

OCCLUSION_SUBTYPES: list[tuple[str, dict[str, list[str]]]] = [
    ("POC", {"satsot": ["POC"], "ootb": ["PO"]}),
    ("FOC", {"satsot": ["FOC"], "ootb": ["FO"]}),
    ("STO", {"sv248s": ["STO"]}),
    ("LTO", {"sv248s": ["LTO"]}),
    ("CO",  {"sv248s": ["CO"]}),
]

DATASET_ROOTS = {
    "ootb":   "/data/ESA_DLSTEM_2025/data/trafic/OOTB",
    "satsot": "/data/ESA_DLSTEM_2025/data/trafic/SatSOT",
    "sv248s": "/data/ESA_DLSTEM_2025/data/trafic/SV248S",
}

DATASET_TAGS = {
    "ootb":   re.compile(r"_ootb_"),
    "satsot": re.compile(r"_satsot_"),
    "sv248s": re.compile(r"_sv248s_"),
}


# ============================================================
# Per-frame GT area lookups (for tiny filter)
# ============================================================

def _polygon_area(coords: np.ndarray) -> float:
    xs = coords[0::2]
    ys = coords[1::2]
    return 0.5 * abs(
        xs[0] * (ys[1] - ys[3])
        + xs[1] * (ys[2] - ys[0])
        + xs[2] * (ys[3] - ys[1])
        + xs[3] * (ys[0] - ys[2])
    )


def _ootb_areas(gt_path: Path) -> list[float | None]:
    areas: list[float | None] = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                areas.append(None); continue
            vals = re.split(r"[,\t ]+", line)
            try:
                coords = np.array([float(v) for v in vals[:8]], dtype=np.float64)
            except ValueError:
                areas.append(None); continue
            if coords.size < 8 or np.isnan(coords).any():
                areas.append(None); continue
            a = _polygon_area(coords)
            areas.append(a if a > 0 else None)
    return areas


def _satsot_areas(gt_path: Path) -> list[float | None]:
    areas: list[float | None] = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or "none" in line.lower():
                areas.append(None); continue
            vals = re.split(r"[,\t ]+", line)
            try:
                xywh = [float(v) for v in vals[:4]]
            except ValueError:
                areas.append(None); continue
            if len(xywh) < 4:
                areas.append(None); continue
            w, h = xywh[2], xywh[3]
            areas.append(w * h if (w > 0 and h > 0) else None)
    return areas


def _sv248s_areas(rect_path: Path, state_path: Path) -> list[float | None]:
    rects: list[tuple[float, float]] = []
    with open(rect_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                rects.append((0.0, 0.0)); continue
            vals = re.split(r"[,\t ]+", line)
            try:
                w, h = float(vals[2]), float(vals[3])
            except (IndexError, ValueError):
                rects.append((0.0, 0.0))
                continue
            rects.append((w, h))
    states: list[int] = [0] * len(rects)
    if state_path.exists():
        with open(state_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or i >= len(states):
                    continue
                try:
                    states[i] = int(line)
                except ValueError:
                    pass
    out: list[float | None] = []
    for (w, h), st in zip(rects, states):
        if st == 1 or w <= 0 or h <= 0:
            out.append(None)
        else:
            out.append(w * h)
    return out


def build_area_lookup(dataset: str) -> dict[str, list[float | None]]:
    """{video_id: [area_or_None_per_frame_id]}"""
    root = Path(DATASET_ROOTS[dataset])
    out: dict[str, list[float | None]] = {}

    if dataset == "ootb":
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name == "anno":
                continue
            gt = seq_dir / "groundtruth.txt"
            if gt.exists():
                out[seq_dir.name] = _ootb_areas(gt)

    elif dataset == "satsot":
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt = seq_dir / "groundtruth.txt"
            if gt.exists():
                out[seq_dir.name] = _satsot_areas(gt)

    elif dataset == "sv248s":
        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue
            ann_dir = video_dir / "annotations"
            seq_dir_root = video_dir / "sequences"
            if not ann_dir.is_dir() or not seq_dir_root.is_dir():
                continue
            for seq in sorted(seq_dir_root.iterdir()):
                if not seq.is_dir():
                    continue
                rect = ann_dir / f"{seq.name}.rect"
                state = ann_dir / f"{seq.name}.state"
                if not rect.exists():
                    continue
                vid = f"{video_dir.name}/{seq.name}"
                out[vid] = _sv248s_areas(rect, state)

    else:
        raise ValueError(dataset)

    return out


# ============================================================
# Sequence attribute lookup
# ============================================================

def load_sequence_attributes() -> dict[str, dict[str, list[str]]]:
    return {
        "ootb":   OOTBDataset(  root=DATASET_ROOTS["ootb"],   split="no_split", mode="detection").sequence_attributes(),
        "satsot": SatSOTDataset(root=DATASET_ROOTS["satsot"], split="no_split", mode="detection").sequence_attributes(),
        "sv248s": SV248SDataset(root=DATASET_ROOTS["sv248s"], split="no_split", mode="detection").sequence_attributes(),
    }


def videos_with_natives(seq_attrs: dict[str, list[str]], natives: list[str]) -> set[str]:
    target = set(natives)
    if not target:
        return set()
    return {vid for vid, attrs in seq_attrs.items() if target & set(attrs)}


# ============================================================
# Run discovery
# ============================================================

def detect_dataset(run_name: str) -> str | None:
    for ds, pat in DATASET_TAGS.items():
        if pat.search(run_name):
            return ds
    return None


def discover_runs(root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for tracker_dir in sorted(root.iterdir()):
        if not tracker_dir.is_dir() or tracker_dir.name == "analysis":
            continue
        for run_dir in sorted(tracker_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "per_image_metrics.json").exists():
                continue
            ds = detect_dataset(run_dir.name)
            if ds is None:
                continue
            out.append((tracker_dir.name, ds, run_dir))
    return out


# ============================================================
# Micro stats from a list of records
# ============================================================

def _micro_stats(records: list[dict]) -> dict:
    if not records:
        return {"SR": np.nan, "PR": np.nan, "P5": np.nan, "n_frames": 0}
    ious = np.fromiter((r["best_iou"] for r in records), dtype=np.float64, count=len(records))
    cd   = np.fromiter((r["center_dist"] for r in records), dtype=np.float64, count=len(records))
    return {
        "SR": float(np.mean([(ious >= t).mean() for t in SUCCESS_THRESHOLDS])),
        "PR": float(np.mean([(cd <= t).mean() for t in PRECISION_THRESHOLDS])),
        "P5": float((cd <= 5.0).mean()),
        "n_frames": len(records),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/work/anon/experiments/NeurIPS/SOT_whole_dataset_04_22",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", default=None,
                    help="Path for the generated .tex file. "
                         "Defaults to NeurIPS submission tables dir.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / "analysis" / "tiny_attributes")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] building per-frame GT area lookups (sqrt(area) < 8 px tiny filter)")
    area_lookup = {ds: build_area_lookup(ds) for ds in DATASET_ORDER}
    for ds, lk in area_lookup.items():
        n_tiny = sum(
            1 for vid, areas in lk.items()
            for a in areas
            if a is not None and a < TINY_SQRT_AREA_THRESH ** 2
        )
        n_total = sum(1 for vid, areas in lk.items() for a in areas if a is not None)
        print(f"   {ds:7s}: {len(lk):4d} seqs, {n_total:7d} valid frames, "
              f"{n_tiny:6d} tiny frames ({100.0*n_tiny/max(1,n_total):.1f}%)")

    print("[2/5] loading sequence attributes")
    seq_attrs = load_sequence_attributes()

    # Pre-compute video sets per (dataset, attribute_label) — for unified, unique, and occlusion subtypes.
    attr_video_sets: dict[tuple[str, str], set[str]] = {}
    for attr in ATTR_ORDER:
        for ds in DATASET_ORDER:
            attr_video_sets[(ds, attr)] = videos_with_natives(
                seq_attrs[ds], UNIFIED_ATTR_MAP[attr][ds],
            )
    for label, ds_natives in UNIQUE_ATTRS + OCCLUSION_SUBTYPES:
        for ds, natives in ds_natives.items():
            attr_video_sets[(ds, label)] = videos_with_natives(seq_attrs[ds], natives)

    # All labels we care about (we output one micro stat per (tracker, dataset, label) where applicable).
    all_labels: list[tuple[str, str]] = []
    for attr in ATTR_ORDER:
        for ds in DATASET_ORDER:
            if UNIFIED_ATTR_MAP[attr][ds]:
                all_labels.append((ds, attr))
    for label, ds_natives in UNIQUE_ATTRS + OCCLUSION_SUBTYPES:
        for ds in ds_natives:
            all_labels.append((ds, label))

    print("[3/5] aggregating tiny-frame micro stats per (tracker, dataset, attribute)")
    runs = discover_runs(root)

    # Cache per-(tracker, dataset) tiny-frame grouping by video_id.
    tiny_by_tracker_ds: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for tracker, ds, run_dir in runs:
        with open(run_dir / "per_image_metrics.json") as f:
            frames = json.load(f)

        ds_areas = area_lookup[ds]

        # Group tiny records by video_id.
        by_seq_tiny: dict[str, list[dict]] = defaultdict(list)
        for fr in frames:
            vid = str(fr["video_id"])
            fid = int(fr["frame_id"])
            seq_areas = ds_areas.get(vid)
            if seq_areas is None or fid >= len(seq_areas):
                continue
            a = seq_areas[fid]
            if a is None or a >= TINY_SQRT_AREA_THRESH ** 2:
                continue
            recs = fr.get("sot_records") or []
            by_seq_tiny[vid].extend(recs)
        tiny_by_tracker_ds[(tracker, ds)] = dict(by_seq_tiny)

    trackers = sorted({t for t, _ in tiny_by_tracker_ds.keys()})

    # tracker × (dataset, label) -> stats dict (per-dataset breakdown, kept for diagnostics)
    per_dataset_rows: list[dict] = []
    # tracker × attr (unified) -> list of dicts (one per annotating dataset, for arithmetic-mean)
    per_attr_dataset_metrics: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for (tracker, ds), by_seq_tiny in tiny_by_tracker_ds.items():
        # Iterate every (label, dataset) where label applies to this dataset.
        labels_for_ds: list[str] = []
        for attr in ATTR_ORDER:
            if UNIFIED_ATTR_MAP[attr][ds]:
                labels_for_ds.append(attr)
        for label, ds_natives in UNIQUE_ATTRS + OCCLUSION_SUBTYPES:
            if ds in ds_natives:
                labels_for_ds.append(label)

        for label in labels_for_ds:
            vids = attr_video_sets[(ds, label)]
            picked_records: list[dict] = []
            n_seq_with_tiny = 0
            for v in vids:
                if v in by_seq_tiny:
                    picked_records.extend(by_seq_tiny[v])
                    n_seq_with_tiny += 1
            stats = _micro_stats(picked_records)
            per_dataset_rows.append({
                "tracker":        tracker,
                "dataset":        ds,
                "attribute":      label,
                "n_seq_with_tiny": n_seq_with_tiny,
                "n_seq_in_attr":   len(vids),
                "n_tiny_frames":   stats["n_frames"],
                "SR":  round(stats["SR"], 4)  if stats["n_frames"] else None,
                "PR":  round(stats["PR"], 4)  if stats["n_frames"] else None,
                "P5":  round(stats["P5"], 4)  if stats["n_frames"] else None,
            })
            if label in ATTR_ORDER and stats["n_frames"]:
                per_attr_dataset_metrics[tracker][label].append({
                    "dataset": ds, "SR": stats["SR"], "PR": stats["PR"], "P5": stats["P5"],
                    "n_tiny_frames": stats["n_frames"],
                })

    # Cross-dataset tiny-frame pooling for non-unified taxonomy rows
    # (POC/FOC pool SatSOT+OOTB; STO/LTO/CO and other dataset-unique rows
    # have a single annotating dataset so the "pool" is a no-op).
    pooled_rows: list[dict] = []
    for tracker in trackers:
        for label, ds_natives in UNIQUE_ATTRS + OCCLUSION_SUBTYPES:
            picked_records: list[dict] = []
            n_seq_with_tiny = 0
            for ds in ds_natives:
                by_seq_tiny = tiny_by_tracker_ds.get((tracker, ds), {})
                vids = attr_video_sets[(ds, label)]
                for v in vids:
                    if v in by_seq_tiny:
                        picked_records.extend(by_seq_tiny[v])
                        n_seq_with_tiny += 1
            stats = _micro_stats(picked_records)
            if not stats["n_frames"]:
                continue
            pooled_rows.append({
                "tracker":        tracker,
                "attribute":      label,
                "datasets":       "|".join(sorted(ds_natives.keys())),
                "n_seq_with_tiny": n_seq_with_tiny,
                "n_tiny_frames":   stats["n_frames"],
                "SR":  round(stats["SR"], 4),
                "PR":  round(stats["PR"], 4),
                "P5":  round(stats["P5"], 4),
            })

    print("[4/5] averaging unified-attribute rates across annotating datasets")
    unified_rows: list[dict] = []
    for tracker in sorted(per_attr_dataset_metrics.keys()):
        for attr in ATTR_ORDER:
            ds_metrics = per_attr_dataset_metrics[tracker][attr]
            if not ds_metrics:
                continue
            unified_rows.append({
                "tracker":     tracker,
                "attribute":   attr,
                "n_datasets":  len(ds_metrics),
                "datasets":    "|".join(sorted(m["dataset"] for m in ds_metrics)),
                "n_tiny_frames": int(sum(m["n_tiny_frames"] for m in ds_metrics)),
                "SR":  round(float(np.mean([m["SR"] for m in ds_metrics])), 4),
                "PR":  round(float(np.mean([m["PR"] for m in ds_metrics])), 4),
                "P5":  round(float(np.mean([m["P5"] for m in ds_metrics])), 4),
            })

    # Unique attribute & occlusion subtype rows: use the pooled-across-datasets
    # rates (POC/FOC pool SatSOT+OOTB; others are single-dataset = no-op pool).
    unique_rows: list[dict] = [
        {
            "tracker":       r["tracker"],
            "attribute":     r["attribute"],
            "datasets":      r["datasets"],
            "n_tiny_frames": r["n_tiny_frames"],
            "SR":  r["SR"], "PR":  r["PR"], "P5":  r["P5"],
        }
        for r in pooled_rows
    ]

    print("[5/5] writing CSVs")
    per_dataset_rows.sort(key=lambda r: (r["tracker"], r["attribute"], r["dataset"]))
    unified_rows.sort(key=lambda r: (r["tracker"], r["attribute"]))
    unique_rows.sort(key=lambda r: (r["tracker"], r["attribute"]))

    def _write(path, rows):
        if not rows:
            print(f"   skip {path.name} (no rows)"); return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"   {path.name}: {len(rows)} rows")

    _write(out_dir / "per_dataset_attr_tiny.csv", per_dataset_rows)
    _write(out_dir / "unified_attr_tiny.csv",     unified_rows)
    _write(out_dir / "unique_attr_tiny.csv",      unique_rows)

    # Pretty-print summary.
    trackers = sorted({r["tracker"] for r in unified_rows})
    print("\nMicro tiny-frame unified-attribute table (SR / PR / P@5):")
    header = "tracker".ljust(10) + " | " + " | ".join(f"{a:>20s}" for a in ATTR_ORDER)
    print(header); print("-" * len(header))
    by_ta = {(r["tracker"], r["attribute"]): r for r in unified_rows}
    for tr in trackers:
        cells = []
        for a in ATTR_ORDER:
            r = by_ta.get((tr, a))
            cells.append("---".center(20) if r is None
                         else f"{r['SR']:.3f}/{r['PR']:.3f}/{r['P5']:.3f}".rjust(20))
        print(tr.ljust(10) + " | " + " | ".join(cells))

    # ---------- LaTeX emission ----------
    tex_path = Path(args.tex) if args.tex else (
        Path(__file__).resolve().parents[1]
        / "Formatting Instructions For NeurIPS 2026"
        / "tables" / "SOT" / "whole_dataset"
        / "sot_attributes_table_tiny.tex"
    )
    print(f"\nWriting LaTeX → {tex_path}")
    emit_latex(tex_path, unified_rows, unique_rows, per_dataset_rows)


# ============================================================
# LaTeX emission
# ============================================================

# Order of trackers + display labels for the .tex header.
TRACKER_ORDER = ["siamrpn", "ostrack", "odtrack", "lorat", "sam2", "samurai", "sam3"]
TRACKER_DISPLAY = {
    "siamrpn": (r"\textbf{SiamRPN++}~\cite{li2019siamrpn++}",      "SiamRPN++"),
    "ostrack": (r"\textbf{OSTrack-384}~\cite{ye2022joint}",        "OSTrack-384"),
    "odtrack": (r"\textbf{ODTrack}~\cite{zheng2024odtrack}",       "ODTrack"),
    "lorat":   (r"\textbf{LoRAT-g378}~\cite{lin2024tracking}",     "LoRAT-g378"),
    "sam2":    (r"\textbf{SAM 2}~\cite{ravi2024sam}",              "SAM 2"),
    "samurai": (r"\textbf{SAMURAI}~\cite{yang2026samurai}",        "SAMURAI"),
    "sam3":    (r"\textbf{SAM 3}~\cite{carion2026sam3}",           "SAM 3"),
}

# (label, source).
# - source == "" → look up in unified_rows (BC/IV/ROT/OCC/SOB/DEF).
# - source is a string of "|" separated dataset codes → look up in unique_rows.
#   Single-dataset rows render the dataset tag next to the abbreviation in the
#   LaTeX table; pooled rows (POC/FOC) leave the tag off because the row is
#   now defined across all annotating datasets.
MAIN_ROWS_UNIFIED = [
    ("BC",  ""),
    ("IV",  ""),
    ("ROT", ""),
    ("OCC", ""),
    ("SOB", ""),
    ("DEF", ""),
]
MAIN_ROWS_AR = [
    ("ARC", "satsot"),
    ("OON", "ootb"),
]
MAIN_ROWS_OTHER = [
    ("LQ",  "satsot"),
    ("TO",  "satsot"),
    ("BJT", "satsot"),
    ("BCH", "sv248s"),
    ("ND",  "sv248s"),
    ("IBG", "sv248s"),
    ("SM",  "sv248s"),
    ("LT",  "ootb"),
    ("MB",  "ootb"),
    ("IM",  "ootb"),
    ("AM",  "ootb"),
]
# POC/FOC pool SatSOT+OOTB → empty dataset tag, row label printed without
# "(SatSOT)" suffix. STO/LTO/CO remain SV248S-only.
OCC_ROWS = [
    ("POC", "ootb|satsot"),
    ("FOC", "ootb|satsot"),
    ("STO", "sv248s"),
    ("LTO", "sv248s"),
    ("CO",  "sv248s"),
]

_DS_DISPLAY = {"satsot": "SatSOT", "sv248s": "SV248S", "ootb": "OOTB"}


def _fmt_cell(val: float | None, is_best: bool, is_second: bool) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return r"---"
    s = f"{val:.3f}"
    if is_best:
        return r"\textbf{" + s + "}"
    if is_second:
        return r"\underline{" + s + "}"
    return s


def _row_values(label: str, source: str, unified_rows: list[dict],
                unique_rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """tracker → {SR, PR, P5} or None values, for the requested attribute row.

    ``source == ""``  → match unified_rows by label.
    Other ``source`` → match unique_rows by (label, datasets), where the
    ``datasets`` field is the pipe-joined sorted list of annotating dataset
    codes from the pooled-row builder (e.g. "ootb|satsot" for POC/FOC,
    "satsot" for LQ, "sv248s" for STO).
    """
    out: dict[str, dict[str, float | None]] = {}
    if source == "":
        for r in unified_rows:
            if r["attribute"] == label:
                out[r["tracker"]] = {"SR": r["SR"], "PR": r["PR"], "P5": r["P5"]}
    else:
        for r in unique_rows:
            if r["attribute"] == label and r["datasets"] == source:
                out[r["tracker"]] = {"SR": r["SR"], "PR": r["PR"], "P5": r["P5"]}
    for tr in TRACKER_ORDER:
        out.setdefault(tr, {"SR": None, "PR": None, "P5": None})
    return out


def _format_row(label: str, source: str, unified_rows: list[dict],
                unique_rows: list[dict]) -> str:
    vals = _row_values(label, source, unified_rows, unique_rows)
    # determine best / second per metric
    rank: dict[str, list[str]] = {}
    for metric in ("SR", "PR", "P5"):
        scored = [(tr, vals[tr][metric]) for tr in TRACKER_ORDER if vals[tr][metric] is not None]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        rank[metric] = [tr for tr, _ in scored]

    # build cells
    cells = []
    for tr in TRACKER_ORDER:
        for metric in ("SR", "PR", "P5"):
            v = vals[tr][metric]
            best = rank[metric][:1]
            second = rank[metric][1:2]
            cells.append(_fmt_cell(v, tr in best, tr in second))
    label_cell = label
    # Only print a dataset tag for genuinely single-dataset rows. Pooled
    # rows (POC/FOC) carry a "|"-joined source like "ootb|satsot" and are
    # printed without a per-row dataset suffix.
    if source and "|" not in source:
        label_cell += r" \tiny(" + _DS_DISPLAY[source] + ")"
    return label_cell + " & " + " & ".join(cells) + r" \\"


_HEADER_BLOCK = r"""\toprule
\multirow{2}{*}{\textbf{Attr.}}
 & \multicolumn{3}{c}{\textbf{SiamRPN++}~\cite{li2019siamrpn++}}
 & \multicolumn{3}{c}{\textbf{OSTrack-384}~\cite{ye2022joint}}
 & \multicolumn{3}{c}{\textbf{ODTrack}~\cite{zheng2024odtrack}}
 & \multicolumn{3}{c}{\textbf{LoRAT-g378}~\cite{lin2024tracking}}
 & \multicolumn{3}{c}{\textbf{SAM 2}~\cite{ravi2024sam}}
 & \multicolumn{3}{c}{\textbf{SAMURAI}~\cite{yang2026samurai}}
 & \multicolumn{3}{c}{\textbf{SAM 3}~\cite{carion2026sam3}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}
\cmidrule(lr){14-16} \cmidrule(lr){17-19} \cmidrule(lr){20-22}
 & SR & PR & P@5 & SR & PR & P@5 & SR & PR & P@5 & SR & PR & P@5
   & SR & PR & P@5 & SR & PR & P@5 & SR & PR & P@5 \\
\midrule"""


def emit_latex(tex_path: Path, unified_rows: list[dict],
               unique_rows: list[dict], per_dataset_rows: list[dict]) -> None:
    # main table rows
    main_lines = []
    main_lines.append(r"\multicolumn{22}{l}{\emph{Unified attributes (across SatSOT / SV248S / OOTB)}} \\")
    for label, source in MAIN_ROWS_UNIFIED:
        main_lines.append(_format_row(label, source, unified_rows, unique_rows))
    main_lines.append(r"\midrule")
    main_lines.append(r"\multicolumn{22}{l}{\emph{Aspect-ratio attributes (dataset-unique)}} \\")
    for label, source in MAIN_ROWS_AR:
        main_lines.append(_format_row(label, source, unified_rows, unique_rows))
    main_lines.append(r"\midrule")
    main_lines.append(r"\multicolumn{22}{l}{\emph{Other dataset-unique attributes}} \\")
    for label, source in MAIN_ROWS_OTHER:
        main_lines.append(_format_row(label, source, unified_rows, unique_rows))

    occ_lines = [_format_row(label, source, unified_rows, unique_rows)
                 for label, source in OCC_ROWS]

    main_caption = (
        "SOT performance on \\emph{tiny} frames (per-frame GT $\\sqrt{\\mathrm{area}} < 8$\\,px) "
        "for the attributes available in Space-tracker-SOT. "
        "Metrics are \\emph{micro}-averaged over all qualifying tiny frames pooled across every dataset that annotates the attribute: "
        "SR is the AUC over IoU thresholds in $[0, 1]$, PR is the AUC over CLE thresholds in $[0, 50]$\\,px, "
        "and P@5 is the fraction of tiny frames with CLE $\\leq 5$\\,px. "
        "For \\emph{unified} attributes (BC / IV / ROT / OCC / SOB / DEF) we first compute per-(dataset, attribute) micro rates and then arithmetic-mean across the annotating datasets (BC and DEF: 2 datasets; IV / ROT / OCC / SOB: 3 datasets), preserving cross-dataset balance. "
        "Dataset-unique rows (ARC / OON / LQ / TO / BJT / BCH / ND / IBG / SM / LT / MB / IM / AM) are computed on the single annotating dataset, shown next to the abbreviation. "
        "Rows with no tiny frames in any annotating dataset are reported as ``---''. "
        "Occlusion sub-types of OCC are reported separately in Tab.~\\ref{tab:sot_attributes_tiny_occlusion}. "
        "\\textbf{Bold} = best, \\underline{underline} = second best, applied per row and per metric. "
        "Methodologically distinct from Tab.~\\ref{tab:sot_attributes}, which uses per-sequence aggregation over the full size range."
    )

    occ_caption = (
        "Tiny-frame SOT performance on the occlusion sub-types of the unified \\textbf{OCC} attribute (Tab.~\\ref{tab:sot_attributes_tiny}). "
        "POC pools SatSOT POC with OOTB PO; FOC pools SatSOT FOC with OOTB FO (mirroring the OCC unification in Tab.~\\ref{tab:sot_attributes_tiny}); "
        "STO / LTO / CO are annotated only by SV248S (temporal axis, no spatial-axis counterpart in the other two datasets). "
        "Metrics are \\emph{micro}-averaged over all tiny frames (sqrt(area) $< 8$\\,px) pooled across every annotating dataset; see Tab.~\\ref{tab:sot_attributes_tiny} caption for definitions of SR, PR, P@5. "
        "\\textbf{Bold} = best, \\underline{underline} = second best, applied per row and per metric."
    )

    body = []
    body.append(r"% Auto-generated by tools/sot_attribute_table_tiny.py — do not edit by hand.")
    body.append(r"% Requires: \usepackage{booktabs, multirow, array, graphicx}")
    body.append(r"\begin{table*}[t]")
    body.append(r"\centering")
    body.append(r"\caption{" + main_caption + "}")
    body.append(r"\label{tab:sot_attributes_tiny}")
    body.append(r"\setlength{\tabcolsep}{3pt}")
    body.append(r"\renewcommand{\arraystretch}{1.15}")
    body.append(r"\resizebox{\textwidth}{!}{%")
    body.append(r"\begin{tabular}{l ccc ccc ccc ccc ccc ccc ccc}")
    body.append(_HEADER_BLOCK)
    body.extend(main_lines)
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}%")
    body.append(r"}")
    body.append(r"\end{table*}")
    body.append("")
    body.append(r"\begin{table*}[t]")
    body.append(r"\centering")
    body.append(r"\caption{" + occ_caption + "}")
    body.append(r"\label{tab:sot_attributes_tiny_occlusion}")
    body.append(r"\setlength{\tabcolsep}{3pt}")
    body.append(r"\renewcommand{\arraystretch}{1.15}")
    body.append(r"\resizebox{\textwidth}{!}{%")
    body.append(r"\begin{tabular}{l ccc ccc ccc ccc ccc ccc ccc}")
    body.append(_HEADER_BLOCK)
    body.extend(occ_lines)
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}%")
    body.append(r"}")
    body.append(r"\end{table*}")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
