"""
Per-tracker SR / NPR / PR for the 18 dataset-unique sequence attributes.

For each unique native attribute (POC, FOC, STO, LTO, CO, ARC, OON, LQ, TO,
BJT, BCH, ND, IBG≡SV248S BCL, SM, LT, MB, IM, AM) we compute per-sequence
SR / NPR / PR (per-sequence aggregation; PR over CLE in [0, 50] px) on the
sequences in the annotating dataset that carry this attribute, then average
across those sequences. No cross-dataset averaging — every unique attribute
lives in exactly one dataset.

Output:
  - ``<root>/analysis/unified_attributes/unique_attr.csv``
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


SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)
NORM_PRECISION_THRESHOLDS = np.linspace(0.0, 0.5, 21)

DATASET_ROOTS = {
    "ootb":   "/data/ESA_DLSTEM_2025/data/trafic/OOTB",
    "satsot": "/data/ESA_DLSTEM_2025/data/trafic/SatSOT",
    "sv248s": "/data/ESA_DLSTEM_2025/data/trafic/SV248S",
}

# (display label, dataset, native attr names that compose it)
UNIQUE_ATTRS: list[tuple[str, str, list[str]]] = [
    ("POC", "satsot", ["POC"]),
    ("FOC", "satsot", ["FOC"]),
    ("STO", "sv248s", ["STO"]),
    ("LTO", "sv248s", ["LTO"]),
    ("CO",  "sv248s", ["CO"]),
    ("ARC", "satsot", ["ARC"]),
    ("OON", "ootb",   ["OON"]),
    ("LQ",  "satsot", ["LQ"]),
    ("TO",  "satsot", ["TO"]),
    ("BJT", "satsot", ["BJT"]),
    ("BCH", "sv248s", ["BCH"]),
    ("ND",  "sv248s", ["ND"]),
    ("IBG", "sv248s", ["BCL"]),
    ("SM",  "sv248s", ["SM"]),
    ("LT",  "ootb",   ["LT"]),
    ("MB",  "ootb",   ["MB"]),
    ("IM",  "ootb",   ["IM"]),
    ("AM",  "ootb",   ["AM"]),
]

DATASET_TAGS = {
    "ootb":   re.compile(r"_ootb_"),
    "satsot": re.compile(r"_satsot_"),
    "sv248s": re.compile(r"_sv248s_"),
}


def _seq_summary(records: list[dict]) -> dict:
    ious = np.array([r["best_iou"] for r in records], dtype=np.float64)
    cd   = np.array([r["center_dist"] for r in records], dtype=np.float64)
    nc   = np.array([r["norm_center_dist"] for r in records], dtype=np.float64)
    return {
        "success_auc":        float(np.mean([(ious >= t).mean() for t in SUCCESS_THRESHOLDS])),
        "precision_auc":      float(np.mean([(cd <= t).mean() for t in PRECISION_THRESHOLDS])),
        "norm_precision_auc": float(np.mean([(nc <= t).mean() for t in NORM_PRECISION_THRESHOLDS])),
        "n_frames": len(records),
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Defaults to <root>/analysis/unified_attributes/unique_attr.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out) if args.out else (
        root / "analysis" / "unified_attributes" / "unique_attr.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading dataset attribute flags")
    seq_attrs = {
        "ootb":   OOTBDataset(  root=DATASET_ROOTS["ootb"],   split="no_split", mode="detection").sequence_attributes(),
        "satsot": SatSOTDataset(root=DATASET_ROOTS["satsot"], split="no_split", mode="detection").sequence_attributes(),
        "sv248s": SV248SDataset(root=DATASET_ROOTS["sv248s"], split="no_split", mode="detection").sequence_attributes(),
    }

    # Pre-compute video sets per (dataset, native_attr).
    attr_video_sets: dict[tuple[str, str], set[str]] = {}
    for label, ds, natives in UNIQUE_ATTRS:
        target = set(natives)
        vids = {v for v, attrs in seq_attrs[ds].items() if target & set(attrs)}
        attr_video_sets[(ds, label)] = vids

    print("[2/3] aggregating per-tracker × unique-attribute metrics")
    runs = discover_runs(root)
    rows: list[dict] = []
    for tracker, ds, run_dir in runs:
        with open(run_dir / "per_image_metrics.json") as f:
            frames = json.load(f)

        by_seq_records: dict[str, list[dict]] = defaultdict(list)
        for fr in frames:
            by_seq_records[str(fr["video_id"])].extend(fr.get("sot_records", []))

        seq_summary = {
            v: _seq_summary(recs) for v, recs in by_seq_records.items() if recs
        }

        for label, attr_ds, _ in UNIQUE_ATTRS:
            if attr_ds != ds:
                continue
            picked = [seq_summary[v] for v in attr_video_sets[(ds, label)] if v in seq_summary]
            if not picked:
                continue
            rows.append({
                "tracker":   tracker,
                "dataset":   ds,
                "attribute": label,
                "n_seq":     len(picked),
                "n_frames":  int(sum(s["n_frames"] for s in picked)),
                "SR":        round(float(np.mean([s["success_auc"]        for s in picked])), 4),
                "NPR":       round(float(np.mean([s["norm_precision_auc"] for s in picked])), 4),
                "PR":        round(float(np.mean([s["precision_auc"]      for s in picked])), 4),
            })

    rows.sort(key=lambda r: (r["tracker"], r["attribute"]))

    print(f"[3/3] writing {out_path}")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Pretty-print final table.
    trackers = sorted({r["tracker"] for r in rows})
    by_tracker_attr = {(r["tracker"], r["attribute"]): r for r in rows}
    print("\nFinal unique-attribute table (per-sequence SR / NPR / PR):")
    header = "tracker".ljust(10) + " | " + " | ".join(
        f"{label:>21s}" for label, _, _ in UNIQUE_ATTRS
    )
    print(header)
    print("-" * len(header))
    for tr in trackers:
        cells = []
        for label, _, _ in UNIQUE_ATTRS:
            r = by_tracker_attr.get((tr, label))
            if r is None:
                cells.append("---".center(21))
            else:
                cells.append(f"{r['SR']:.3f}/{r['NPR']:.3f}/{r['PR']:.3f}".rjust(21))
        print(tr.ljust(10) + " | " + " | ".join(cells))


if __name__ == "__main__":
    main()
