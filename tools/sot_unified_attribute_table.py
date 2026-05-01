"""
Per-attribute SOT performance under our unified attribute taxonomy.

Unified attributes (see ``Formatting Instructions For NeurIPS 2026/tables/SOT/
whole_dataset/split_attributes_table.tex``) span SatSOT / SV248S / OOTB and
sometimes consolidate several native dataset labels into one (e.g. OCC unifies
SatSOT POC/FOC, SV248S STO/LTO/CO, OOTB PO/FO).

Pipeline:
  1. Load ``per_image_metrics.json`` from each tracker run.
  2. For each (tracker, dataset, unified_attr):
       - Pick sequences that carry at least one native label mapped to
         ``unified_attr`` for this dataset.
       - Per-sequence SR / NPR / PR (PR over CLE in [0, 50] px) — same
         protocol as ``tools/reaggregate_sot_per_sequence.py``.
       - Mean across the selected sequences.
  3. For each (tracker, unified_attr): arithmetic mean across the datasets
     that annotate it (BC: 2, IV: 3, ROT: 3, OCC: 3, SOB: 3, DEF: 2).

Outputs (under ``--out``):
  - ``per_dataset_attr.csv`` — intermediate (tracker, dataset, attr, n_seq, SR, NPR, PR)
  - ``unified_attr.csv``     — final (tracker, attr, n_datasets, SR, NPR, PR)

Usage::

    micromamba run -n esa_dlstem python tools/sot_unified_attribute_table.py
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


# --- Curve thresholds (per-sequence aggregation, PR over [0, 50]) ---
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=float)
NORM_PRECISION_THRESHOLDS = np.linspace(0.0, 0.5, 21)


# --- Unified attribute taxonomy: dataset → list[native attr] ---
# Empty list ⇒ this dataset does not annotate this unified attribute.
UNIFIED_ATTR_MAP: dict[str, dict[str, list[str]]] = {
    "BC":  {"satsot": ["BC"],            "sv248s": [],                 "ootb": ["BC"]},
    "IV":  {"satsot": ["IV"],            "sv248s": ["IV"],             "ootb": ["IV"]},
    "ROT": {"satsot": ["ROT"],           "sv248s": ["IPR"],            "ootb": ["IPR"]},
    "OCC": {"satsot": ["POC", "FOC"],    "sv248s": ["STO", "LTO", "CO"], "ootb": ["PO", "FO"]},
    "SOB": {"satsot": ["SOB"],           "sv248s": ["DS"],             "ootb": ["SA"]},
    "DEF": {"satsot": ["DEF"],           "sv248s": [],                 "ootb": ["DEF"]},
}
ATTR_ORDER = ["BC", "IV", "ROT", "OCC", "SOB", "DEF"]
DATASET_ORDER = ["satsot", "sv248s", "ootb"]

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
# Per-sequence metrics
# ============================================================

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


def _avg_seq_summaries(seqs: list[dict]) -> dict:
    if not seqs:
        return {"SR": np.nan, "NPR": np.nan, "PR": np.nan, "n_seq": 0, "n_frames": 0}
    return {
        "SR":       float(np.mean([s["success_auc"]        for s in seqs])),
        "NPR":      float(np.mean([s["norm_precision_auc"] for s in seqs])),
        "PR":       float(np.mean([s["precision_auc"]      for s in seqs])),
        "n_seq":    len(seqs),
        "n_frames": int(sum(s["n_frames"] for s in seqs)),
    }


# ============================================================
# Sequence attribute lookup
# ============================================================

def load_sequence_attributes() -> dict[str, dict[str, list[str]]]:
    """{dataset: {video_id: [native_attrs]}} for the whole (no_split) dataset."""
    out: dict[str, dict[str, list[str]]] = {}
    out["ootb"]   = OOTBDataset(  root=DATASET_ROOTS["ootb"],   split="no_split", mode="detection").sequence_attributes()
    out["satsot"] = SatSOTDataset(root=DATASET_ROOTS["satsot"], split="no_split", mode="detection").sequence_attributes()
    out["sv248s"] = SV248SDataset(root=DATASET_ROOTS["sv248s"], split="no_split", mode="detection").sequence_attributes()
    return out


def videos_with_unified_attr(
    dataset: str,
    unified_attr: str,
    seq_attrs: dict[str, list[str]],
) -> set[str]:
    """Set of video_ids in ``dataset`` whose native attrs contain ANY of the
    native labels mapped to ``unified_attr`` (union semantics, e.g. OCC =
    POC ∪ FOC for SatSOT)."""
    natives = set(UNIFIED_ATTR_MAP[unified_attr][dataset])
    if not natives:
        return set()
    return {vid for vid, attrs in seq_attrs.items() if natives & set(attrs)}


# ============================================================
# Run discovery
# ============================================================

def detect_dataset(run_name: str) -> str | None:
    for ds, pat in DATASET_TAGS.items():
        if pat.search(run_name):
            return ds
    return None


def discover_runs(root: Path) -> list[tuple[str, str, Path]]:
    runs = []
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
            runs.append((tracker_dir.name, ds, run_dir))
    return runs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Defaults to <root>/analysis/unified_attributes",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / "analysis" / "unified_attributes")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading sequence attributes for SatSOT / SV248S / OOTB")
    seq_attrs = load_sequence_attributes()
    for ds, attrs in seq_attrs.items():
        print(f"   {ds:7s}: {len(attrs)} sequences with attribute annotations")

    # Pre-compute video sets per (dataset, unified_attr).
    print("[2/4] resolving unified-attribute → video_id sets")
    attr_video_sets: dict[tuple[str, str], set[str]] = {}
    for attr in ATTR_ORDER:
        for ds in DATASET_ORDER:
            vids = videos_with_unified_attr(ds, attr, seq_attrs[ds])
            attr_video_sets[(ds, attr)] = vids
            if UNIFIED_ATTR_MAP[attr][ds]:
                print(f"   {attr:3s} on {ds:7s}: {len(vids)} sequences "
                      f"(natives = {UNIFIED_ATTR_MAP[attr][ds]})")

    # Iterate runs.
    print("[3/4] aggregating per-tracker metrics")
    runs = discover_runs(root)
    # tracker → dataset → attr → {SR, NPR, PR, n_seq, n_frames}
    per_dataset_rows: list[dict] = []
    # tracker → attr → list of {SR, NPR, PR} from each annotating dataset
    per_attr_dataset_metrics: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for tracker, ds, run_dir in runs:
        with open(run_dir / "per_image_metrics.json") as f:
            frames = json.load(f)

        # Group records by video_id.
        by_seq_records: dict[str, list[dict]] = defaultdict(list)
        for fr in frames:
            vid = str(fr["video_id"])
            by_seq_records[vid].extend(fr.get("sot_records", []))

        # Per-sequence summaries.
        seq_summary: dict[str, dict] = {
            vid: _seq_summary(recs) for vid, recs in by_seq_records.items() if recs
        }

        for attr in ATTR_ORDER:
            vids = attr_video_sets[(ds, attr)]
            if not vids:
                continue   # attribute not annotated in this dataset
            picked = [seq_summary[v] for v in vids if v in seq_summary]
            agg = _avg_seq_summaries(picked)
            per_dataset_rows.append({
                "tracker":   tracker,
                "dataset":   ds,
                "attribute": attr,
                "natives":   "|".join(UNIFIED_ATTR_MAP[attr][ds]),
                "n_seq":     agg["n_seq"],
                "n_frames":  agg["n_frames"],
                "SR":        round(agg["SR"], 4)  if agg["n_seq"] else None,
                "NPR":       round(agg["NPR"], 4) if agg["n_seq"] else None,
                "PR":        round(agg["PR"], 4)  if agg["n_seq"] else None,
            })
            if agg["n_seq"]:
                per_attr_dataset_metrics[tracker][attr].append({
                    "dataset": ds,
                    "SR":  agg["SR"],  "NPR": agg["NPR"], "PR":  agg["PR"],
                })

    # Final unified per-attribute scores: mean across annotating datasets.
    print("[4/4] averaging across annotating datasets")
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
                "SR":          round(float(np.mean([m["SR"]  for m in ds_metrics])), 4),
                "NPR":         round(float(np.mean([m["NPR"] for m in ds_metrics])), 4),
                "PR":          round(float(np.mean([m["PR"]  for m in ds_metrics])), 4),
            })

    # Write CSVs.
    per_dataset_rows.sort(key=lambda r: (r["tracker"], r["attribute"], r["dataset"]))
    unified_rows.sort(key=lambda r: (r["tracker"], r["attribute"]))

    with open(out_dir / "per_dataset_attr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_dataset_rows[0].keys()))
        w.writeheader()
        w.writerows(per_dataset_rows)

    with open(out_dir / "unified_attr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unified_rows[0].keys()))
        w.writeheader()
        w.writerows(unified_rows)

    print(f"\nDone. Outputs under: {out_dir}")
    print(f"  per_dataset_attr.csv  ({len(per_dataset_rows)} rows)")
    print(f"  unified_attr.csv      ({len(unified_rows)} rows)")

    # Pretty-print final table.
    trackers = sorted({r["tracker"] for r in unified_rows})
    print("\nFinal unified-attribute table (per-sequence SR / NPR / PR, "
          "averaged across annotating datasets):")
    header = "tracker".ljust(10) + " | " + " | ".join(
        f"{a:>20s}" for a in ATTR_ORDER
    )
    print(header)
    print("-" * len(header))
    by_tracker_attr = {(r["tracker"], r["attribute"]): r for r in unified_rows}
    for tr in trackers:
        cells = []
        for a in ATTR_ORDER:
            r = by_tracker_attr.get((tr, a))
            if r is None:
                cells.append("---".center(20))
            else:
                cells.append(f"{r['SR']:.3f}/{r['NPR']:.3f}/{r['PR']:.3f}".rjust(20))
        print(tr.ljust(10) + " | " + " | ".join(cells))


if __name__ == "__main__":
    main()
