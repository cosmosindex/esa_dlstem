"""
Per-sequence bbox-size analysis for the Space-tracker SOT + MOT manifests.

Answers the question: *can every dataset be split into a "small" folder
(boxes <= 32 x 32) and a "large" folder (everything else)?*

For SOT this is nearly trivial — one annotated target per frame, so a
sequence has a single size regime. For MOT a sequence may contain a mix of
sizes, so the script reports, per sequence:

  * ``n_boxes``           — annotated GT boxes in the sequence
  * ``frac_small``        — fraction of boxes with sqrt(area) <= 32 px
  * ``purity``            — ``pure_small`` / ``pure_large`` / ``mixed``
  * percentile spread of sqrt(area)

Size criterion (COCO convention): a box is *small* when
``sqrt(w * h) <= 32`` px, i.e. area <= 1024 px^2. The alternative
side-wise criterion (``w <= 32 and h <= 32``) is reported alongside so the
two definitions can be compared before committing to one.

Usage::

    python tools/analyze_size_split.py --out-dir /tmp/size_split
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data import iter_frames
from space_tracker.data_mot import _parse_gt
from space_tracker.manifest import Manifest
from space_tracker.manifest_mot import MOTManifest

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/data/ESA_DLSTEM_2025/data/trafic")

SOT_ROOTS = {
    "ootb":   DATA / "OOTB",
    "satsot": DATA / "SatSOT",
    "sv248s": DATA / "SV248S",
}
MOT_ROOTS = {
    "airmot":    DATA / "AIR-MOT-100",
    "satmtb":    DATA / "SAT-MTB",
    "viso":      DATA / "VISO",
    "sdmcar":    DATA / "SDM-Car",
    "rscardata": DATA / "RsCarData",
}

SMALL_THRESH = 32.0     # px — sqrt(area) <= 32  <=>  area <= 32*32


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _size_stats(wh: np.ndarray) -> dict:
    """``wh`` is (N, 2) of box widths / heights in px."""
    if wh.size == 0:
        return {}
    w, h = wh[:, 0], wh[:, 1]
    sqrt_area = np.sqrt(np.maximum(w, 0) * np.maximum(h, 0))
    small_area = sqrt_area <= SMALL_THRESH                  # COCO-style
    small_side = (w <= SMALL_THRESH) & (h <= SMALL_THRESH)  # side-wise
    return {
        "n_boxes":       int(wh.shape[0]),
        "sqrt_area_min":    round(float(sqrt_area.min()), 2),
        "sqrt_area_p05":    round(float(np.percentile(sqrt_area, 5)), 2),
        "sqrt_area_median": round(float(np.median(sqrt_area)), 2),
        "sqrt_area_p95":    round(float(np.percentile(sqrt_area, 95)), 2),
        "sqrt_area_max":    round(float(sqrt_area.max()), 2),
        "frac_small_area":  round(float(small_area.mean()), 4),
        "frac_small_side":  round(float(small_side.mean()), 4),
    }


def _purity(frac: float) -> str:
    if frac >= 1.0:
        return "pure_small"
    if frac <= 0.0:
        return "pure_large"
    return "mixed"


# ---------------------------------------------------------------------------
# SOT
# ---------------------------------------------------------------------------

def analyse_sot(manifest_path: Path) -> list[dict]:
    man = Manifest.load(manifest_path)
    rows: list[dict] = []
    for i, seq in enumerate(man.sequences, 1):
        wh = []
        for fr in iter_frames(seq, SOT_ROOTS):
            if not fr.visible or fr.gt_box_xyxy is None:
                continue
            x1, y1, x2, y2 = fr.gt_box_xyxy
            wh.append((x2 - x1, y2 - y1))
        stats = _size_stats(np.asarray(wh, dtype=np.float64).reshape(-1, 2))
        if not stats:
            print(f"  [warn] {seq.id}: no visible GT boxes")
            continue
        rows.append({
            "task": "SOT", "dataset": seq.dataset, "seq_id": seq.id,
            "category": seq.category, "n_frames": seq.n_frames,
            "purity": _purity(stats["frac_small_area"]), **stats,
        })
        if i % 50 == 0:
            print(f"  SOT {i}/{len(man.sequences)}")
    return rows


# ---------------------------------------------------------------------------
# MOT
# ---------------------------------------------------------------------------

def analyse_mot(manifest_path: Path) -> list[dict]:
    man = MOTManifest.load(manifest_path)
    rows: list[dict] = []
    for i, seq in enumerate(man.sequences, 1):
        root = MOT_ROOTS[seq.dataset]
        try:
            ann = _parse_gt(seq, root)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [warn] {seq.id}: GT parse failed — {exc}")
            continue

        wh, per_cat = [], defaultdict(list)
        for objs in ann.values():
            for o in objs:
                x1, y1, x2, y2 = o.bbox_xyxy
                wh.append((x2 - x1, y2 - y1))
                per_cat[o.category].append((x2 - x1, y2 - y1))

        stats = _size_stats(np.asarray(wh, dtype=np.float64).reshape(-1, 2))
        if not stats:
            print(f"  [warn] {seq.id}: no GT boxes")
            continue

        # Which classes inside this sequence are small / large?
        cat_flags = {}
        for cat, cwh in per_cat.items():
            cs = _size_stats(np.asarray(cwh, dtype=np.float64).reshape(-1, 2))
            cat_flags[cat] = cs["frac_small_area"]

        rows.append({
            "task": "MOT", "dataset": seq.dataset, "seq_id": seq.id,
            "category": seq.category, "n_frames": seq.n_frames,
            "purity": _purity(stats["frac_small_area"]), **stats,
            "cats_in_seq": "|".join(sorted(per_cat)),
            "frac_small_per_cat": "|".join(
                f"{c}:{cat_flags[c]:.3f}" for c in sorted(cat_flags)),
        })
        if i % 50 == 0:
            print(f"  MOT {i}/{len(man.sequences)}")
    return rows


# ---------------------------------------------------------------------------

def summarise(rows: list[dict]) -> None:
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[(r["task"], r["dataset"])].append(r)

    print(f"\n{'task/dataset':<18} {'seqs':>5} {'pure_small':>11} "
          f"{'pure_large':>11} {'mixed':>7}  {'any-small→small':>16}")
    print("-" * 78)
    for (task, ds), rs in sorted(by_ds.items()):
        pure_s = sum(1 for r in rs if r["purity"] == "pure_small")
        pure_l = sum(1 for r in rs if r["purity"] == "pure_large")
        mixed  = sum(1 for r in rs if r["purity"] == "mixed")
        any_s  = sum(1 for r in rs if r["frac_small_area"] > 0)
        print(f"{task}/{ds:<13} {len(rs):>5} {pure_s:>11} {pure_l:>11} "
              f"{mixed:>7}  {any_s:>7} ({any_s / len(rs):.0%})")

    print("\nMixed sequences — how much of the sequence is actually small:")
    print(f"{'dataset':<12} {'seq_id':<28} {'n_boxes':>8} {'frac_small':>11} "
          f"{'p05':>7} {'median':>7} {'p95':>8} {'max':>8}")
    print("-" * 92)
    mixed_rows = [r for r in rows if r["purity"] == "mixed"]
    for r in sorted(mixed_rows, key=lambda r: r["frac_small_area"])[:40]:
        print(f"{r['dataset']:<12} {r['seq_id']:<28} {r['n_boxes']:>8} "
              f"{r['frac_small_area']:>11.3f} {r['sqrt_area_p05']:>7.1f} "
              f"{r['sqrt_area_median']:>7.1f} {r['sqrt_area_p95']:>8.1f} "
              f"{r['sqrt_area_max']:>8.1f}")
    if len(mixed_rows) > 40:
        print(f"  ... {len(mixed_rows) - 40} more mixed sequences")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", choices=["sot", "mot", "both"], default="both")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.task in ("sot", "both"):
        print("== SOT ==")
        rows += analyse_sot(REPO / "space_tracker" / "space_tracker.json")
    if args.task in ("mot", "both"):
        print("== MOT ==")
        rows += analyse_mot(REPO / "space_tracker" / "space_tracker_mot.json")

    fields = sorted({k for r in rows for k in r})
    order = ["task", "dataset", "seq_id", "category", "n_frames", "n_boxes",
             "purity", "frac_small_area", "frac_small_side"]
    fields = order + [f for f in fields if f not in order]
    out_csv = args.out_dir / "per_sequence_size.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} sequences)")

    summarise(rows)


if __name__ == "__main__":
    main()
