"""
Confidence-threshold analysis for SOT, from the tau=0 dumps.

The shipped runs applied tau=0.5 BEFORE writing per-frame records
(`lightning_modules/visualization.py`), so the discarded predictions -- and the
scores that would justify discarding them -- were never stored. That made three
reviewer points unanswerable:

  1. is tau=0.5 calibrated the same way across tracker families?
  2. what does "no valid box" actually mean per tracker?
  3. how much of a tracker's abstention is it declaring loss, versus us
     filtering it out?

Re-running at tau=0 keeps every prediction with its score, so all three become
offline questions. This tool answers them:

* **sweep**   -- SR/NPR/PR/P@5 as a function of tau, per tracker.
* **split**   -- abstention decomposed into `declared` (tracker emitted nothing
                 at all) versus `filtered` (tracker emitted a box that a given
                 tau removes). At tau=0 only `declared` can be non-zero, which
                 is exactly the unified "no valid box" definition.
* **scores**  -- the score distribution each tracker actually produces, which is
                 what makes a single global tau incomparable in the first place.

Usage:
    python tools/analyse_sot_tau.py --runs /path/to/SOT_tau0 \
        --release /path/to/release/space_tracker --out docs/space_tracker/sot_tau_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning_modules.sot_metrics import SOTRecord, _compute_from_sequences
from tools.aggregate_sot_release import _aabb_rescore, obb_eval_mode_for
from tools.make_wacv_sot_table import released_sequences

DATASETS = ["ootb", "satsot", "sv248s"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def find_run(runs_root: Path, tracker: str, dataset: str) -> Path | None:
    cands = sorted(runs_root.glob(f"{tracker}*_first_frame_{dataset}_*/per_image_metrics.json"))
    return cands[-1] if cands else None


def load_records(pim: Path, keep: set[str], rescore_aabb: bool):
    """Per-sequence list of (record, score, has_box), OOTB rescored to HBB."""
    data = json.loads(pim.read_text())
    seqs: dict[str, list] = defaultdict(list)
    for frame in data:
        vid = frame["video_id"]
        if vid not in keep:
            continue
        for rec in frame.get("sot_records", []):
            box = rec.get("pred_box")
            has_box = box is not None and not (isinstance(box, (list, tuple)) and len(box) == 0)
            if rescore_aabb:
                iou, cdist, ncdist = _aabb_rescore(rec)
            else:
                iou = float(rec.get("best_iou", 0.0))
                cdist = float(rec.get("center_dist", 0.0))
                ncdist = float(rec.get("norm_center_dist", 0.0))
            score = rec.get("pred_score")
            seqs[vid].append((
                SOTRecord(video_id=vid, frame_id=frame["frame_id"],
                          gt_class=rec.get("gt_class", ""), gt_size=rec.get("gt_size", ""),
                          best_iou=iou, center_dist=cdist, norm_center_dist=ncdist),
                float(score) if score is not None else None,
                has_box,
            ))
    return dict(seqs)


def apply_tau(seqs: dict[str, list], tau: float):
    """Re-score with predictions below tau treated as misses (IoU 0, inf CLE)."""
    out = []
    for recs in seqs.values():
        kept = []
        for r, score, has_box in recs:
            drop = (not has_box) or (score is not None and score < tau)
            kept.append(r if not drop else SOTRecord(
                video_id=r.video_id, frame_id=r.frame_id, gt_class=r.gt_class,
                gt_size=r.gt_size, best_iou=0.0,
                center_dist=float("inf"), norm_center_dist=float("inf")))
        if kept:
            out.append(kept)
    return _compute_from_sequences(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="SOT_tau0 experiment root")
    ap.add_argument("--release", required=True)
    ap.add_argument("--configs", default="configs/SOT")
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    released = released_sequences(Path(args.release))

    trackers = sorted({p.parent.name.split("_first_frame_")[0].rsplit("_", 0)[0]
                       for p in runs_root.glob("*/per_image_metrics.json")})
    # map run-dir prefix back to the short tracker key used by the configs
    keys = {}
    for p in runs_root.glob("*/per_image_metrics.json"):
        stem = p.parent.name.split("_first_frame_")[0]
        keys.setdefault(stem.split("_")[0], stem)
    print(f"trackers: {sorted(keys)}")

    sweep_rows, split_rows, score_rows = [], [], []
    for key in sorted(keys):
        pooled: dict[str, list] = {}
        for ds in DATASETS:
            pim = find_run(runs_root, key, ds)
            if pim is None:
                print(f"  [{key}] missing {ds}, skipped")
                continue
            mode = obb_eval_mode_for(key, ds, Path(args.configs))
            seqs = load_records(pim, set(released[ds]),
                                rescore_aabb=(ds == "ootb"))
            for vid, recs in seqs.items():
                pooled[f"{ds}/{vid}"] = recs
        if not pooled:
            continue

        flat = [x for recs in pooled.values() for x in recs]
        scores = np.array([s for _, s, has in flat if has and s is not None], dtype=np.float64)
        n_all = len(flat)
        declared = sum(1 for _, _, has in flat if not has)

        score_rows.append({
            "tracker": key, "n_frames": n_all,
            "declared_no_box": declared,
            "declared_pct": round(100 * declared / max(n_all, 1), 2),
            "score_min": round(float(scores.min()), 4) if len(scores) else None,
            "score_p05": round(float(np.percentile(scores, 5)), 4) if len(scores) else None,
            "score_p50": round(float(np.percentile(scores, 50)), 4) if len(scores) else None,
            "score_p95": round(float(np.percentile(scores, 95)), 4) if len(scores) else None,
            "score_max": round(float(scores.max()), 4) if len(scores) else None,
            "constant_score": bool(len(scores) and scores.min() == scores.max()),
        })

        for tau in TAUS:
            m = apply_tau(pooled, tau)
            filtered = sum(1 for _, s, has in flat
                           if has and s is not None and s < tau)
            sweep_rows.append({
                "tracker": key, "tau": tau,
                "SR": m.get("success_auc"), "NPR": m.get("norm_precision_auc"),
                "PR": m.get("precision_auc"), "P@5": m.get("precision_5"),
            })
            split_rows.append({
                "tracker": key, "tau": tau,
                "declared_pct": round(100 * declared / max(n_all, 1), 2),
                "filtered_pct": round(100 * filtered / max(n_all, 1), 2),
                "total_missing_pct": round(100 * (declared + filtered) / max(n_all, 1), 2),
            })
        print(f"  [{key}] {n_all} frames, declared-no-box {100*declared/max(n_all,1):.1f}%, "
              f"score p05={score_rows[-1]['score_p05']} p95={score_rows[-1]['score_p95']}"
              + ("  [CONSTANT SCORE]" if score_rows[-1]["constant_score"] else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    for name, rows in [("sweep", sweep_rows), ("abstention_split", split_rows),
                       ("score_dist", score_rows)]:
        p = out.parent / f"{out.name}_{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
