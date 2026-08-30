"""
Re-aggregate whole-dataset SOT runs onto the released Space-Tracker sequence set.

The trackers were all evaluated on the *full* source datasets (OOTB 110 +
SatSOT 105 + SV248S 248 = 463 sequences). The published benchmark keeps 395 of
those. Since every tracker is zero-shot and the frames are unchanged, the
released numbers can be recomputed from the per-frame dumps without re-running
anything on a GPU: we filter each run's records down to the released sequences
and re-aggregate.

Aggregation reuses `lightning_modules.sot_metrics._compute_from_sequences`, the
same per-sequence code path the eval callback uses, so the numbers are
identical in construction to the ones the pipeline reports.

Beyond the primary metrics this also reports, per tracker:

* **abstention rate** — the fraction of frames on which the tracker emitted no
  box. Our protocol records a lost target as an empty prediction for the whole
  remainder of the video, which scores as IoU 0. Trackers that never declare
  loss (SiamFC, OSTrack) are structurally advantaged over ones that do (LoRAT,
  SAM 3), so this belongs in the table as its own column rather than buried in
  the headline metric.
* **conditional metrics** — the same metrics computed only over frames where
  the tracker did answer, which separates "how well does it track when it is
  tracking" from "how often does it give up".

Usage:
    python tools/aggregate_sot_release.py \
        --runs /path/to/SOT_whole_dataset_<date> \
        --release /path/to/release/space_tracker \
        --out docs/space_tracker/sot_release_metrics
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning_modules.sot_metrics import SOTRecord, _compute_from_sequences


# Metrics carried into the output table, in display order.
PRIMARY_KEYS = [
    ("success_auc", "SR"),
    ("norm_precision_auc", "NPR"),
    ("precision_auc", "PR"),
    ("precision_5", "P@5"),
    ("mean_iou", "mIoU"),
]


def load_released_sequences(release_root: Path) -> dict[str, set[str]]:
    """
    Map source dataset → set of source video ids that survived into the release.

    `source_sequence_id` is `<dataset>/<video_id>` and `video_id` is exactly the
    key the eval dumps use, so the join needs no normalisation.
    """
    ann = release_root / "sot" / "annotations" / "space_tracker_sot.json"
    data = json.loads(ann.read_text())
    kept: dict[str, set[str]] = defaultdict(set)
    for v in data["videos"]:
        ds = v["source_dataset"]
        sid = v["source_sequence_id"]
        prefix = f"{ds}/"
        kept[ds].add(sid[len(prefix):] if sid.startswith(prefix) else sid)
    return dict(kept)


def parse_run_dir(run_dir: Path) -> tuple[str, str] | None:
    """Return (tracker, dataset) for a run directory, or None if unparseable."""
    name = run_dir.name
    if "_first_frame_" not in name:
        return None
    dataset = name.split("_first_frame_")[1].rsplit("_", 2)[0]
    return run_dir.parent.name, dataset.lower()


def _aabb_rescore(rec: dict) -> tuple[float, float, float]:
    """
    Recompute (iou, center_dist, norm_center_dist) under the OOTB AABB protocol.

    `per_image_metrics.json` is written by the visualization callback, which
    hard-codes polygon IoU for OBB datasets and never sees `obb_eval_mode`. For
    HBB trackers on OOTB the official path instead collapses BOTH the GT
    polygon and the prediction to their AABBs and scores plain AABB IoU
    (`sot_metrics.py`, the `use_ootb_aabb` branch). We reproduce that here from
    the stored polygons.
    """
    import numpy as np
    from obb_utils import obb_to_aabb

    gt_poly = rec.get("gt_poly")
    pred_poly = rec.get("pred_poly")
    if gt_poly is None:
        return 0.0, float("inf"), float("inf")
    gt_a = obb_to_aabb(np.asarray(gt_poly, dtype=np.float32))
    if pred_poly is None:
        return 0.0, float("inf"), float("inf")
    pr_a = obb_to_aabb(np.asarray(pred_poly, dtype=np.float32))

    ix1, iy1 = max(gt_a[0], pr_a[0]), max(gt_a[1], pr_a[1])
    ix2, iy2 = min(gt_a[2], pr_a[2]), min(gt_a[3], pr_a[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ga = max(0.0, gt_a[2] - gt_a[0]) * max(0.0, gt_a[3] - gt_a[1])
    pa = max(0.0, pr_a[2] - pr_a[0]) * max(0.0, pr_a[3] - pr_a[1])
    union = ga + pa - inter
    iou = float(inter / union) if union > 0 else 0.0

    gt_cx, gt_cy = (gt_a[0] + gt_a[2]) / 2, (gt_a[1] + gt_a[3]) / 2
    pr_cx, pr_cy = (pr_a[0] + pr_a[2]) / 2, (pr_a[1] + pr_a[3]) / 2
    cdist = float(((gt_cx - pr_cx) ** 2 + (gt_cy - pr_cy) ** 2) ** 0.5)

    # Normalisation uses the axis-aligned GT box, as in sot_metrics.py. Note the
    # stored gt_box is int-rounded by the viz callback, so NPR can differ from
    # the pipeline in the last decimal; SR and PR are unaffected.
    gt_box = rec.get("gt_box")
    if gt_box:
        gw, gh = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
        diag = float((gw ** 2 + gh ** 2) ** 0.5)
    else:
        diag = 1e-6
    return iou, cdist, cdist / max(diag, 1e-6)


def records_by_sequence(
    per_image_path: Path, keep: set[str] | None, rescore_aabb: bool = False
) -> tuple[dict[str, list[SOTRecord]], dict[str, list[bool]]]:
    """
    Read a per_image_metrics.json into per-sequence records plus an
    "did the tracker answer on this frame" flag per record.

    When `rescore_aabb` is set (HBB tracker on OOTB), overlap and centre
    distance are recomputed under the AABB protocol instead of trusting the
    polygon-IoU values the viz callback stored.
    """
    data = json.loads(per_image_path.read_text())
    seqs: dict[str, list[SOTRecord]] = defaultdict(list)
    answered: dict[str, list[bool]] = defaultdict(list)

    for frame in data:
        vid = frame["video_id"]
        if keep is not None and vid not in keep:
            continue
        for rec in frame.get("sot_records", []):
            if rescore_aabb:
                iou, cdist, ncdist = _aabb_rescore(rec)
            else:
                iou = float(rec.get("best_iou", 0.0))
                cdist = float(rec.get("center_dist", 0.0))
                ncdist = float(rec.get("norm_center_dist", 0.0))
            seqs[vid].append(
                SOTRecord(
                    video_id=vid,
                    frame_id=frame["frame_id"],
                    gt_class=rec.get("gt_class", ""),
                    gt_size=rec.get("gt_size", ""),
                    best_iou=iou,
                    center_dist=cdist,
                    norm_center_dist=ncdist,
                )
            )
            box = rec.get("pred_box")
            answered[vid].append(
                box is not None and not (isinstance(box, (list, tuple)) and len(box) == 0)
            )
    return dict(seqs), dict(answered)


def obb_eval_mode_for(tracker: str, dataset: str, configs_dir: Path) -> str:
    """
    Read the tracker's own eval config to learn which OOTB branch it used.

    The split is not cosmetic: mask-producing trackers are scored under polygon
    IoU, HBB trackers against the GT's enclosing AABB. Guessing from the tracker
    name would silently mis-score any future tracker, so we read the config.
    """
    cfg_path = configs_dir / f"{tracker}_{dataset}.yaml"
    if not cfg_path.exists():
        return "polygon"
    for line in cfg_path.read_text().splitlines():
        if line.strip().startswith("obb_eval_mode:"):
            return line.split(":", 1)[1].split("#")[0].strip()
    return "polygon"


def abstention_rate(answered: dict[str, list[bool]]) -> float:
    """Per-sequence abstention rate, averaged across sequences."""
    per_seq = [
        1.0 - (sum(flags) / len(flags))
        for flags in answered.values()
        if flags
    ]
    return float(sum(per_seq) / len(per_seq)) if per_seq else 0.0


def conditional_metrics(
    seqs: dict[str, list[SOTRecord]], answered: dict[str, list[bool]]
) -> dict:
    """Metrics over answered frames only. Sequences with no answer are dropped."""
    kept = []
    for vid, recs in seqs.items():
        flags = answered.get(vid, [])
        sub = [r for r, ok in zip(recs, flags) if ok]
        if sub:
            kept.append(sub)
    return _compute_from_sequences(kept) if kept else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True,
                    help="Directory holding <tracker>/<run>/per_image_metrics.json")
    ap.add_argument("--release", required=True,
                    help="Built Space-Tracker release root")
    ap.add_argument("--out", required=True, help="Output path prefix (no extension)")
    ap.add_argument("--configs", default="configs/SOT",
                    help="Directory holding <tracker>_<dataset>.yaml eval configs")
    ap.add_argument("--force-obb-mode", choices=["config", "polygon", "ootb_aabb"],
                    default="config",
                    help="Override the per-tracker OOTB scoring branch. 'config' (default) "
                         "honours each tracker's obb_eval_mode; 'polygon' reproduces the "
                         "NeurIPS tables, which scored every tracker with polygon IoU.")
    ap.add_argument("--no-filter", action="store_true",
                    help="Skip release filtering (reproduce the original whole-dataset numbers)")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    kept = load_released_sequences(Path(args.release))
    print("Released sequences per source dataset: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(kept.items())))

    rows = []
    for run_dir in sorted(runs_root.glob("*/*/")):
        pim = run_dir / "per_image_metrics.json"
        if not pim.exists():
            continue
        parsed = parse_run_dir(run_dir)
        if parsed is None:
            continue
        tracker, dataset = parsed
        keep = None if args.no_filter else kept.get(dataset)
        if keep is None and not args.no_filter:
            print(f"  [skip] {tracker}/{dataset}: dataset not in release")
            continue

        mode = obb_eval_mode_for(tracker, dataset, Path(args.configs))
        if args.force_obb_mode != "config":
            mode = args.force_obb_mode if dataset == "ootb" else "polygon"
        seqs, answered = records_by_sequence(
            pim, keep, rescore_aabb=(mode == "ootb_aabb")
        )
        if not seqs:
            print(f"  [skip] {tracker}/{dataset}: no records survived the filter")
            continue

        primary = _compute_from_sequences(list(seqs.values()))
        cond = conditional_metrics(seqs, answered)
        row = {
            "tracker": tracker,
            "dataset": dataset,
            "n_sequences": primary.get("n_sequences", 0),
            "n_frames": primary.get("n_frames", 0),
            "abstention": round(abstention_rate(answered), 4),
        }
        for key, label in PRIMARY_KEYS:
            row[label] = primary.get(key)
            row[f"{label}_answered"] = cond.get(key)
        row["obb_eval_mode"] = mode
        rows.append(row)
        print(f"  {tracker:12s} {dataset:8s} "
              f"{row['n_sequences']:4d} seq  SR={row['SR']:.4f}  "
              f"abstain={row['abstention']*100:.1f}%"
              + ("  [aabb-rescored]" if mode == "ootb_aabb" else ""))

    if not rows:
        print("No runs aggregated.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import csv
    fieldnames = list(rows[0].keys())
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(out.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote {out.with_suffix('.csv')} and {out.with_suffix('.json')} "
          f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
