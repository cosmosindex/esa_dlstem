"""Per-class HOTA computation for the multi-class (FasterRCNN-driven) MOT runs.

Counterpart to ``compute_hota.py`` for the non-car pipeline. Takes a
tracker-output root containing ``<tracker>_<dataset>_<TS>/`` dirs whose
``mot_format/`` is *itself* class-segregated (``mot_format/<class>/<seq>.txt``)
and writes one row per (dataset, tracker, class) into a CSV plus a
macro-aggregated row per (dataset, tracker) and a global per-tracker
row averaged across all (dataset, class).

Datasets supported: ``satmtb_nocar`` ``viso_nocar`` ``airmot``.

Usage::

    python compute_hota_multiclass.py \
        --tracker-output-root /data/.../experiments/MOT/tracker_satmtb_hbb_<TS> \
        --workspace /tmp/hota_satmtb_hbb \
        --output /data/.../experiments/MOT/tracker_satmtb_hbb_<TS>/hota_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.airmot import AIRMOTDataset
from datasets.satmtb import SATMTBDataset
from datasets.viso import VISODataset


# Dataset (cls, root, build_kwargs, split, class_map name → id, class_id used in GT files)
_DATASET_TABLE = {
    "satmtb_nocar": (
        SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
        {"task": "mot", "categories": ["airplane", "ship", "train"]},
        "test",
        {"airplane": 1, "ship": 2, "train": 3},
    ),
    "viso_nocar": (
        VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
        {"categories": ["plane", "ship", "train"]},
        "no_split",
        {"plane": 1, "ship": 2, "train": 3},
    ),
    "airmot": (
        AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100",
        {},
        "no_split",
        {"airplane": 1, "ship": 2},
    ),
}


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str):
    cls, root, extra, split, cmap = _DATASET_TABLE[name]
    if cls is VISODataset or cls is AIRMOTDataset:
        return cls(root=root, split=split, class_map=dict(cmap), **extra)
    return cls(root=root, split=split, mode="detection",
               class_map=dict(cmap), **extra)


def _write_gt_for_class(
    dataset_name: str, class_name: str, gt_root: Path,
) -> tuple[list[str], dict[str, int]]:
    """Dump GT for one (dataset, class) into TrackEval layout.

    Sequences with zero GT for the target class are skipped (TrackEval
    can't evaluate empty-GT seqs).
    """
    ds = _build_dataset(dataset_name)
    cmap = _DATASET_TABLE[dataset_name][4]
    target_id = cmap[class_name]

    seq_names: list[str] = []
    seq_offsets: dict[str, int] = {}
    for v in ds.videos:
        first_fid = min(v.frame_ids)
        offset = 1 - int(first_fid) if first_fid < 1 else 0

        # Pre-scan to skip empty-GT seqs.
        any_box = False
        prepared: list[str] = []
        seen_in_frame: dict[int, set] = {}
        for fid in v.frame_ids:
            ann = ds._load_annotations(v, fid)
            boxes = ann["boxes"]; tids = ann["track_ids"]; labels = ann["labels"]
            for j in range(len(boxes)):
                if int(labels[j]) != target_id:
                    continue
                tid = int(tids[j])
                if tid < 0:
                    continue
                seen = seen_in_frame.setdefault(int(fid), set())
                if tid in seen:
                    continue
                seen.add(tid)
                x1, y1, x2, y2 = boxes[j]
                w, h = float(x2 - x1), float(y2 - y1)
                prepared.append(
                    f"{int(fid)+offset},{tid},{float(x1):.2f},"
                    f"{float(y1):.2f},{w:.2f},{h:.2f},1,1,1.0"
                )
                any_box = True
        if not any_box:
            continue

        seq = _safe_video_id(v.video_id)
        seq_names.append(seq)
        seq_offsets[seq] = offset
        seq_dir = gt_root / seq
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(prepared))
        (seq_dir / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={seq}\n"
            f"seqLength={len(v.frame_ids)}\n"
            "imWidth=1024\n"
            "imHeight=1024\n"
            "imExt=.jpg\n"
        )

    seqmap_dir = gt_root.parent.parent / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    smfile = seqmap_dir / f"{dataset_name}_{class_name}-test.txt"
    smfile.write_text("name\n" + "\n".join(seq_names) + "\n")
    return seq_names, seq_offsets


def _populate_tracker(
    run_dir: Path, class_name: str, tracker_name: str,
    trackers_root: Path, seq_offsets: dict[str, int],
    seq_names: list[str],
) -> None:
    src = run_dir / "mot_format" / class_name
    if not src.is_dir():
        raise FileNotFoundError(f"missing {src}")
    dst = trackers_root / tracker_name / "data"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    seq_set = set(seq_names)
    for f in src.glob("*.txt"):
        seq = f.stem
        if seq not in seq_set:
            continue
        offset = seq_offsets.get(seq, 0)
        if offset == 0:
            shutil.copyfile(f, dst / f.name)
        else:
            out_lines = []
            for line in f.read_text().splitlines():
                if not line:
                    continue
                parts = line.split(",")
                parts[0] = str(int(parts[0]) + offset)
                out_lines.append(",".join(parts))
            (dst / f.name).write_text("\n".join(out_lines))


def _eval_dataset_class(
    dataset_name: str, class_name: str, workspace: Path,
    runs_for_dataset: dict[str, Path],
) -> dict[str, dict]:
    import trackeval

    benchmark = f"{dataset_name}_{class_name}"
    split = "test"

    gt_root = workspace / "gt" / f"{benchmark}-{split}"
    trackers_root = workspace / "trackers" / f"{benchmark}-{split}"
    if gt_root.exists():  shutil.rmtree(gt_root)
    if trackers_root.exists(): shutil.rmtree(trackers_root)
    gt_root.mkdir(parents=True)
    trackers_root.mkdir(parents=True)

    seq_names, seq_offsets = _write_gt_for_class(dataset_name, class_name, gt_root)
    if not seq_names:
        print(f"  [skip] {dataset_name}/{class_name}: no sequences contain GT")
        return {}

    for tr_name, run_dir in runs_for_dataset.items():
        try:
            _populate_tracker(run_dir, class_name, tr_name, trackers_root,
                              seq_offsets, seq_names)
        except FileNotFoundError as e:
            print(f"  [warn] {tr_name}: {e}")

    out: dict[str, dict] = {}
    for tr_name in runs_for_dataset.keys():
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                         "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                         "BREAK_ON_ERROR": False, "RETURN_ON_ERROR": True})
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER":         str(workspace / "gt"),
            "TRACKERS_FOLDER":   str(workspace / "trackers"),
            "OUTPUT_FOLDER":     str(workspace / "output"),
            "TRACKERS_TO_EVAL":  [tr_name],
            "CLASSES_TO_EVAL":   ["pedestrian"],
            "BENCHMARK":         benchmark,
            "SPLIT_TO_EVAL":     split,
            "PRINT_CONFIG":      False,
            "DO_PREPROC":        False,
            "SEQMAP_FOLDER":     str(workspace / "seqmaps"),
            "SEQMAP_FILE":       str(workspace / "seqmaps" / f"{benchmark}-{split}.txt"),
            "SKIP_SPLIT_FOL":    False,
        })
        metrics_list = [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        try:
            raw, _ = trackeval.Evaluator(eval_cfg).evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics_list,
            )
        except Exception as e:
            print(f"  {tr_name}: evaluate raised {e!r}; skip")
            continue
        bench = raw.get("MotChallenge2DBox", {}).get(tr_name)
        if not bench or "COMBINED_SEQ" not in bench:
            print(f"  {tr_name}: no COMBINED_SEQ; skip")
            continue
        cs = bench["COMBINED_SEQ"].get("pedestrian")
        if cs is None:
            continue
        hota = cs["HOTA"]; clear = cs["CLEAR"]; ident = cs["Identity"]
        out[tr_name] = {
            "HOTA": float(np.mean(hota["HOTA"])),
            "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])),
            "LocA": float(np.mean(hota["LocA"])),
            "MOTA": float(clear["MOTA"]),
            "MOTP": float(clear["MOTP"]),
            "IDF1": float(ident["IDF1"]),
            "IDsw": int(clear["IDSW"]),
            "MT":   int(clear["MT"]),
            "ML":   int(clear["ML"]),
            "n_dets": int(clear["CLR_TP"]) + int(clear["CLR_FP"]),
            "n_seqs": len(seq_names),
        }
    return out


_DATASET_SLOTS = ("satmtb_nocar", "viso_nocar", "airmot")
_RUN_RE = re.compile(
    r"^(?P<tracker>[a-zA-Z][a-zA-Z0-9_]*?)_"
    r"(?P<dataset>" + "|".join(_DATASET_SLOTS) + r")_"
    r"(?P<ts>\d{8}_\d{6})$"
)


def _collect_runs(root: Path) -> dict[str, dict[str, Path]]:
    by_pair: dict[tuple[str, str], Path] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        key = (m["dataset"], m["tracker"])
        prev = by_pair.get(key)
        if prev is None or d.name > prev.name:
            by_pair[key] = d
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for (ds, tr), path in by_pair.items():
        grouped[ds][tr] = path
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker-output-root", required=True)
    ap.add_argument("--workspace", default="/tmp/hota_workspace_satmtb_hbb")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.tracker_output_root)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(root)
    if not runs:
        print(f"no runs in {root}", file=sys.stderr); sys.exit(1)
    print("Runs found:")
    for ds in sorted(runs):
        for tr in sorted(runs[ds]):
            print(f"  {ds}/{tr} → {runs[ds][tr].name}")

    rows: list[dict] = []
    for ds_name, runs_for_ds in runs.items():
        cmap = _DATASET_TABLE[ds_name][4]
        for class_name in cmap.keys():
            print(f"\n=== {ds_name} / {class_name} ===")
            metrics = _eval_dataset_class(ds_name, class_name, workspace, runs_for_ds)
            for tr_name, vals in metrics.items():
                rows.append({"dataset": ds_name, "tracker": tr_name,
                             "class": class_name, **vals})
                print(f"  {tr_name:12s}  HOTA={vals['HOTA']:.3f}  DetA={vals['DetA']:.3f}  "
                      f"AssA={vals['AssA']:.3f}  MOTA={vals['MOTA']:.3f}  "
                      f"IDF1={vals['IDF1']:.3f}  IDsw={vals['IDsw']}  n_seqs={vals['n_seqs']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if not rows:
            print("no rows produced!", file=sys.stderr); sys.exit(2)
        fieldnames = ["dataset", "tracker", "class", "HOTA", "DetA", "AssA", "LocA",
                      "MOTA", "MOTP", "IDF1", "IDsw", "MT", "ML", "n_dets", "n_seqs"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\n{len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
