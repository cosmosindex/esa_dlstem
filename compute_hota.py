"""
Compute HOTA / DetA / AssA / IDF1 / MOTA across every (tracker × dataset)
output produced by ``run_all_tracker.sh``. Builds a TrackEval workspace
under ``--workspace``, populates GT + tracker files in MOTChallenge
format, and writes a single CSV summary.

The eval script in this repo (``eval_tracker.py``) already dumps each
tracker's per-video output to ``<run_dir>/mot_format/<safe_video_id>.txt``
in MOTChallenge format::

    frame, id, x, y, w, h, conf, -1, -1, -1

We just need to (a) write GT files per dataset in the same format and
(b) lay out the workspace TrackEval expects::

    workspace/
      gt/<dataset>/<seq>/gt/gt.txt
      gt/<dataset>/<seq>/seqinfo.ini
      gt/<dataset>/<seq>map_test.txt           (sequence list)
      trackers/<dataset>/<tracker>/data/<seq>.txt

Each *dataset* is treated as a separate TrackEval ``BENCHMARK`` so the
sequence sets and metric tables don't get cross-contaminated.

Usage::

    python compute_hota.py \
        --tracker-output-root /data/.../experiments/MOT/tracker_20260427 \
        --workspace /tmp/hota_workspace \
        --output /data/.../experiments/MOT/tracker_20260427/hota_summary.csv
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
from datasets.rscardata import RsCarDataset
from datasets.satmtb import SATMTBDataset
from datasets.sdmcar import SDMCarDataset
from datasets.viso import VISODataset


# For SAM3 + HOTA we evaluate every dataset's car/plane/ship/train
# tracks together (no per-class restriction). The original SORT-family
# eval restricted SAT-MTB to car only; SAM3 is multi-class so we lift
# that restriction here. ``class_map`` for HOTA is only used to gate
# `_map_label` in `_load_annotations`, and TrackEval treats every kept
# track as a single foreground class anyway.
_SAM3_CLASS_MAPS = {
    "rscardata":   {"car": 0},
    "satmtb":      {"airplane": 0, "car": 1, "ship": 2, "train": 3},
    "sdmcar":      {"car": 0},
    "airmot":      {"airplane": 0, "ship": 1},
    "viso_no_car": {"plane": 0, "ship": 1, "train": 2},
}

_DATASET_TABLE = {
    "rscardata":   (RsCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/RsCarData", {}),
    "satmtb":      (SATMTBDataset, "/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB",
                    {"task": "mot"}),
    "sdmcar":      (SDMCarDataset, "/data/ESA_DLSTEM_2025/data/trafic/SDM-Car", {}),
    "airmot":      (AIRMOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100", {}),
    # VISO non-car split — used by the SAM3 + RAFT pipeline as the
    # "everything except car" half; the car half comes from rscardata.
    "viso_no_car": (VISODataset, "/data/ESA_DLSTEM_2025/data/trafic/VISO",
                    {"categories": ["plane", "ship", "train"]}),
    # viso_combined is materialised in main() by merging the GT + tracker
    # dirs of viso_no_car and rscardata; no standalone dataset class.
}

# Datasets that contribute to the "viso_combined" composite benchmark.
_VISO_COMBINE_PARTS = ("viso_no_car", "rscardata")


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _build_dataset(name: str):
    cls, root, extra = _DATASET_TABLE[name]
    kwargs = dict(extra)
    # The class_map only gates which categories `_load_annotations`
    # surfaces. TrackEval downstream pools every surviving box as a
    # single foreground class, so multi-class datasets need every
    # category mapped non-negative (otherwise plane/ship boxes get
    # silently dropped on read). See _SAM3_CLASS_MAPS.
    cmap = _SAM3_CLASS_MAPS.get(name, {"car": 0})
    if cls is VISODataset or cls is AIRMOTDataset:
        # Neither accepts ``mode="detection"`` — they're MOT-only loaders.
        return cls(root=root, split="test", class_map=cmap, **kwargs)
    return cls(root=root, split="test", mode="detection",
               class_map=cmap, **kwargs)


# ----------------------------------------------------------------------
# Workspace materialisation
# ----------------------------------------------------------------------

def _write_gt(dataset_name: str, gt_root: Path) -> tuple[list[str], dict[str, int]]:
    """Dump GT for every test-split video in MOTChallenge format.

    Returns ``(seq_names, seq_offsets)`` where ``seq_offsets`` maps the
    MOT-format seq name (the safe video id) → the per-video frame
    offset applied so timesteps are 1-indexed (TrackEval requirement;
    SDM-Car ships 0-indexed frame_ids).
    """
    ds = _build_dataset(dataset_name)
    seq_names: list[str] = []
    seq_offsets: dict[str, int] = {}

    for v in ds.videos:
        seq = _safe_video_id(v.video_id)
        seq_names.append(seq)
        seq_dir = gt_root / seq
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)

        first_fid = min(v.frame_ids)
        offset = 1 - int(first_fid) if first_fid < 1 else 0
        seq_offsets[seq] = offset

        lines = []
        for fid in v.frame_ids:
            ann = ds._load_annotations(v, fid)
            boxes = ann["boxes"]; tids = ann["track_ids"]
            # Dedupe within a frame — TrackEval rejects duplicate (frame,
            # id) pairs (SDM-Car's gt.csv ships occasional duplicates).
            seen_in_frame: set[int] = set()
            for j in range(len(boxes)):
                tid = int(tids[j])
                # Skip sentinel "no track" rows (SDM-Car uses tid=-1 for
                # detections without an assigned track). TrackEval's
                # contiguous-id relabeling collapses negative ids onto
                # tail indices, producing spurious dup-id errors.
                if tid < 0:
                    continue
                if tid in seen_in_frame:
                    continue
                seen_in_frame.add(tid)
                x1, y1, x2, y2 = boxes[j]
                w, h = float(x2 - x1), float(y2 - y1)
                # MOTChallenge GT: frame,id,x,y,w,h,conf,cls,vis. We set
                # conf=1 (always considered), cls=1 (TrackEval's
                # MotChallenge2DBox treats class 1 as 'pedestrian' /
                # single foreground), vis=1.0.
                lines.append(
                    f"{int(fid)+offset},{tid},{float(x1):.2f},"
                    f"{float(y1):.2f},{w:.2f},{h:.2f},1,1,1.0"
                )
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines))

        # seqinfo.ini — TrackEval reads ``seqLength`` from this.
        seqinfo = (
            "[Sequence]\n"
            f"name={seq}\n"
            f"seqLength={len(v.frame_ids)}\n"
            "imWidth=1024\n"
            "imHeight=1024\n"
            "imExt=.jpg\n"
        )
        (seq_dir / "seqinfo.ini").write_text(seqinfo)

    # seqmap (TrackEval needs a "name" column with the sequence list).
    seqmap_dir = gt_root.parent.parent / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    smfile = seqmap_dir / f"{dataset_name}-test.txt"
    smfile.write_text("name\n" + "\n".join(seq_names) + "\n")
    return seq_names, seq_offsets


def _populate_tracker_output(
    run_dir: Path, dataset_name: str, tracker_name: str,
    trackers_root: Path, seq_offsets: dict[str, int] | None = None,
    mot_subdir: str = "mot_format",
):
    """Copy tracker MOT output into the TrackEval workspace,
    applying any per-sequence frame-index offset (e.g. SDM-Car's
    0-indexed frames need shifting to 1-indexed). ``mot_subdir`` selects
    which child of ``run_dir`` to read (e.g. ``mot_format_filtered`` for
    the RAFT-filtered SAM3 output)."""
    src = run_dir / mot_subdir
    if not src.is_dir():
        raise FileNotFoundError(f"missing {mot_subdir}/ under {run_dir}")
    dst = trackers_root / tracker_name / "data"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        seq = f.stem
        offset = (seq_offsets or {}).get(seq, 0)
        if offset == 0:
            shutil.copyfile(f, dst / f.name)
            continue
        out_lines = []
        for line in f.read_text().splitlines():
            if not line:
                continue
            parts = line.split(",")
            parts[0] = str(int(parts[0]) + offset)
            out_lines.append(",".join(parts))
        (dst / f.name).write_text("\n".join(out_lines))


# ----------------------------------------------------------------------
# Eval entrypoint
# ----------------------------------------------------------------------

def _eval_one_dataset(
    dataset_name: str, workspace: Path, run_dirs: dict[str, Path],
    mot_subdir_per_tracker: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Run TrackEval HOTA + CLEAR + Identity for one dataset → tracker
    metrics dict ``{tracker: {"HOTA": ..., "DetA": ..., "AssA": ..., "MOTA": ..., "IDF1": ...}}``."""
    import trackeval

    benchmark = dataset_name
    split_to_eval = "test"

    gt_root_for_bench = workspace / "gt" / f"{benchmark}-{split_to_eval}"
    trackers_for_bench = workspace / "trackers" / f"{benchmark}-{split_to_eval}"
    if gt_root_for_bench.exists():
        shutil.rmtree(gt_root_for_bench)
    if trackers_for_bench.exists():
        shutil.rmtree(trackers_for_bench)
    gt_root_for_bench.mkdir(parents=True)
    trackers_for_bench.mkdir(parents=True)

    seq_names, seq_offsets = _write_gt(dataset_name, gt_root_for_bench)
    mot_subdir_per_tracker = mot_subdir_per_tracker or {}
    for tracker_name, run_dir in run_dirs.items():
        _populate_tracker_output(
            run_dir, dataset_name, tracker_name, trackers_for_bench,
            seq_offsets=seq_offsets,
            mot_subdir=mot_subdir_per_tracker.get(tracker_name, "mot_format"),
        )

    # Run each tracker in its own evaluator call so a TrackEval edge case
    # on one (tracker, seq) doesn't kill the rest of the sweep.
    out: dict[str, dict] = {}
    for tracker_name in run_dirs.keys():
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "BREAK_ON_ERROR": False,
            "RETURN_ON_ERROR": True,
        })

        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER":         str(workspace / "gt"),
            "TRACKERS_FOLDER":   str(workspace / "trackers"),
            "OUTPUT_FOLDER":     str(workspace / "output"),
            "TRACKERS_TO_EVAL":  [tracker_name],
            "CLASSES_TO_EVAL":   ["pedestrian"],
            "BENCHMARK":         benchmark,
            "SPLIT_TO_EVAL":     split_to_eval,
            "PRINT_CONFIG":      False,
            "DO_PREPROC":        False,
            "SEQMAP_FOLDER":     str(workspace / "seqmaps"),
            "SEQMAP_FILE":       str(workspace / "seqmaps" / f"{dataset_name}-test.txt"),
            "SKIP_SPLIT_FOL":    False,
        })

        metrics_list = [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]

        evaluator = trackeval.Evaluator(eval_cfg)
        try:
            raw, _ = evaluator.evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                metrics_list,
            )
        except Exception as exc:
            print(f"  {tracker_name}: evaluate() raised {exc!r}; skipping.")
            continue

        res = raw.get("MotChallenge2DBox", {})
        bench_res = res.get(tracker_name)
        if bench_res is None or "COMBINED_SEQ" not in bench_res:
            print(f"  {tracker_name}: no COMBINED_SEQ result (per-seq error?).")
            continue
        cs = bench_res["COMBINED_SEQ"].get("pedestrian")
        if cs is None:
            print(f"  {tracker_name}: combined-seq is None; skipping.")
            continue
        hota = cs["HOTA"]
        clear = cs["CLEAR"]
        ident = cs["Identity"]
        out[tracker_name] = {
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
        }
    return out




def _eval_viso_combined(
    workspace: Path, run_dirs: dict[str, Path],
    mot_subdir_per_tracker: dict[str, str],
) -> dict[str, dict]:
    """Evaluate viso_no_car + rscardata pooled into a single TrackEval
    benchmark. The two datasets ship disjoint sequence names, so we just
    materialise both into ``gt/viso_combined-test/`` and merge the
    ``mot_format/`` files of the matching tracker run from each half.
    """
    import trackeval

    benchmark = "viso_combined"
    split = "test"
    gt_root = workspace / "gt" / f"{benchmark}-{split}"
    trackers_root = workspace / "trackers" / f"{benchmark}-{split}"
    if gt_root.exists():
        shutil.rmtree(gt_root)
    if trackers_root.exists():
        shutil.rmtree(trackers_root)
    gt_root.mkdir(parents=True)
    trackers_root.mkdir(parents=True)

    # Reuse _write_gt by writing each part's GT into the same combined
    # gt root. Sequence names collide-proof because the source datasets
    # have disjoint seq slugs (viso uses "plane_044" etc, rscardata
    # uses "test1024_002" etc).
    seq_offsets_combined: dict[str, int] = {}
    seq_names_combined: list[str] = []
    for part in _VISO_COMBINE_PARTS:
        seqs, offsets = _write_gt(part, gt_root)
        seq_names_combined.extend(seqs)
        seq_offsets_combined.update(offsets)

    seqmap_dir = workspace / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    (seqmap_dir / f"{benchmark}-{split}.txt").write_text(
        "name\n" + "\n".join(seq_names_combined) + "\n"
    )

    # Track which sequences belong to which part so we can copy from
    # the right run dir per tracker.
    seq_to_part: dict[str, str] = {}
    for part in _VISO_COMBINE_PARTS:
        ds = _build_dataset(part)
        for v in ds.videos:
            seq_to_part[_safe_video_id(v.video_id)] = part

    # Each tracker key here was set up above to point at *some* run dir;
    # but for pooling we actually want to take the matching half from
    # the corresponding tracker run for each dataset_part. We rebuild
    # by re-reading runs[part][tr_base] for both halves of every tr.
    # Since main() handed us a flat dict (tracker_key -> some path),
    # we instead re-discover via root scan — keep it simple: derive the
    # canonical run dir for each (part, tracker_key) by stripping any
    # "_raft" suffix.
    out: dict[str, dict] = {}

    # Sniff trackers from the input dict; for each one we copy *both*
    # halves' mot output into the same trackers/<tr>/data dir.
    for tr_key, any_run_dir in run_dirs.items():
        sub = mot_subdir_per_tracker.get(tr_key, "mot_format")
        # Find the canonical run dir for each part. The run_dirs dict
        # only kept the most recent path per (part, tr_key) — but
        # pooling needs the run from each half. Walk the parent dir
        # of any_run_dir (== tracker_output_root) and pick the latest
        # match for the *base* tracker name (strip "_raft").
        base_tr = tr_key[:-len("_raft")] if tr_key.endswith("_raft") else tr_key
        root = any_run_dir.parent
        per_part_run: dict[str, Path] = {}
        for part in _VISO_COMBINE_PARTS:
            cand = sorted(
                d for d in root.iterdir()
                if d.is_dir() and (
                    d.name.startswith(f"{base_tr}_{part}_")
                )
            )
            if cand:
                per_part_run[part] = cand[-1]

        if len(per_part_run) != len(_VISO_COMBINE_PARTS):
            print(f"  {tr_key}: missing one or both halves "
                  f"({sorted(per_part_run.keys())}); skipping combine.")
            continue

        dst = trackers_root / tr_key / "data"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        for part, run_dir in per_part_run.items():
            src = run_dir / sub
            if not src.is_dir():
                print(f"  {tr_key}/{part}: missing {sub}/, skipping half.")
                continue
            for f in src.glob("*.txt"):
                seq = f.stem
                offset = seq_offsets_combined.get(seq, 0)
                if offset == 0:
                    shutil.copyfile(f, dst / f.name)
                    continue
                lines = []
                for ln in f.read_text().splitlines():
                    if not ln:
                        continue
                    parts = ln.split(",")
                    parts[0] = str(int(parts[0]) + offset)
                    lines.append(",".join(parts))
                (dst / f.name).write_text("\n".join(lines))

        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({
            "USE_PARALLEL": False, "PRINT_RESULTS": False,
            "PRINT_CONFIG": False, "TIME_PROGRESS": False,
            "BREAK_ON_ERROR": False, "RETURN_ON_ERROR": True,
        })
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER":         str(workspace / "gt"),
            "TRACKERS_FOLDER":   str(workspace / "trackers"),
            "OUTPUT_FOLDER":     str(workspace / "output"),
            "TRACKERS_TO_EVAL":  [tr_key],
            "CLASSES_TO_EVAL":   ["pedestrian"],
            "BENCHMARK":         benchmark,
            "SPLIT_TO_EVAL":     split,
            "PRINT_CONFIG":      False,
            "DO_PREPROC":        False,
            "SEQMAP_FOLDER":     str(seqmap_dir),
            "SEQMAP_FILE":       str(seqmap_dir / f"{benchmark}-{split}.txt"),
            "SKIP_SPLIT_FOL":    False,
        })
        metrics_list = [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        evaluator = trackeval.Evaluator(eval_cfg)
        try:
            raw, _ = evaluator.evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                metrics_list,
            )
        except Exception as exc:
            print(f"  {tr_key}: combined evaluate() raised {exc!r}; skipping.")
            continue

        bench_res = raw.get("MotChallenge2DBox", {}).get(tr_key)
        if bench_res is None or "COMBINED_SEQ" not in bench_res:
            print(f"  {tr_key}: no COMBINED_SEQ result.")
            continue
        cs = bench_res["COMBINED_SEQ"].get("pedestrian")
        if cs is None:
            continue
        hota = cs["HOTA"]; clear = cs["CLEAR"]; ident = cs["Identity"]
        out[tr_key] = {
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
        }
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

# Tracker name is anything word-ish (sam3, sam3_text, sort, …) so this
# can pick up both the original SORT-family runs and SAM3 / RAFT runs
# created downstream. The dataset slot is the bounded set of registered
# benchmarks (so e.g. sdmcar in the timestamp suffix doesn't get parsed
# as a dataset by accident).
_DATASET_SLOTS = ("rscardata", "satmtb", "sdmcar", "viso_no_car",
                  "viso", "airmot", "satmot")
_RUN_RE = re.compile(
    r"^(?P<tracker>[a-zA-Z][a-zA-Z0-9_]*?)_"
    r"(?P<dataset>" + "|".join(_DATASET_SLOTS) + r")_"
    r"(?P<ts>\d{8}_\d{6})$"
)


def _collect_runs(root: Path) -> dict[str, dict[str, Path]]:
    """``{dataset: {tracker: most_recent_run_dir}}``."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-output-root", required=True,
                        help="Dir containing <tracker>_<dataset>_<TS>/ run dirs.")
    parser.add_argument("--workspace", default="/tmp/hota_workspace",
                        help="Scratch dir for the TrackEval layout.")
    parser.add_argument("--output", required=True,
                        help="CSV path for the combined HOTA / MOTA / IDF1 table.")
    parser.add_argument("--also-filtered", action="store_true",
                        help="For every discovered run, also evaluate "
                             "mot_format_filtered/ as a sibling tracker "
                             "named '<tracker>_raft'. Skips runs that have "
                             "no mot_format_filtered/ subdir.")
    parser.add_argument("--combine-viso", action="store_true",
                        help="After per-dataset eval, also report a "
                             "'viso_combined' row that pools the "
                             "viso_no_car and rscardata sequences into a "
                             "single TrackEval benchmark.")
    args = parser.parse_args()

    root = Path(args.tracker_output_root)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(root)
    if not runs:
        print(f"no tracker runs found under {root}", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered runs:")
    for ds in sorted(runs):
        for tr in sorted(runs[ds]):
            print(f"  {ds}/{tr} → {runs[ds][tr].name}")

    # Optionally double-up each tracker with a "_raft" twin reading the
    # filtered MOT output. We expand here so _eval_one_dataset runs both
    # in a single pass per dataset (one GT-write, two trackers).
    mot_subdir_per_tracker: dict[str, dict[str, str]] = defaultdict(dict)
    if args.also_filtered:
        for ds, tracker_runs in runs.items():
            extra: dict[str, Path] = {}
            for tr, run_dir in tracker_runs.items():
                if (run_dir / "mot_format_filtered").is_dir():
                    extra[f"{tr}_raft"] = run_dir
                    mot_subdir_per_tracker[ds][f"{tr}_raft"] = "mot_format_filtered"
            tracker_runs.update(extra)

    rows: list[dict] = []
    for ds, tracker_runs in sorted(runs.items()):
        print(f"\n=== {ds} ===")
        try:
            metrics = _eval_one_dataset(
                ds, workspace, tracker_runs,
                mot_subdir_per_tracker=mot_subdir_per_tracker.get(ds),
            )
        except Exception as exc:
            print(f"  failed: {exc!r}")
            continue
        for tr in sorted(metrics):
            m = metrics[tr]
            rows.append({
                "dataset": ds, "tracker": tr,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            })
            print(f"  {tr:14s}  HOTA={m['HOTA']*100:5.2f}  "
                  f"DetA={m['DetA']*100:5.2f}  AssA={m['AssA']*100:5.2f}  "
                  f"MOTA={m['MOTA']*100:5.2f}  IDF1={m['IDF1']*100:5.2f}  "
                  f"IDsw={m['IDsw']}")

    # ---- Composite VISO benchmark (no_car + rscardata) ----
    if args.combine_viso:
        combined_runs: dict[str, Path] = {}
        combined_subdir: dict[str, str] = {}
        for ds_part in _VISO_COMBINE_PARTS:
            for tr, rd in runs.get(ds_part, {}).items():
                # Use a tracker key that survives both halves; if the
                # same tracker (e.g. sam3_text) appears in both, point
                # at the last-seen run dir for that key. The eval below
                # consults `combined_subdir` to pick mot_format vs
                # mot_format_filtered per (tracker_part, ds_part).
                combined_runs[tr] = rd
                # Inherit the per-(ds, tr) subdir hint chosen above so
                # tracker '<tr>_raft' reads filtered output for both
                # halves.
                if tr in mot_subdir_per_tracker.get(ds_part, {}):
                    combined_subdir[tr] = mot_subdir_per_tracker[ds_part][tr]
        if combined_runs:
            print(f"\n=== viso_combined ===")
            try:
                metrics = _eval_viso_combined(
                    workspace, combined_runs, combined_subdir,
                )
            except Exception as exc:
                print(f"  failed: {exc!r}")
                metrics = {}
            for tr in sorted(metrics):
                m = metrics[tr]
                rows.append({
                    "dataset": "viso_combined", "tracker": tr,
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in m.items()},
                })
                print(f"  {tr:14s}  HOTA={m['HOTA']*100:5.2f}  "
                      f"DetA={m['DetA']*100:5.2f}  AssA={m['AssA']*100:5.2f}  "
                      f"MOTA={m['MOTA']*100:5.2f}  IDF1={m['IDF1']*100:5.2f}  "
                      f"IDsw={m['IDsw']}")

    if rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
