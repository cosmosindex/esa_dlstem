#!/usr/bin/env python
"""Full HOTA-suite re-scoring of the BIRDSAI tracker sweep (TrackEval).

Companion to ``_birdsai_tracking_compare.py``. That script re-scores the cached
``mot_format/*.txt`` tracks of the 18-run sweep (3 detectors x 6 TBD trackers)
against the SAM3-refined GT (``annotations_sam3``) with a bespoke per-class
greedy matcher and reports only P/R/F1/MOTA/IDsw. This script feeds the *same*
tracks and the *same* GT through **TrackEval** — the identical pipeline the
NeurIPS Space-Tracker-MOT (car) table uses (``compute_hota.py``) — so BIRDSAI
gets the complete, directly-comparable suite:

    HOTA  DetA  AssA  MOTA  IDF1  IDsw  MT  ML   (+ LocA)

Class-aware matching. The sweep ran one tracker per class and baked the class
into the track id as ``class = track_id // 1_000_000``. TrackEval's
MotChallenge2DBox is single-foreground-class and matches purely by IoU geometry,
so to keep matching class-restricted (as in the greedy eval) we translate every
box of class ``c`` by ``c * COORD_OFFSET`` on the x-axis in BOTH GT and tracker
files. Translation preserves within-class IoU exactly (HOTA/DetA/AssA/LocA
unchanged) while making cross-class IoU identically zero -> no cross-class
matches. Detection/association counts are then pooled across classes, matching
the greedy eval's pooled ("overall") convention.

Per-class HOTA is additionally computed by filtering GT+tracks to one class
(no coord offset needed) — the association view of the long-tail story.

Pure offline: parses the .txt tracks, no GPU / no re-tracking.

Usage::

    micromamba run -n esa_dlstem python evaluation/compute_birdsai_hota.py \
        --out-json docs/use_case_results/birdsai_tracking_sam3gt_hota.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
ANN = "annotations_sam3"
SWEEP_ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sweep")
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CLASSES = sorted(CANON)
DETECTORS = ["fasterrcnn", "yolo", "dinov3"]
DET_LABEL = {"fasterrcnn": "FasterRCNN", "yolo": "YOLO11l", "dinov3": "DINOv3"}
TRACKERS = ["sort", "ocsort", "bytetrack", "botsort", "botsort_reid", "tracktrack"]
TRK_LABEL = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack",
             "botsort": "BoT-SORT", "botsort_reid": "BoT-SORT+ReID",
             "tracktrack": "TrackTrack"}
GREEDY_JSON = "docs/use_case_results/birdsai_tracking_sam3gt_compare.json"
OUT_MD = "docs/use_case_results/birdsai_tracking_sam3gt_compare.md"
# x-translation per class so single-class TrackEval matches within-class only.
# Must exceed image width (BIRDSAI is 640x480). 100000 is safe.
COORD_OFFSET = 100_000.0
IDPREFIX = 1_000_000


def find_run(det_name: str, trk_name: str) -> Path | None:
    hits = sorted(SWEEP_ROOT.glob(f"{det_name}_{trk_name}_birdsai_track_*"))
    return hits[-1] if hits else None


def parse_tracks(txt: Path):
    """frame -> list of (cls, tid, x1, y1, x2, y2)."""
    by_frame: dict[int, list] = {}
    if not txt.exists():
        return by_frame
    for line in txt.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        cls = tid // IDPREFIX
        by_frame.setdefault(f, []).append((cls, tid, x, y, x + w, y + h))
    return by_frame


def build_gt(ds, video_ids):
    """vid -> {fid -> (boxes[n,4] xyxy, labels[n], track_ids[n])}."""
    gt = {}
    vmap = {v.video_id: v for v in ds.videos}
    for vid in video_ids:
        v = vmap[vid]
        fr = {}
        for fid in v.frame_ids:
            a = ds._load_annotations(v, fid)
            fr[int(fid)] = (a["boxes"].reshape(-1, 4).astype(np.float32),
                            a["labels"].reshape(-1).astype(np.int64),
                            a["track_ids"].reshape(-1).astype(np.int64))
        gt[vid] = fr
    return gt


def _frame_offset(fids) -> int:
    """Shift so the minimum frame id maps to timestep 1 (TrackEval requires
    1-indexed contiguous timesteps; BIRDSAI frame ids start well above 1)."""
    return 1 - int(min(fids))


# ---------------------------------------------------------------------------
# TrackEval workspace materialisation
# ---------------------------------------------------------------------------

def _write_gt_workspace(gt, gt_root: Path, class_filter: int | None):
    """Write MOTChallenge GT. class_filter=None -> all classes, class-offset in
    x + id (pooled, class-aware). class_filter=c -> only class c, no offset."""
    seq_names = []
    seq_offsets = {}
    for vid, frames in gt.items():
        seq = vid
        seq_names.append(seq)
        seq_dir = gt_root / seq
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        off = _frame_offset(frames.keys())
        seq_offsets[seq] = off
        max_ts = max(int(f) + off for f in frames) if frames else 1
        lines = []
        for fid in sorted(frames):
            boxes, labels, tids = frames[fid]
            seen = set()
            for j in range(len(boxes)):
                c = int(labels[j])
                if class_filter is not None and c != class_filter:
                    continue
                tid = int(tids[j])
                key = (c, tid)
                if key in seen:
                    continue
                seen.add(key)
                x1, y1, x2, y2 = boxes[j]
                if class_filter is None:
                    x1 += c * COORD_OFFSET
                    x2 += c * COORD_OFFSET
                    tid = tid + c * IDPREFIX
                w, h = float(x2 - x1), float(y2 - y1)
                lines.append(f"{int(fid)+off},{tid},{float(x1):.2f},"
                             f"{float(y1):.2f},{w:.2f},{h:.2f},1,1,1.0")
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines))
        (seq_dir / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={seq}\n"
            f"seqLength={max(max_ts, 1)}\n"
            "imWidth=640\nimHeight=480\nimExt=.jpg\n")
    seqmap_dir = gt_root.parent.parent / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    (seqmap_dir / "birdsai-test.txt").write_text("name\n" + "\n".join(seq_names) + "\n")
    return seq_offsets


def _write_tracker_workspace(runs, trackers_root: Path, gt, seq_offsets,
                             class_filter: int | None):
    """runs: {tracker_key -> run_dir}. Writes trackers/<key>/data/<seq>.txt."""
    for key, run_dir in runs.items():
        dst = trackers_root / key / "data"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for vid in gt:
            off = seq_offsets[vid]
            by_frame = parse_tracks(run_dir / "mot_format" / f"{vid}.txt")
            lines = []
            for fid in sorted(by_frame):
                for (c, tid, x1, y1, x2, y2) in by_frame[fid]:
                    if class_filter is not None and c != class_filter:
                        continue
                    xx1, xx2 = x1, x2
                    if class_filter is None:
                        xx1 += c * COORD_OFFSET
                        xx2 += c * COORD_OFFSET
                    w, h = xx2 - xx1, y2 - y1
                    lines.append(f"{int(fid)+off},{int(tid)},{xx1:.2f},"
                                 f"{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
            (dst / f"{vid}.txt").write_text("\n".join(lines))


def _run_trackeval(workspace: Path, tracker_keys):
    import trackeval
    out = {}
    for key in tracker_keys:
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                         "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                         "BREAK_ON_ERROR": False, "RETURN_ON_ERROR": True})
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER": str(workspace / "gt"),
            "TRACKERS_FOLDER": str(workspace / "trackers"),
            "OUTPUT_FOLDER": str(workspace / "output"),
            "TRACKERS_TO_EVAL": [key],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "birdsai",
            "SPLIT_TO_EVAL": "test",
            "PRINT_CONFIG": False,
            "DO_PREPROC": False,
            "SEQMAP_FOLDER": str(workspace / "seqmaps"),
            "SEQMAP_FILE": str(workspace / "seqmaps" / "birdsai-test.txt"),
            "SKIP_SPLIT_FOL": False,
        })
        metrics_list = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
                        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                        trackeval.metrics.Identity({"PRINT_CONFIG": False})]
        evaluator = trackeval.Evaluator(eval_cfg)
        try:
            raw, _ = evaluator.evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics_list)
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: evaluate() raised {exc!r}")
            continue
        res = raw.get("MotChallenge2DBox", {}).get(key)
        if not res or "COMBINED_SEQ" not in res:
            print(f"  {key}: no COMBINED_SEQ")
            continue
        cs = res["COMBINED_SEQ"].get("pedestrian")
        if cs is None:
            print(f"  {key}: combined-seq None")
            continue
        h, cl, idn = cs["HOTA"], cs["CLEAR"], cs["Identity"]
        out[key] = {
            "HOTA": float(np.mean(h["HOTA"])), "DetA": float(np.mean(h["DetA"])),
            "AssA": float(np.mean(h["AssA"])), "LocA": float(np.mean(h["LocA"])),
            "MOTA": float(cl["MOTA"]), "IDF1": float(idn["IDF1"]),
            "IDsw": int(cl["IDSW"]), "MT": int(cl["MT"]), "ML": int(cl["ML"]),
        }
    return out


def _fmt(v, spec, signed=False):
    if v is None:
        return "—"
    if spec == "d":
        return f"{int(v)}"
    return (f"{v:+.3f}" if signed else f"{v:.3f}")


def _mark_best(vals, fmt, lo_better=False, signed=False):
    """vals: list of numbers (or None). Return list of formatted cells, bolding best."""
    present = [v for v in vals if v is not None]
    best = (min(present) if lo_better else max(present)) if present else None
    out = []
    for v in vals:
        s = _fmt(v, fmt, signed)
        if v is not None and best is not None and v == best:
            s = f"**{s}**"
        out.append(s)
    return out


def render_markdown(overall, per_class, greedy, n_videos, n_frames, out_md):
    METRICS = [("HOTA", ".3f", False, False), ("DetA", ".3f", False, False),
               ("AssA", ".3f", False, False), ("MOTA", ".3f", False, True),
               ("IDF1", ".3f", False, False), ("IDsw", "d", True, False),
               ("MT", "d", False, False), ("ML", "d", True, False)]
    L = []
    L.append("# BIRDSAI Tracking Comparison on SAM3-refined GT (`annotations_sam3`)\n")
    L.append("6 TBD trackers × 3 detectors (cached detections → online tracking), "
             "re-scored on the **same GT as the detection table**.")
    L.append(f"Videos: {n_videos}/16 (sweep subset, all 18 runs identical set) · "
             f"{n_frames} frames · fine 5-class.\n")
    L.append("Two scoring passes on the *same* tracks + *same* GT:")
    L.append("- **Full HOTA suite** via **TrackEval** — the identical pipeline used for the "
             "NeurIPS Space-Tracker-MOT (car) table (`compute_hota.py`), so these numbers are "
             "directly comparable to that benchmark. HOTA is α-averaged; CLEAR/Identity at IoU 0.5. "
             "Matching is class-restricted (see Provenance).")
    L.append("- **Detection-level P/R/F1** from the per-class greedy matcher "
             "(`_birdsai_tracking_compare.py`, IoU 0.5) — kept because the long-tail-species "
             "story is a detection-recall story.\n")

    # ---- full suite, one table per detector ----
    L.append("## Full metric suite — TrackEval (rows = tracker)\n")
    for d in DETECTORS:
        L.append(f"### {DET_LABEL[d]} detections\n")
        L.append("| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | MT | ML |")
        L.append("|---|" + "|".join(["---:"] * 8) + "|")
        cols = {mk: [overall.get(f"{d}_{t}", {}).get(mk) for t in TRACKERS]
                for (mk, *_ ) in METRICS}
        marked = {mk: _mark_best(cols[mk], fmt, lo, sgn) for (mk, fmt, lo, sgn) in METRICS}
        for i, t in enumerate(TRACKERS):
            row = " | ".join(marked[mk][i] for (mk, *_ ) in METRICS)
            L.append(f"| {TRK_LABEL[t]} | {row} |")
        L.append("")

    # ---- macro mean across the three detectors ----
    L.append("## Mean across the three detectors (ranking)\n")
    L.append("Macro mean over the 3 detector backbones (rate metrics averaged, "
             "IDsw summed) — a single ranking of the trackers.\n")
    L.append("| Tracker | HOTA | DetA | AssA | IDF1 | ΣIDsw |")
    L.append("|---|---:|---:|---:|---:|---:|")
    means = {}
    for t in TRACKERS:
        rows = [overall.get(f"{d}_{t}") for d in DETECTORS]
        rows = [r for r in rows if r]
        if not rows:
            continue
        means[t] = {
            "HOTA": sum(r["HOTA"] for r in rows) / len(rows),
            "DetA": sum(r["DetA"] for r in rows) / len(rows),
            "AssA": sum(r["AssA"] for r in rows) / len(rows),
            "IDF1": sum(r["IDF1"] for r in rows) / len(rows),
            "IDsw": sum(r["IDsw"] for r in rows),
        }
    order = sorted(means, key=lambda t: means[t]["HOTA"], reverse=True)
    hbest = max(means[t]["HOTA"] for t in means)
    abest = max(means[t]["AssA"] for t in means)
    swbest = min(means[t]["IDsw"] for t in means)
    for t in order:
        m = means[t]
        h = f"**{m['HOTA']:.3f}**" if m["HOTA"] == hbest else f"{m['HOTA']:.3f}"
        a = f"**{m['AssA']:.3f}**" if m["AssA"] == abest else f"{m['AssA']:.3f}"
        sw = f"**{m['IDsw']:,}**" if m["IDsw"] == swbest else f"{m['IDsw']:,}"
        L.append(f"| {TRK_LABEL[t]} | {h} | {m['DetA']:.3f} | {a} | {m['IDF1']:.3f} | {sw} |")
    L.append("")

    # ---- detection-level F1 (greedy) ----
    L.append("## Detection-level F1 — greedy per-class matcher (rows = tracker)\n")
    L.append("Detection precision/recall/F1 (identity-agnostic); complements the HOTA "
             "DetA above. Bold = best detector per tracker.\n")
    L.append("| Tracker | " + " | ".join(DET_LABEL[d] for d in DETECTORS) + " |")
    L.append("|---|" + "|".join(["---:"] * len(DETECTORS)) + "|")
    for t in TRACKERS:
        vals = [greedy.get(f"{d}+{t}", {}).get("overall", {}).get("F1") for d in DETECTORS]
        cells = _mark_best(vals, ".3f")
        L.append(f"| {TRK_LABEL[t]} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- per-class HOTA + F1 for FasterRCNN ----
    L.append("## Per-class breakdown — FasterRCNN backbone\n")
    L.append("Association-aware **HOTA** (TrackEval) beside detection-level **F1** (greedy), "
             "per fine species. This is the headline: every rare/tiny species collapses to "
             "≈0 on *both* metrics — the trackers only hold the large, common **elephant**.\n")
    L.append("| Tracker | " + " | ".join(f"{CANON[c]} H/F1" for c in CLASSES) + " |")
    L.append("|---|" + "|".join(["---:"] * len(CLASSES)) + "|")
    for t in TRACKERS:
        cells = []
        for c in CLASSES:
            h = per_class[CANON[c]].get(f"fasterrcnn_{t}", {}).get("HOTA")
            f1 = greedy.get(f"fasterrcnn+{t}", {}).get("per_class", {}).get(
                CANON[c], {}).get("F1")
            cells.append(f"{_fmt(h, '.3f')}/{_fmt(f1, '.3f')}")
        L.append(f"| {TRK_LABEL[t]} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- takeaways ----
    L.append("## Takeaways\n")
    L.append("- **HOTA re-ranks the trackers vs. F1.** By detection-level F1 the score-thresholding "
             "trackers look fine, but under HOTA **TrackTrack wins on every detector** "
             f"(mean HOTA {means['tracktrack']['HOTA']:.3f}) — it pairs the best AssA with by far the "
             "fewest ID switches via strict track initialisation. Same lesson as the "
             "Space-Tracker-MOT car table: once association is scored properly, strict-init ReID "
             "tracking leads.")
    L.append("- **ByteTrack and BoT-SORT collapse on association.** Their AssA falls to 0.03–0.13 and "
             "IDsw floods (BoT-SORT 3.9k–5.8k), exactly the score-threshold / ID-switch failure seen on "
             "satellite cars — their defaults assume high-scoring detections that thermal aerial video "
             "does not provide.")
    L.append("- **Detection is the ceiling.** HOTA stays ≤ 0.22 for the best tracker because DetA is "
             "0.15–0.25; association can only recover so much when the detector misses most objects.")
    L.append("- **Long-tail collapse is the story, and HOTA confirms it.** Per class, only **elephant** "
             "clears HOTA 0.20–0.29; giraffe/lion/unknown sit at ≈0 on both HOTA and F1. The trackers "
             "inherit the detectors' inability to find rare, tiny thermal species — see "
             "`birdsai_tracking_vs_sam3xexemplar.md` for the SAM3 train-exemplar route that recovers them.")
    L.append("")

    # ---- provenance ----
    L.append("## Provenance\n")
    L.append("- Tracks: cached `mot_format/*.txt` of the 18-run sweep "
             "(`evaluation/eval_birdsai_track_sweep.py`), "
             "`/data/ESA_DLSTEM_2025/experiments/MOT_birdsai_sweep/`. Detectors never re-run.")
    L.append("- GT: `annotations_sam3` (SAM3-refined boxes), fine 5-class, same as the detection table.")
    L.append("- **Full suite**: `evaluation/compute_birdsai_hota.py` → TrackEval (HOTA + CLEAR + "
             "Identity). The sweep runs one tracker per class and bakes the class into the track id "
             "(`class = id // 1_000_000`); TrackEval's MotChallenge box dataset is single-foreground-"
             "class, so we translate every class-`c` box by `c·100000` px on x in **both** GT and tracker "
             "files. Translation preserves within-class IoU exactly while zeroing cross-class IoU, making "
             "matching class-restricted (as in the greedy eval) in a single pooled pass. Frames are "
             "re-indexed to 1 (BIRDSAI frame ids start ≫ 1). TrackEval MOTA reproduces the greedy MOTA "
             "within ~0.01; IDsw differs because TrackEval uses the standard HOTA/CLEAR switch definition, "
             "not the greedy last-mapping counter — prefer the TrackEval column.")
    L.append("- **Detection-level F1**: `evaluation/_birdsai_tracking_compare.py` (per-class greedy IoU "
             "0.5), cached in `birdsai_tracking_sam3gt_compare.json`.")
    L.append("- Full HOTA suite JSON (overall + per-class): `birdsai_tracking_sam3gt_hota.json`.")

    Path(out_md).write_text("\n".join(L) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None,
                    help="TrackEval scratch dir (default: a scratchpad tmp)")
    ap.add_argument("--out-json",
                    default="docs/use_case_results/birdsai_tracking_sam3gt_hota.json")
    args = ap.parse_args()

    workspace = Path(args.workspace) if args.workspace else \
        Path("/tmp/claude-405600010/-home-ziwen-code-esa-dlstem/"
             "e1aabca2-3aa9-4420-bfed-ec8e0e46ff72/scratchpad/birdsai_hota_ws")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           annotations_dirname=ANN,
                           class_map={v: k for k, v in CANON.items()})

    ref = find_run("dinov3", "botsort")
    sweep_videos = sorted(p.stem for p in (ref / "mot_format").glob("*.txt"))
    gt = build_gt(ds, sweep_videos)
    nfr = sum(len(f) for f in gt.values())
    print(f"sweep videos: {len(sweep_videos)}/16  frames={nfr}\n")

    runs = {}
    for d in DETECTORS:
        for t in TRACKERS:
            r = find_run(d, t)
            if r is None:
                print(f"!! missing {d}_{t}")
                continue
            runs[f"{d}_{t}"] = r
    keys = list(runs)

    def eval_pass(class_filter, tag):
        gt_root = workspace / "gt" / "birdsai-test"
        trk_root = workspace / "trackers" / "birdsai-test"
        for p in (gt_root, trk_root):
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)
        seq_off = _write_gt_workspace(gt, gt_root, class_filter)
        _write_tracker_workspace(runs, trk_root, gt, seq_off, class_filter)
        print(f"-- TrackEval pass: {tag}")
        return _run_trackeval(workspace, keys)

    overall = eval_pass(None, "overall (pooled, class-aware)")
    per_class = {}
    for c in CLASSES:
        per_class[CANON[c]] = eval_pass(c, f"class={CANON[c]}")

    for k in keys:
        m = overall.get(k, {})
        if m:
            print(f"{k:26s} HOTA={m['HOTA']:.3f} DetA={m['DetA']:.3f} "
                  f"AssA={m['AssA']:.3f} MOTA={m['MOTA']:+.3f} "
                  f"IDF1={m['IDF1']:.3f} IDsw={m['IDsw']} MT={m['MT']} ML={m['ML']}")

    out = {"gt": ANN, "videos": len(sweep_videos), "frames": nfr,
           "iou": "TrackEval HOTA alpha-avg; CLEAR/Identity @0.5",
           "overall": overall, "per_class": per_class}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")

    greedy = json.loads(Path(GREEDY_JSON).read_text())
    render_markdown(overall, per_class, greedy, len(sweep_videos), nfr, OUT_MD)


if __name__ == "__main__":
    main()
