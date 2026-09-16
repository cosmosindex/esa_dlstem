"""
Run TrackEval over an ALREADY-BUILT HOTA workspace.

`compute_hota_multiclass.py` rebuilds its workspace from the original tracker
run directories on every call -- it `rmtree`s `gt/` and `trackers/` first. That
is right when those run directories exist, and useless when they do not: the
non-car per-class workspace still holds complete, correctly-formatted data for
all seven class-predicting trackers, but two of the source run directories
(botsort_reid, tracktrack) are gone, so re-running the normal path would delete
recoverable results rather than reproduce them.

This scores whatever is already laid out under

    <workspace>/gt/<benchmark>-test/…
    <workspace>/trackers/<benchmark>-test/<tracker>/data/*.txt
    <workspace>/seqmaps/<benchmark>-test.txt

using the identical TrackEval configuration and the identical metric extraction
as `_eval_dataset_class`, so numbers produced here are interchangeable with
numbers produced there. It never writes into gt/ or trackers/.

Usage:
    python tools/score_hota_workspace.py \
        --workspace /path/to/hota_ws4 \
        --benchmarks space_tracker_nocar_airplane space_tracker_nocar_ship \
        --out docs/space_tracker/hota_nocar_perclass.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def score(workspace: Path, benchmark: str, split: str = "test") -> dict[str, dict]:
    import trackeval

    troot = workspace / "trackers" / f"{benchmark}-{split}"
    if not troot.is_dir():
        raise SystemExit(f"no tracker data at {troot}")
    trackers = sorted(d.name for d in troot.iterdir() if (d / "data").is_dir())
    seqmap = workspace / "seqmaps" / f"{benchmark}-{split}.txt"
    n_seqs = sum(1 for i, l in enumerate(seqmap.read_text().splitlines())
                 if i and l.strip())
    print(f"[{benchmark}] {len(trackers)} trackers, {n_seqs} sequences: "
          f"{', '.join(trackers)}")

    out: dict[str, dict] = {}
    for tr_name in trackers:
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                         "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                         "BREAK_ON_ERROR": False, "RETURN_ON_ERROR": True})
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER":        str(workspace / "gt"),
            "TRACKERS_FOLDER":  str(workspace / "trackers"),
            "OUTPUT_FOLDER":    str(workspace / "output"),
            "TRACKERS_TO_EVAL": [tr_name],
            "CLASSES_TO_EVAL":  ["pedestrian"],
            "BENCHMARK":        benchmark,
            "SPLIT_TO_EVAL":    split,
            "PRINT_CONFIG":     False,
            "DO_PREPROC":       False,
            "SEQMAP_FOLDER":    str(workspace / "seqmaps"),
            "SEQMAP_FILE":      str(seqmap),
            "SKIP_SPLIT_FOL":   False,
        })
        metrics_list = [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        try:
            raw, _ = trackeval.Evaluator(eval_cfg).evaluate(
                [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics_list)
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
        hota, clear, ident = cs["HOTA"], cs["CLEAR"], cs["Identity"]
        out[tr_name] = {
            "HOTA": float(np.mean(hota["HOTA"])),
            "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])),
            "LocA": float(np.mean(hota["LocA"])),
            "MOTA": float(clear["MOTA"]), "MOTP": float(clear["MOTP"]),
            "IDF1": float(ident["IDF1"]), "IDsw": int(clear["IDSW"]),
            "MT": int(clear["MT"]), "ML": int(clear["ML"]),
            "n_dets": int(clear["CLR_TP"]) + int(clear["CLR_FP"]),
            "n_seqs": n_seqs,
        }
        print(f"  {tr_name:14s} HOTA={out[tr_name]['HOTA']:.4f} "
              f"DetA={out[tr_name]['DetA']:.4f} AssA={out[tr_name]['AssA']:.4f} "
              f"IDF1={out[tr_name]['IDF1']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--dataset", default="space_tracker_nocar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for b in args.benchmarks:
        cls = b[len(args.dataset) + 1:] if b.startswith(args.dataset + "_") else b
        for tr, vals in score(Path(args.workspace), b).items():
            rows.append({"dataset": args.dataset, "tracker": tr,
                         "class": cls, **vals})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
