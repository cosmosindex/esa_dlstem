"""
Post-hoc fix for runs whose MOTFormatDumpCallback dropped videos with
zero predictions. Walks every <run_dir>/mot_format/ and ensures one
.txt file exists for every test-split video_id of the dataset matching
that run's name. Empty files are valid MOTChallenge tracker output (0
predictions) and let compute_hota.py / TrackEval evaluate them as
HOTA=0 / Recall=0 instead of bailing out with "Tracker file not found".

Also touches mot_format_filtered/ if present (RAFT-filtered output).

Usage:
  python tools/fill_missing_mot_format.py \\
      --tracker-output-root /data/.../sam3p1_raft_filtered_<TS>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse compute_hota.py's dataset table so we get the canonical test
# video lists for each registered benchmark.
from compute_hota import _DATASET_TABLE, _build_dataset, _safe_video_id


_DATASET_RE = "|".join(sorted(_DATASET_TABLE.keys(), key=len, reverse=True))


def _resolve_dataset(run_name: str) -> str | None:
    """Pick the longest dataset slug that appears between two underscores
    in the run dir name, e.g. 'sam3p1_text_viso_no_car_20260430_...' →
    'viso_no_car'."""
    parts = run_name
    for ds in sorted(_DATASET_TABLE, key=len, reverse=True):
        if f"_{ds}_" in f"_{parts}_":
            return ds
    return None


def _expected_seqs(dataset_name: str) -> Iterable[str]:
    ds = _build_dataset(dataset_name)
    return [_safe_video_id(v.video_id) for v in ds.videos]


def fill_for_run(run_dir: Path) -> tuple[int, int]:
    """Returns (created_in_mot_format, created_in_filtered)."""
    ds_name = _resolve_dataset(run_dir.name)
    if ds_name is None:
        print(f"  {run_dir.name}: no dataset slug recognised, skipping")
        return 0, 0

    expected = set(_expected_seqs(ds_name))
    n_raw = n_filt = 0
    for sub in ("mot_format", "mot_format_filtered"):
        d = run_dir / sub
        if not d.is_dir():
            continue
        present = {p.stem for p in d.glob("*.txt")}
        missing = expected - present
        for seq in sorted(missing):
            (d / f"{seq}.txt").write_text("")
            if sub == "mot_format":
                n_raw += 1
            else:
                n_filt += 1
        if missing:
            print(f"  {run_dir.name}/{sub}: created {len(missing)} empty "
                  f"files for missing seqs (e.g. {sorted(missing)[:3]}...)")
    return n_raw, n_filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker-output-root", required=True, type=Path)
    args = ap.parse_args()

    total_raw = total_filt = 0
    for run_dir in sorted(args.tracker_output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "mot_format").is_dir():
            continue
        nr, nf = fill_for_run(run_dir)
        total_raw += nr; total_filt += nf

    print(f"\ndone: created {total_raw} raw + {total_filt} filtered empty files")


if __name__ == "__main__":
    main()
