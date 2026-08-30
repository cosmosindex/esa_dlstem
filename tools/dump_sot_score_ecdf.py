"""
Dump the full empirical distribution of each SOT tracker's confidence scores.

`analyse_sot_tau.py` only kept five order statistics per tracker, which is
enough for a summary bar but not enough to choose *how* to draw the
distribution, or to draw an ECDF. This walks the same tau=0 dumps and writes:

* `<prefix>_quantiles.csv` -- a full quantile ladder, so we can check whether a
  standard Tukey box plot would degenerate on these scores before committing to
  one (it does: the IQR of most trackers is narrower than a line width).
* `<prefix>_ecdf.csv` -- F(x) on a fine grid. The ECDF is the honest object
  here: read at x=tau it *is* the fraction of boxed frames that tau removes,
  which is exactly the quantity panel (c) reports, so the two panels stop being
  separate claims.

Usage:
    python tools/dump_sot_score_ecdf.py --runs /work/anon/experiments/SOT_tau0 \
        --release /data/ESA_DLSTEM_2025/release/space_tracker \
        --out docs/space_tracker/sot_tau
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.analyse_sot_tau import DATASETS, find_run, load_records
from tools.make_wacv_sot_table import released_sequences

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
# Dense near 1.0: that is where seven of nine trackers keep all their mass, and
# a uniform grid would render them as a single vertical jump at the right edge.
GRID = np.unique(np.concatenate([
    np.linspace(0.0, 0.9, 181),
    np.linspace(0.9, 1.0, 201),
]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    released = released_sequences(Path(args.release))

    keys = {}
    for p in runs_root.glob("*/per_image_metrics.json"):
        stem = p.parent.name.split("_first_frame_")[0]
        keys.setdefault(stem.split("_")[0], stem)

    q_rows, e_rows = [], []
    for key in sorted(keys):
        scores, n_frames, n_declared = [], 0, 0
        for ds in DATASETS:
            pim = find_run(runs_root, key, ds)
            if pim is None:
                continue
            seqs = load_records(pim, set(released[ds]), rescore_aabb=(ds == "ootb"))
            for recs in seqs.values():
                for _, sc, has_box in recs:
                    n_frames += 1
                    if not has_box:
                        n_declared += 1
                    elif sc is not None:
                        scores.append(sc)
        if not scores:
            print(f"  [{key}] no scores, skipped")
            continue
        a = np.asarray(scores, dtype=np.float64)
        qs = np.quantile(a, QUANTILES)
        row = {"tracker": key, "n_frames": n_frames, "n_boxed": len(a),
               "n_declared": n_declared}
        row.update({f"q{int(q*100):02d}": round(float(v), 5)
                    for q, v in zip(QUANTILES, qs)})
        row["iqr"] = round(float(qs[6] - qs[4]), 5)
        row["frac_below_0.5"] = round(float((a < 0.5).mean()), 5)
        q_rows.append(row)

        # F(x) over boxed frames, and over ALL frames (the latter is what
        # panel (c)'s "filtered" bar counts, so keep both explicit).
        f_boxed = np.searchsorted(np.sort(a), GRID, side="left") / len(a)
        for x, f in zip(GRID, f_boxed):
            e_rows.append({"tracker": key, "x": round(float(x), 5),
                           "F_boxed": round(float(f), 6),
                           "F_all": round(float(f * len(a) / n_frames), 6)})
        print(f"  [{key}] n={len(a)}  IQR={row['iqr']:.5f} "
              f"(Q1={qs[4]:.4f}, Q3={qs[6]:.4f})  P(score<0.5)={row['frac_below_0.5']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in [("quantiles", q_rows), ("ecdf", e_rows)]:
        pth = out.parent / f"{out.name}_{name}.csv"
        with open(pth, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {pth} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
