"""Score the non-car (airplane / ship / train) MOT half on the TEST split only,
for the tracking-by-detection trackers and the 4-class union JDT models together.

Why a separate pass instead of reusing `tracker_satmtb_hbb_*/hota_summary.csv`:
that CSV scores VISO and AIR-MOT on **all** sequences, which is fair for Faster
R-CNN (it never saw either dataset) but not for FairMOT/TGraM — `union_all.json`
trains on `airmot.train` and `viso_no_car.train`, so scoring them over all
sequences would score them on their own training data. Restricting both families
to the test split puts them on identical sequences.

No inference is re-run: the TbD trackers already dumped per-class tracks for
every sequence, and `compute_hota_multiclass` only scores sequences that are in
the GT seqmap, so switching the split is enough to drop the train/val ones.

Usage::

    python tools/hota_nocar_testsplit.py \
        --jdt-root  /data/.../MOT/nocar_perclass_20260811 \
        --tbd-root  /data/.../MOT/tracker_satmtb_hbb_20260505_183921 \
        --output    "NeurIPS 2026/tables/MOT/_nocar_testsplit_hota.csv"
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import compute_hota_multiclass as chm
from project_paths import DATA_ROOT

DEFAULT_JDT_ROOT = Path(
    f"{DATA_ROOT}/experiments/MOT/nocar_perclass_20260811")
DEFAULT_TBD_ROOT = Path(
    f"{DATA_ROOT}/experiments/MOT/tracker_satmtb_hbb_20260505_183921")
DEFAULT_OUTPUT = (_REPO_ROOT / "NeurIPS 2026" / "tables" / "MOT"
                  / "_nocar_testsplit_hota.csv")


def _force_test_split() -> None:
    """VISO / AIR-MOT default to 'no_split' in compute_hota_multiclass; the JDT
    models trained on those sequences, so pin every dataset to its test split."""
    for name, entry in list(chm._DATASET_TABLE.items()):
        cls, root, extra, _split, cmap = entry
        chm._DATASET_TABLE[name] = (cls, root, extra, "test", cmap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jdt-root", default=str(DEFAULT_JDT_ROOT))
    ap.add_argument("--tbd-root", default=str(DEFAULT_TBD_ROOT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--workspace", default="/tmp/hota_ws_nocar_testsplit")
    args = ap.parse_args()

    _force_test_split()

    # One flat root of symlinks so _collect_runs sees both families at once.
    merged = Path(tempfile.mkdtemp(prefix="nocar_runs_"))
    n = 0
    for root in (Path(args.tbd_root), Path(args.jdt_root)):
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not chm._RUN_RE.match(d.name):
                continue
            (merged / d.name).symlink_to(d)
            n += 1
    print(f"[merged] {n} run dirs -> {merged}")

    runs = chm._collect_runs(merged)
    if not runs:
        sys.exit(f"no runs matched under {merged}")
    print("Runs found:")
    for ds in sorted(runs):
        for tr in sorted(runs[ds]):
            print(f"  {ds}/{tr} → {runs[ds][tr].name}")

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ds_name, runs_for_ds in runs.items():
        cmap = chm._DATASET_TABLE[ds_name][4]
        for class_name in cmap:
            print(f"\n=== {ds_name} / {class_name} (test split) ===")
            metrics = chm._eval_dataset_class(
                ds_name, class_name, workspace, runs_for_ds)
            for tr_name, vals in metrics.items():
                rows.append({"dataset": ds_name, "tracker": tr_name,
                             "class": class_name, **vals})
                print(f"  {tr_name:14s} HOTA={vals['HOTA']:.3f} "
                      f"DetA={vals['DetA']:.3f} AssA={vals['AssA']:.3f} "
                      f"IDF1={vals['IDF1']:.3f} n_seqs={vals['n_seqs']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "tracker", "class", "HOTA", "DetA", "AssA", "LocA",
              "MOTA", "MOTP", "IDF1", "IDsw", "MT", "ML", "n_dets", "n_seqs"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n{len(rows)} rows → {out_path}")
    shutil.rmtree(merged, ignore_errors=True)


if __name__ == "__main__":
    main()
