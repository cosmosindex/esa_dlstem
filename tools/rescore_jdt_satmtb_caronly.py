"""Re-score the ALREADY-DUMPED all-class JDT track files on SAT-MTB with the
car-only protocol used by every TBD row in the master MOT CSV.

No inference is re-run: this only re-feeds
``allclass_20260608/<tracker>_satmtb_<TS>/mot_format/*.txt`` to TrackEval
against a car-only GT / seqmap (the 21 ``car_*`` sequences), exactly the
restriction that ``compute_hota.py`` applies for the SORT-family runs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import compute_hota as ch

# The all-class eval lifted SAT-MTB to 4 classes; the car-only benchmark
# protocol keeps car only. ``class_map`` alone only gates which boxes are
# surfaced — the video list has to be restricted with ``categories``, or the
# airplane/ship/train sequences stay in the seqmap with empty GT and every
# prediction there is scored as a false positive.
ch._SAM3_CLASS_MAPS["satmtb"] = {"car": 0}
_cls, _root, _extra = ch._DATASET_TABLE["satmtb"]
ch._DATASET_TABLE["satmtb"] = (_cls, _root, {**_extra, "categories": ["car"]})

ROOT = Path("/data/ESA_DLSTEM_2025/experiments/MOT/allclass_20260608")
WS = Path("/tmp/hota_ws_jdt_satmtb_caronly")
OUT = _REPO_ROOT / "NeurIPS 2026" / "tables" / "MOT" / "_jdt_satmtb_caronly.csv"

runs = {}
for d in sorted(ROOT.iterdir()):
    if d.is_dir() and "_satmtb_" in d.name:
        tracker = d.name.split("_satmtb_")[0]
        runs[tracker] = d
print("runs:", {k: v.name for k, v in runs.items()})

WS.mkdir(parents=True, exist_ok=True)
metrics = ch._eval_one_dataset("satmtb", WS, runs)

seqmap = (WS / "seqmaps" / "satmtb-test.txt").read_text().splitlines()
print(f"seqmap: {len(seqmap) - 1} sequences -> {seqmap[1:6]} ...")

fields = ["dataset", "tracker", "HOTA", "DetA", "AssA", "LocA", "MOTA",
          "MOTP", "IDF1", "IDsw", "MT", "ML", "n_dets"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for tr in sorted(metrics):
        m = metrics[tr]
        row = {"dataset": "satmtb", "tracker": tr}
        row.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in m.items()})
        w.writerow(row)
        print(f"{tr:14s} HOTA={m['HOTA']:.4f} DetA={m['DetA']:.4f} "
              f"AssA={m['AssA']:.4f} MOTA={m['MOTA']:.4f} "
              f"IDF1={m['IDF1']:.4f} IDsw={m['IDsw']} "
              f"MT={m['MT']} ML={m['ML']}")
print(f"[ok] wrote {OUT}")
