"""
Per-attribute (sequence, frame) counts for SatSOT / SV248S / OOTB.

Counts are computed from each dataset's native sequence-attribute flags + the
``num_frames`` of each video. For the *shared* unified attributes (BC, IV,
ROT, OCC, SOB, DEF) we report the sum across the datasets that annotate
them. For *unique* native attributes (POC, FOC, STO, LTO, CO, ARC, OON, LQ,
TO, BJT, BCH, ND, IBG/BCL, SM, LT, MB, IM, AM) we report the count from the
single annotating dataset.

Output: ``<root>/analysis/unified_attributes/attribute_counts.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ootb import OOTBDataset
from datasets.satsot import SatSOTDataset
from datasets.sv248s import SV248SDataset


DATASET_ROOTS = {
    "ootb":   "/data/ESA_DLSTEM_2025/data/trafic/OOTB",
    "satsot": "/data/ESA_DLSTEM_2025/data/trafic/SatSOT",
    "sv248s": "/data/ESA_DLSTEM_2025/data/trafic/SV248S",
}

# Unified shared attributes — used by Table 1.
SHARED_ATTR_MAP = {
    "BC":  {"satsot": ["BC"],         "sv248s": [],                 "ootb": ["BC"]},
    "IV":  {"satsot": ["IV"],         "sv248s": ["IV"],             "ootb": ["IV"]},
    "ROT": {"satsot": ["ROT"],        "sv248s": ["IPR"],            "ootb": ["IPR"]},
    "OCC": {"satsot": ["POC", "FOC"], "sv248s": ["STO", "LTO", "CO"], "ootb": ["PO", "FO"]},
    "SOB": {"satsot": ["SOB"],        "sv248s": ["DS"],             "ootb": ["SA"]},
    "DEF": {"satsot": ["DEF"],        "sv248s": [],                 "ootb": ["DEF"]},
}
SHARED_ORDER = ["BC", "IV", "ROT", "OCC", "SOB", "DEF"]

# Native (dataset-unique) attributes — used by Table 2.
# (label as printed in the table, {dataset: [native attr names]}). POC and
# FOC are pooled across SatSOT (POC/FOC) and OOTB (PO/FO), mirroring the
# OCC unification in the main attribute table; STO/LTO/CO stay SV248S-only
# because no other dataset annotates the temporal axis.
UNIQUE_ATTRS = [
    ("POC", {"satsot": ["POC"], "ootb": ["PO"]}),
    ("FOC", {"satsot": ["FOC"], "ootb": ["FO"]}),
    ("STO", {"sv248s": ["STO"]}),
    ("LTO", {"sv248s": ["LTO"]}),
    ("CO",  {"sv248s": ["CO"]}),
    ("ARC", {"satsot": ["ARC"]}),
    ("OON", {"ootb":   ["OON"]}),
    ("LQ",  {"satsot": ["LQ"]}),
    ("TO",  {"satsot": ["TO"]}),
    ("BJT", {"satsot": ["BJT"]}),
    ("BCH", {"sv248s": ["BCH"]}),
    ("ND",  {"sv248s": ["ND"]}),
    ("IBG", {"sv248s": ["BCL"]}),   # IBG (renamed in our taxonomy) ≡ SV248S BCL
    ("SM",  {"sv248s": ["SM"]}),
    ("LT",  {"ootb":   ["LT"]}),
    ("MB",  {"ootb":   ["MB"]}),
    ("IM",  {"ootb":   ["IM"]}),
    ("AM",  {"ootb":   ["AM"]}),
]


def load_datasets():
    return {
        "ootb":   OOTBDataset(  root=DATASET_ROOTS["ootb"],   split="no_split", mode="detection"),
        "satsot": SatSOTDataset(root=DATASET_ROOTS["satsot"], split="no_split", mode="detection"),
        "sv248s": SV248SDataset(root=DATASET_ROOTS["sv248s"], split="no_split", mode="detection"),
    }


def per_video_frames(dataset_obj) -> dict[str, int]:
    return {v.video_id: int(v.num_frames) for v in dataset_obj.videos}


def attr_counts(
    seq_attrs: dict[str, list[str]],
    frame_counts: dict[str, int],
    natives: list[str],
) -> tuple[int, int]:
    """(#sequences, #frames) for videos whose attr list intersects ``natives``."""
    if not natives:
        return 0, 0
    target = set(natives)
    n_seq = 0
    n_frames = 0
    for vid, attrs in seq_attrs.items():
        if target & set(attrs):
            n_seq += 1
            n_frames += frame_counts.get(vid, 0)
    return n_seq, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/work/ziwen/experiments/NeurIPS/SOT_whole_dataset_04_22/"
                "analysis/unified_attributes/attribute_counts.csv",
    )
    args = ap.parse_args()

    print("[1/3] loading datasets")
    ds = load_datasets()
    seq_attrs = {name: d.sequence_attributes() for name, d in ds.items()}
    frame_counts = {name: per_video_frames(d) for name, d in ds.items()}

    print(f"   ootb  : {len(seq_attrs['ootb'])} seqs, "
          f"{sum(frame_counts['ootb'].values())} frames")
    print(f"   satsot: {len(seq_attrs['satsot'])} seqs, "
          f"{sum(frame_counts['satsot'].values())} frames")
    print(f"   sv248s: {len(seq_attrs['sv248s'])} seqs, "
          f"{sum(frame_counts['sv248s'].values())} frames")

    rows = []

    print("\n[2/3] shared attributes (sum across annotating datasets)")
    for attr in SHARED_ORDER:
        ds_natives = SHARED_ATTR_MAP[attr]
        per_ds = {}
        for ds_name in ("satsot", "sv248s", "ootb"):
            per_ds[ds_name] = attr_counts(
                seq_attrs[ds_name], frame_counts[ds_name], ds_natives[ds_name],
            )
        total_seq    = sum(s for s, _ in per_ds.values())
        total_frames = sum(f for _, f in per_ds.values())
        rows.append({
            "table": "shared",
            "attr":  attr,
            "satsot_seq":    per_ds["satsot"][0], "satsot_frames": per_ds["satsot"][1],
            "sv248s_seq":    per_ds["sv248s"][0], "sv248s_frames": per_ds["sv248s"][1],
            "ootb_seq":      per_ds["ootb"][0],   "ootb_frames":   per_ds["ootb"][1],
            "total_seq":     total_seq,           "total_frames":  total_frames,
        })
        print(f"   {attr:3s}: SatSOT {per_ds['satsot'][0]:3d}/{per_ds['satsot'][1]:6d}"
              f"  SV248S {per_ds['sv248s'][0]:3d}/{per_ds['sv248s'][1]:7d}"
              f"  OOTB {per_ds['ootb'][0]:3d}/{per_ds['ootb'][1]:6d}"
              f"  =>  {total_seq} seqs / {total_frames} frames")

    print("\n[3/3] unique native attributes (pooled across annotating datasets)")
    for label, ds_natives in UNIQUE_ATTRS:
        per_ds = {ds_name: (0, 0) for ds_name in ("satsot", "sv248s", "ootb")}
        for ds_name, natives in ds_natives.items():
            per_ds[ds_name] = attr_counts(
                seq_attrs[ds_name], frame_counts[ds_name], natives,
            )
        total_seq    = sum(s for s, _ in per_ds.values())
        total_frames = sum(f for _, f in per_ds.values())
        rows.append({
            "table": "unique",
            "attr":  label,
            "satsot_seq":    per_ds["satsot"][0], "satsot_frames": per_ds["satsot"][1],
            "sv248s_seq":    per_ds["sv248s"][0], "sv248s_frames": per_ds["sv248s"][1],
            "ootb_seq":      per_ds["ootb"][0],   "ootb_frames":   per_ds["ootb"][1],
            "total_seq":     total_seq,
            "total_frames":  total_frames,
        })
        ds_names = ",".join(sorted(ds_natives.keys()))
        print(f"   {label:3s} ({ds_names:14s}): "
              f"SatSOT {per_ds['satsot'][0]:3d}/{per_ds['satsot'][1]:6d}"
              f"  SV248S {per_ds['sv248s'][0]:3d}/{per_ds['sv248s'][1]:7d}"
              f"  OOTB {per_ds['ootb'][0]:3d}/{per_ds['ootb'][1]:6d}"
              f"  =>  {total_seq} seqs / {total_frames} frames")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
