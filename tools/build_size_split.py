"""
Split every Space-tracker dataset (SOT + MOT) into a *small-object* and a
*large-object* half.

A sequence lands in the **small** bucket when the median ``sqrt(w * h)`` of
its GT boxes is <= 32 px (COCO's small-object threshold), otherwise in
**large**. The median — rather than "contains at least one small box" — is
what keeps sequences whose annotations merely jitter across the 32 px line
out of the small bucket; on this data ~11 MOT sequences differ between the
two rules, e.g. ``airmot/47`` has 6 small boxes out of 2696.

Two artefacts are produced:

1. **The split manifest** (``size_split.json``) — the source of truth:
   ``seq_id -> {bucket, median_sqrt_area_px, n_boxes, ...}``.
2. **A symlink tree** (optional, ``--link-root``) — a browsable view of the
   manifest where each bucket is a drop-in dataset root::

       <link-root>/OOTB/small/car_1 -> /data/.../OOTB/car_1

   Symlinks carry absolute targets, so the tree can live anywhere and
   deleting it never touches the source data.

Usage::

    # manifest only
    python tools/build_size_split.py --out docs/size_split/size_split.json

    # manifest + symlink tree, reusing precomputed per-sequence stats
    python tools/build_size_split.py \
        --out docs/size_split/size_split.json \
        --stats-csv /tmp/size_split/per_sequence_size.csv \
        --link-root /data/ESA_DLSTEM_2025/data/trafic/_size_split
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.manifest import Manifest
from space_tracker.manifest_mot import MOTManifest
from tools.analyze_size_split import (
    MOT_ROOTS,
    REPO,
    SMALL_THRESH,
    SOT_ROOTS,
    analyse_mot,
    analyse_sot,
)

ROOTS = {**SOT_ROOTS, **MOT_ROOTS}

# Directory name each dataset gets inside the symlink tree.
LINK_DIRNAME = {
    "ootb": "OOTB", "satsot": "SatSOT", "sv248s": "SV248S",
    "airmot": "AIR-MOT-100", "satmtb": "SAT-MTB", "viso": "VISO",
    "sdmcar": "SDM-Car", "rscardata": "RsCarData",
}

# Root-relative paths that are not part of any single sequence but that the
# dataset class still reads (attribute tables, official split sheets, ...).
# Linked into *both* buckets so each bucket stays a valid dataset root.
SHARED_ENTRIES = {
    "ootb":      ["anno"],                              # per-sequence attribute files
    "satmtb":    ["SAT-MTB_Dataset/data_split.xlsx"],   # official train/test sheet
    "rscardata": ["annotations"],                       # split-level COCO/MOT JSONs
}

SV248S_ANN_EXTS = (".rect", ".state", ".abs", ".poly", ".attr")


# ---------------------------------------------------------------------------
# Per-dataset link plans: sequence -> [(path_relative_to_bucket, abs_source)]
# ---------------------------------------------------------------------------

def _plan_sot(seq) -> list[tuple[str, Path]]:
    root = ROOTS[seq.dataset]
    if seq.dataset in ("ootb", "satsot"):
        # Self-contained sequence directory: <vid>/{img,groundtruth.txt}
        return [(seq.video_id, root / seq.video_id)]

    if seq.dataset == "sv248s":
        # Imagery and annotations live in two sibling trees under <group>/.
        group, sid = seq.video_id.split("/", 1)
        out = [(f"{group}/sequences/{sid}", root / group / "sequences" / sid)]
        for ext in SV248S_ANN_EXTS:
            src = root / group / "annotations" / f"{sid}{ext}"
            if src.exists():
                out.append((f"{group}/annotations/{sid}{ext}", src))
        return out

    raise ValueError(f"no link plan for SOT dataset {seq.dataset!r}")


def _plan_mot(seq) -> list[tuple[str, Path]]:
    root = ROOTS[seq.dataset]

    if seq.dataset == "airmot":
        return [(seq.video_id, root / seq.video_id)]

    if seq.dataset in ("satmtb", "viso"):
        # Sequence dir is the grandparent of the image pattern:
        #   SAT-MTB_Dataset/airplane/02/img/{...}.png -> SAT-MTB_Dataset/airplane/02
        #   mot/plane/039/img/{...}.jpg               -> mot/plane/039
        rel = str(Path(seq.image_path_pattern).parent.parent)
        return [(rel, root / rel)]

    if seq.dataset == "sdmcar":
        # One .avi + one .csv per sequence — files, not directories.
        return [(seq.video_path, root / seq.video_path),
                (seq.gt_path,    root / seq.gt_path)]

    if seq.dataset == "rscardata":
        rel_img = str(Path(seq.image_path_pattern).parent.parent)   # images/<split>/<seq>
        out = [(rel_img, root / rel_img)]
        if seq.gt_format == "pascal_voc_xml_per_frame":
            # Per-sequence XML directory. (The train/val COCO JSON is a
            # split-level file and comes in via SHARED_ENTRIES instead.)
            out.append((seq.gt_path, root / seq.gt_path))
        return out

    raise ValueError(f"no link plan for MOT dataset {seq.dataset!r}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def load_stats(stats_csv: Path | None) -> list[dict]:
    if stats_csv is not None:
        with open(stats_csv) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["n_boxes"] = int(r["n_boxes"])
            for k in ("sqrt_area_median", "frac_small_area", "frac_small_side",
                      "sqrt_area_p05", "sqrt_area_p95", "sqrt_area_min",
                      "sqrt_area_max"):
                if k in r:
                    r[k] = float(r[k])
        return rows

    print("== computing per-sequence size stats (SOT) ==")
    rows = analyse_sot(REPO / "space_tracker" / "space_tracker.json")
    print("== computing per-sequence size stats (MOT) ==")
    rows += analyse_mot(REPO / "space_tracker" / "space_tracker_mot.json")
    return rows


def bucket_of(row: dict) -> str:
    return "small" if row["sqrt_area_median"] <= SMALL_THRESH else "large"


# ---------------------------------------------------------------------------
# Symlink tree
# ---------------------------------------------------------------------------

def build_links(link_root: Path, assignments: dict[str, str],
                dry_run: bool = False) -> None:
    """``assignments`` maps seq_id -> bucket."""
    sot = {s.id: s for s in Manifest.load(REPO / "space_tracker" / "space_tracker.json").sequences}
    mot = {s.id: s for s in MOTManifest.load(REPO / "space_tracker" / "space_tracker_mot.json").sequences}

    n_links = Counter()
    missing: list[str] = []
    nonempty: set[tuple[str, str]] = set()      # (dataset, bucket) with >= 1 sequence

    for seq_id, bucket in sorted(assignments.items()):
        if seq_id in sot:
            seq, entries = sot[seq_id], _plan_sot(sot[seq_id])
        else:
            seq, entries = mot[seq_id], _plan_mot(mot[seq_id])
        nonempty.add((seq.dataset, bucket))
        bucket_dir = link_root / LINK_DIRNAME[seq.dataset] / bucket

        for rel, src in entries:
            if not src.exists():
                missing.append(f"{seq_id}: {src}")
                continue
            dst = bucket_dir / rel
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
            n_links[seq.dataset] += 1

    # Dataset-level files each bucket also needs to be a valid root.
    for ds in sorted({d for d, _ in nonempty}):
        for rel in SHARED_ENTRIES.get(ds, []):
            src = ROOTS[ds] / rel
            if not src.exists():
                missing.append(f"[shared] {ds}: {src}")
                continue
            for bucket in ("small", "large"):
                if (ds, bucket) not in nonempty:     # bucket empty for this dataset
                    continue
                dst = link_root / LINK_DIRNAME[ds] / bucket / rel
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src, dst)
                n_links[ds] += 1

    print(f"\n{'symlinks' if not dry_run else 'symlinks (dry-run)'} under {link_root}:")
    for ds, n in sorted(n_links.items()):
        print(f"  {LINK_DIRNAME[ds]:<14} {n:>5}")
    if missing:
        print(f"  [warn] {len(missing)} sources missing:")
        for m in missing[:10]:
            print(f"    {m}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="path of the split manifest JSON to write")
    ap.add_argument("--stats-csv", type=Path, default=None,
                    help="reuse per_sequence_size.csv from analyze_size_split.py")
    ap.add_argument("--link-root", type=Path, default=None,
                    help="also build a symlink tree under this directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --link-root: report what would be linked, create nothing")
    args = ap.parse_args()

    rows = load_stats(args.stats_csv)
    assignments = {r["seq_id"]: bucket_of(r) for r in rows}

    per_seq = {
        r["seq_id"]: {
            "task": r["task"],
            "dataset": r["dataset"],
            "category": r["category"],
            "bucket": bucket_of(r),
            "median_sqrt_area_px": r["sqrt_area_median"],
            "n_boxes": r["n_boxes"],
            "frac_small_area": r["frac_small_area"],
        }
        for r in sorted(rows, key=lambda r: (r["task"], r["dataset"], r["seq_id"]))
    }

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"small": 0, "large": 0})
    for r in rows:
        counts[f"{r['task']}/{r['dataset']}"][bucket_of(r)] += 1

    out = {
        "version": "1.0",
        "name": "space-tracker-size-split",
        "description": (
            "Per-sequence small / large object split for the Space-tracker SOT "
            "and MOT manifests."
        ),
        "criterion": {
            "metric": "median_sqrt_area_px",
            "definition": "median over all GT boxes in the sequence of sqrt(w * h) in pixels",
            "threshold_px": SMALL_THRESH,
            "small_if": f"median_sqrt_area_px <= {SMALL_THRESH}",
            "note": (
                "COCO's small-object threshold (area <= 32*32). Sequence-level "
                "assignment by the median keeps sequences that merely jitter "
                "across the threshold out of the small bucket."
            ),
        },
        "source_manifests": ["space_tracker.json", "space_tracker_mot.json"],
        "counts": dict(sorted(counts.items())),
        "n_sequences": len(per_seq),
        "sequences": per_seq,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}  ({len(per_seq)} sequences)")
    print(f"\n{'task/dataset':<20}{'small':>7}{'large':>7}")
    print("-" * 34)
    for k, v in out["counts"].items():
        print(f"{k:<20}{v['small']:>7}{v['large']:>7}")
    tot_s = sum(v["small"] for v in counts.values())
    print("-" * 34)
    print(f"{'TOTAL':<20}{tot_s:>7}{len(per_seq) - tot_s:>7}")

    if args.link_root is not None:
        build_links(args.link_root, assignments, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
