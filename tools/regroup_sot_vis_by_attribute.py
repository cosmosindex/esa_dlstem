"""
Regroup flat SOT visualizations into per-attribute subfolders (in place).

Existing layout:
    <run_dir>/visualizations/<vid>_frame<fid>.jpg     # flat

Target layout:
    <run_dir>/visualizations/<attr>/<vid>_frame<fid>.jpg
    <run_dir>/visualizations/_no_attr/<file>           # videos without attrs

The first attribute folder gets the real file (via `os.rename`); the rest
are hardlinked to the same inode so disk usage does not multiply.

Usage:
    python tools/regroup_sot_vis_by_attribute.py /work/anon/experiments/NeurIPS/SOT_0420
    python tools/regroup_sot_vis_by_attribute.py <root> --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.ootb import OOTBDataset
from datasets.satsot import SatSOTDataset
from datasets.sv248s import SV248SDataset

# Matches the tracker-run naming convention — run dirs end in a dataset tag
# plus a timestamp.
_DATASET_TAG = re.compile(r"_(ootb|satsot|sv248s)_\d{8}_\d{6}$", re.IGNORECASE)

_FRAME_RE = re.compile(r"^(.+)_frame\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)

_DATASET_SPEC = {
    "ootb":   (OOTBDataset,   "/data/ESA_DLSTEM_2025/data/trafic/OOTB"),
    "satsot": (SatSOTDataset, "/data/ESA_DLSTEM_2025/data/trafic/SatSOT"),
    "sv248s": (SV248SDataset, "/data/ESA_DLSTEM_2025/data/trafic/SV248S"),
}

_NO_ATTR_FOLDER = "_no_attr"

# Cache: tag -> {video_id: [attr,...]}
_ATTR_CACHE: dict[str, dict[str, list[str]]] = {}


def _attrs_for(tag: str) -> dict[str, list[str]]:
    """Return {video_id: [attr, ...]} with both `a/b` and `a_b` key forms.

    The eval callback writes filenames with `/` replaced by `_` (see
    `SAM2VisualizationCallback._save_vis`); we mirror that so lookups on
    the parsed filename (which can't recover the original `/`) still hit.
    """
    tag = tag.lower()
    if tag in _ATTR_CACHE:
        return _ATTR_CACHE[tag]
    cls, root = _DATASET_SPEC[tag]
    ds = cls(root=root, split="no_split", mode="video", clip_len=1, clip_stride=1)
    raw = ds.sequence_attributes()
    expanded: dict[str, list[str]] = {}
    for vid, attrs in raw.items():
        expanded[vid] = attrs
        safe = vid.replace("/", "_")
        if safe != vid:
            expanded[safe] = attrs
    _ATTR_CACHE[tag] = expanded
    return expanded


def _parse_video_id(filename: str) -> str | None:
    m = _FRAME_RE.match(filename)
    return m.group(1) if m else None


def _detect_tag(run_dir: Path) -> str | None:
    m = _DATASET_TAG.search(run_dir.name)
    return m.group(1).lower() if m else None


def _looks_already_grouped(vis_dir: Path) -> bool:
    # If any direct child is a directory, assume already regrouped.
    return any(p.is_dir() for p in vis_dir.iterdir())


def regroup_run(run_dir: Path, dry_run: bool = False) -> dict:
    tag = _detect_tag(run_dir)
    vis_dir = run_dir / "visualizations"
    stats = {
        "run": str(run_dir),
        "tag": tag,
        "status": "ok",
        "moved": 0,
        "hardlinked": 0,
        "unmatched": 0,
    }
    if tag is None:
        stats["status"] = "skipped_no_tag"
        return stats
    if not vis_dir.is_dir():
        stats["status"] = "skipped_no_vis"
        return stats
    if _looks_already_grouped(vis_dir):
        stats["status"] = "skipped_already_grouped"
        return stats

    attr_map = _attrs_for(tag)

    for jpg in sorted(vis_dir.iterdir()):
        if not jpg.is_file():
            continue
        vid = _parse_video_id(jpg.name)
        if vid is None:
            stats["unmatched"] += 1
            continue

        attrs = attr_map.get(vid) or []
        folders = attrs if attrs else [_NO_ATTR_FOLDER]

        primary = vis_dir / folders[0] / jpg.name
        if dry_run:
            stats["moved"] += 1
            stats["hardlinked"] += max(0, len(folders) - 1)
            continue

        primary.parent.mkdir(parents=True, exist_ok=True)
        os.rename(jpg, primary)
        stats["moved"] += 1

        for folder in folders[1:]:
            sub = vis_dir / folder
            sub.mkdir(parents=True, exist_ok=True)
            link_path = sub / jpg.name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                os.link(primary, link_path)
            except OSError:
                os.symlink(os.path.relpath(primary, sub), link_path)
            stats["hardlinked"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Root under which to scan for <tracker>/<run>/visualizations/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs: list[Path] = []
    for tracker_dir in sorted(args.root.iterdir()):
        if not tracker_dir.is_dir():
            continue
        for run_dir in sorted(tracker_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "visualizations").is_dir():
                runs.append(run_dir)

    print(f"Found {len(runs)} run dirs under {args.root}")
    if args.dry_run:
        print("(dry-run — no files will be moved)")

    for run in runs:
        stats = regroup_run(run, dry_run=args.dry_run)
        print(
            f"[{stats['status']:<22}] {stats['tag'] or '?'}  "
            f"moved={stats['moved']:>5}  hardlinked={stats['hardlinked']:>6}  "
            f"unmatched={stats['unmatched']:>3}  {Path(stats['run']).name}"
        )


if __name__ == "__main__":
    main()
