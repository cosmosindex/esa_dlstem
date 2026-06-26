#!/usr/bin/env python
"""SOT combined visualisations: GT (green) + 7 tracker preds, one image per frame.

For every frame across the 21 SOT-eval runs (7 trackers × 3 datasets) at
``--run-root``, compose a single overlay image:

  • ground-truth box / OBB polygon in **green**
  • each tracker's prediction in a distinct BGR colour
  • bottom legend strip mapping tracker → colour

Output layout follows Space-tracker-SOT taxonomy:

  visualizations/<paper-attr>/<dataset>_<video_id>_frame<NNNN>.jpg
  visualizations/OCC/<sub-type>/<dataset>_<video_id>_frame<NNNN>.jpg

18 top-level paper attributes + OCC's 5 sub-types nested under OCC/.
First matching folder for a frame gets the real file; the others receive
hardlinks (with symlink fallback), so total disk ≈ 1 × raw frame count.

Reads ``per_image_metrics.json`` from each run dir (must contain
``pred_box`` / ``gt_box`` / ``pred_poly`` / ``gt_poly`` — see
``lightning_modules/visualization.py``).
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


TRACKERS = ["siamrpn", "ostrack", "odtrack", "lorat", "sam2", "samurai", "sam3"]
DATASETS = ["ootb", "satsot", "sv248s"]

# BGR colours, distinct hues, all avoid GT green.
TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
    "siamrpn": (0, 165, 255),     # orange
    "ostrack": (255, 144, 30),    # dodger blue
    "odtrack": (0, 215, 255),     # gold
    "lorat":   (0, 0, 255),       # red
    "sam2":    (147, 20, 255),    # deep pink
    "samurai": (255, 255, 0),     # cyan
    "sam3":    (211, 0, 148),     # purple
}
GT_COLOR = (0, 255, 0)  # green (BGR)

DATASET_ROOTS = {
    "ootb":   Path("/data/ESA_DLSTEM_2025/data/trafic/OOTB"),
    "satsot": Path("/data/ESA_DLSTEM_2025/data/trafic/SatSOT"),
    "sv248s": Path("/data/ESA_DLSTEM_2025/data/trafic/SV248S"),
}

OCC_SUBTYPES = {"POC", "FOC", "STO", "LTO", "CO"}


# ----------------------------------------------------------------------------
# Index building
# ----------------------------------------------------------------------------

def find_run_dir(root: Path, tracker: str, dataset: str) -> Path | None:
    """Return the latest run dir under ``root/<tracker>`` for ``dataset``
    that actually has a ``per_image_metrics.json`` (i.e. completed run).

    An in-progress run dir exists before its JSON is dumped; skip those so
    the compositor doesn't false-miss a tracker whose redundant rerun is
    still spinning while the original completed copy is sitting next to it.
    """
    cands = sorted((root / tracker).glob(f"*_{dataset}_*"))
    for cand in reversed(cands):
        if (cand / "per_image_metrics.json").exists():
            return cand
    return None


def build_per_frame_index(run_root: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Merge per-frame records from all 21 runs.

    Returns ``(index, missing)`` where
      index[(dataset, video_id, frame_id)] = {
        'gt_box': [...], 'gt_poly': [...] or None,
        'preds': {tracker: {'pred_box':..., 'pred_poly':..., 'best_iou':...}}
      }
    """
    index: dict[tuple, dict] = {}
    missing: list[tuple[str, str]] = []
    for dataset in DATASETS:
        for tracker in TRACKERS:
            run_dir = find_run_dir(run_root, tracker, dataset)
            if run_dir is None:
                missing.append((dataset, tracker))
                continue
            json_path = run_dir / "per_image_metrics.json"
            if not json_path.exists():
                missing.append((dataset, tracker))
                continue
            with open(json_path) as f:
                data = json.load(f)
            for rec in data:
                vid = rec["video_id"]
                fid = int(rec["frame_id"])
                key = (dataset, vid, fid)
                entry = index.setdefault(
                    key, {"gt_box": None, "gt_poly": None, "preds": {}}
                )
                for sr in rec["sot_records"]:
                    if entry["gt_box"] is None and sr.get("gt_box") is not None:
                        entry["gt_box"] = sr["gt_box"]
                        entry["gt_poly"] = sr.get("gt_poly")
                    entry["preds"][tracker] = {
                        "pred_box": sr.get("pred_box"),
                        "pred_poly": sr.get("pred_poly"),
                        "best_iou": sr.get("best_iou"),
                    }
    return index, missing


# ----------------------------------------------------------------------------
# Image-path resolution
# ----------------------------------------------------------------------------

def resolve_image_paths(dataset: str, image_dir: str, video_id: str,
                        frame_ids: list[int]) -> dict[int, Path | None]:
    """Resolve {frame_id: Path} for all requested frames of one sequence."""
    root = DATASET_ROOTS[dataset]
    seq_dir = root / image_dir
    out: dict[int, Path | None] = {}
    if dataset in ("ootb", "satsot"):
        for fid in frame_ids:
            out[fid] = seq_dir / f"{fid + 1:04d}.jpg"
        return out
    # sv248s: glob once, index by sorted order
    files = sorted(seq_dir.glob("*.tiff"))
    for fid in frame_ids:
        out[fid] = files[fid] if 0 <= fid < len(files) else None
    return out


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

def _draw_poly(img: np.ndarray, poly: list[float], colour, thickness: int = 2) -> None:
    pts = np.asarray(poly).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=colour, thickness=thickness)


def _draw_box(img: np.ndarray, box: list[int], colour, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)


def _render_legend(width: int, row_height: int = 26) -> np.ndarray:
    """Bottom legend strip. Wraps to multiple rows if the canvas is too narrow.

    Each item occupies ~75 px (swatch + label). For narrow OOTB frames we use
    2–4 rows; for SV248S's 1024-wide images everything fits on one row.
    """
    items = [("GT", GT_COLOR)] + [(t, TRACKER_COLORS[t]) for t in TRACKERS]
    n = len(items)
    pad = 6
    item_w = 75
    cols = max(1, min(n, (width - 2 * pad) // item_w))
    rows = (n + cols - 1) // cols
    height = rows * row_height + 4
    strip = np.full((height, width, 3), 40, dtype=np.uint8)
    actual_item_w = (width - 2 * pad) // cols
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (name, colour) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * actual_item_w
        y_top = r * row_height + 4
        cv2.rectangle(strip, (x, y_top + 4), (x + 18, y_top + row_height - 4),
                      colour, -1)
        cv2.rectangle(strip, (x, y_top + 4), (x + 18, y_top + row_height - 4),
                      (220, 220, 220), 1)
        cv2.putText(strip, name, (x + 24, y_top + row_height - 7),
                    font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return strip


def compose_frame(image: np.ndarray, gt_poly, gt_box, preds: dict[str, dict]) -> np.ndarray:
    canvas = image.copy()
    if gt_poly:
        _draw_poly(canvas, gt_poly, GT_COLOR)
    elif gt_box:
        _draw_box(canvas, gt_box, GT_COLOR)
    for tracker in TRACKERS:
        rec = preds.get(tracker)
        if rec is None:
            continue
        colour = TRACKER_COLORS[tracker]
        if rec.get("pred_poly"):
            _draw_poly(canvas, rec["pred_poly"], colour)
        elif rec.get("pred_box"):
            _draw_box(canvas, rec["pred_box"], colour)
    legend = _render_legend(canvas.shape[1])
    return np.vstack([canvas, legend])


# ----------------------------------------------------------------------------
# Folder placement
# ----------------------------------------------------------------------------

def folders_for_seq(taxonomy_attrs: list[str], paper_attrs: set[str]) -> list[str]:
    folders: list[str] = []
    for a in taxonomy_attrs:
        if a in paper_attrs and a not in folders:
            folders.append(a)
        elif a in OCC_SUBTYPES:
            sub = f"OCC/{a}"
            if sub not in folders:
                folders.append(sub)
    if not folders:
        folders = ["_no_attr"]
    return folders


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------

def _read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def render_one(payload: tuple) -> str | None:
    (img_path, dataset, video_id, frame_id,
     gt_box, gt_poly, preds, folders, output_root) = payload
    if img_path is None:
        return f"NO_PATH {dataset}/{video_id}/{frame_id}"
    img_path = Path(img_path)
    img = _read_image(img_path)
    if img is None:
        return f"BAD_READ {img_path}"

    vis = compose_frame(img, gt_poly, gt_box, preds)

    safe_vid = f"{dataset}_{video_id.replace('/', '_')}"
    filename = f"{safe_vid}_frame{frame_id:04d}.jpg"

    output_root = Path(output_root)
    primary_dir = output_root / folders[0]
    primary_dir.mkdir(parents=True, exist_ok=True)
    primary_path = primary_dir / filename
    cv2.imwrite(str(primary_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    for folder in folders[1:]:
        sub = output_root / folder
        sub.mkdir(parents=True, exist_ok=True)
        link_path = sub / filename
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            os.link(primary_path, link_path)
        except OSError:
            os.symlink(os.path.relpath(primary_path, sub), link_path)
    return None


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def write_readme(root: Path, run_root: str, paper_attrs: set[str]) -> None:
    colour_lines = "\n".join(
        f"| {t} | {TRACKER_COLORS[t]} |" for t in TRACKERS
    )
    paper_attr_block = "  ".join(sorted(paper_attrs))
    body = f"""# SOT combined visualisations

Source runs: ``{run_root}``

Each image overlays one frame with **ground truth in green** and each
tracker's prediction in a distinct colour:

| Tracker | Colour (BGR) |
|---------|--------------|
| GT      | (0, 255, 0)  |
{colour_lines}

## Layout

18 paper-taxonomy top-level attributes:

    {paper_attr_block}

OCC's 5 sub-types are nested under ``OCC/``:

    OCC/POC  OCC/FOC  OCC/STO  OCC/LTO  OCC/CO

A frame in a sequence with attributes ``[BC, ROT, OCC, POC]`` lands in
``BC/``, ``ROT/``, ``OCC/`` and ``OCC/POC/`` — the first folder gets the
real file, others receive hardlinks (symlink fallback), so total disk ≈
1× the raw frame count.

Filename: ``{{dataset}}_{{video_id}}_frame{{NNNN}}.jpg`` (SV248S's ``/``
replaced with ``_``).

## Provenance

- Per-frame predictions read from each run's ``per_image_metrics.json``
  (schema includes ``pred_box`` / ``pred_poly`` / ``gt_box`` /
  ``gt_poly``, added by ``lightning_modules/visualization.py::_sot_draw``
  and ``_sot_draw_obb``).
- 18 paper-taxonomy attrs + 5 OCC sub-types come from
  ``space_tracker/space_tracker.json``::``attribute_taxonomy``.
"""
    (root / "README.md").write_text(body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",
                   default="/home/anon/code/esa_dlstem/space_tracker/space_tracker.json")
    p.add_argument("--run-root", required=True,
                   help="e.g. /work/anon/experiments/NeurIPS/SOT_whole_dataset_20260518")
    p.add_argument("--output", required=True,
                   help="e.g. /data/ESA_DLSTEM_2025/experiments/SOT_combined_viz_20260518")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap frames (smoke test)")
    args = p.parse_args()

    manifest = json.load(open(args.manifest))
    paper_attrs = (
        set(manifest["attribute_taxonomy"]["groups"]["shared"]["members"])
        | set(manifest["attribute_taxonomy"]["groups"]["aspect_ratio"]["members"])
        | set(manifest["attribute_taxonomy"]["groups"]["dataset_unique_other"]["members"])
    )
    assert len(paper_attrs) == 18, f"expected 18 paper attrs, got {len(paper_attrs)}"

    seq_meta = {(s["dataset"], s["video_id"]): s for s in manifest["sequences"]}

    print(f"[index] building from {args.run_root}")
    index, missing = build_per_frame_index(Path(args.run_root))
    print(f"[index] frames: {len(index)}    missing runs: {len(missing)}")
    for d, t in missing:
        print(f"  MISSING {d}/{t}")

    out_root = Path(args.output) / "visualizations"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve image paths per-sequence (batched glob for SV248S).
    by_seq: dict[tuple, list[int]] = {}
    for (dataset, vid, fid) in index:
        by_seq.setdefault((dataset, vid), []).append(fid)
    img_paths: dict[tuple, Path | None] = {}
    for (dataset, vid), fids in by_seq.items():
        meta = seq_meta.get((dataset, vid))
        if meta is None:
            for fid in fids:
                img_paths[(dataset, vid, fid)] = None
            continue
        resolved = resolve_image_paths(dataset, meta["image_dir"], vid, fids)
        for fid, path in resolved.items():
            img_paths[(dataset, vid, fid)] = path

    # Build work list.
    work: list[tuple] = []
    keys = list(index.keys())
    if args.limit is not None:
        keys = keys[: args.limit]
    skipped_no_meta = 0
    for key in keys:
        dataset, vid, fid = key
        entry = index[key]
        meta = seq_meta.get((dataset, vid))
        if meta is None:
            skipped_no_meta += 1
            continue
        folders = folders_for_seq(meta["taxonomy_attrs"], paper_attrs)
        work.append((
            str(img_paths.get(key)) if img_paths.get(key) is not None else None,
            dataset, vid, fid,
            entry["gt_box"], entry["gt_poly"], entry["preds"],
            folders, str(out_root),
        ))
    if skipped_no_meta:
        print(f"[work] skipped {skipped_no_meta} frames (no manifest entry)")
    print(f"[work] rendering {len(work)} frames with {args.workers} workers")

    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(ex.map(render_one, work, chunksize=64), 1):
            if err:
                errors.append(err)
            if i % 5000 == 0:
                print(f"  {i}/{len(work)}   errors={len(errors)}")

    print(f"[done] errors={len(errors)}")
    for e in errors[:10]:
        print(" ", e)

    write_readme(Path(args.output), args.run_root, paper_attrs)
    print(f"[done] README written to {Path(args.output) / 'README.md'}")


if __name__ == "__main__":
    main()
