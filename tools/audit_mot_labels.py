"""
Audit MOT category labels by rendering one crop per *track*, ranked by how
suspicious the label looks.

Motivation: SAT-MTB labels 450 boxes in `car/18` as ``airplane`` when they are
plainly ships. Auditing 300k boxes is hopeless; auditing tracks is not — the
whole of SAT-MTB is 9,604 tracks, only 461 of them non-car, and a track's
category is constant by construction. Reviewing a contact sheet of 461 crops
is an afternoon, not a project.

Each track contributes its largest-area frame (the most legible view), cropped
with context and drawn with its GT box. Tiles are sorted most-suspicious-first
so errors surface in the first sheet.

Suspicion signals (cheap, no model needed):
  * label is rare for the directory it lives in (`airplane` inside `car/`)
  * size far from the median of its own class (a 6 px "airplane")
  * aspect ratio far from its class median

Output: `sheet_XX.jpg` contact sheets + `tracks.csv` with a blank
``corrected_label`` column to fill in. Feed the filled CSV to
``--emit-overrides`` to get a label-override JSON; the raw dataset is never
modified.

Usage::

    python tools/audit_mot_labels.py --dataset satmtb --skip-label car \\
        --out-dir /work/anon/experiments/label_audit
    python tools/audit_mot_labels.py --dataset satmtb \\
        --out-dir ... --emit-overrides reviewed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from space_tracker.data_mot import _enumerate_frame_paths, _parse_gt
from space_tracker.manifest_mot import MOTManifest
from tools.analyze_size_split import MOT_ROOTS, REPO

COLOURS = {"car": (0, 255, 0), "airplane": (0, 165, 255),
           "ship": (255, 0, 255), "train": (255, 255, 0)}


def collect_tracks(dataset: str, skip_labels: set[str]) -> list[dict]:
    man = MOTManifest.load(REPO / "space_tracker" / "space_tracker_mot.json")
    out: list[dict] = []
    for seq in (s for s in man.sequences if s.dataset == dataset):
        ann = _parse_gt(seq, MOT_ROOTS[dataset])
        per: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
        for fid, objs in ann.items():
            for o in objs:
                if o.category in skip_labels:
                    continue
                per[(o.category, o.track_id)].append((fid, o.bbox_xyxy))
        for (label, tid), obs in per.items():
            b = np.stack([x for _, x in obs]).astype(float)
            w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            area = w * h
            best = int(np.argmax(area))
            out.append({
                "seq_id": seq.id, "video_id": seq.video_id,
                "dir_category": seq.video_id.split("/")[0],
                "label": label, "track_id": tid, "n_obs": len(obs),
                "median_size": float(np.median(np.sqrt(area))),
                "median_ar": float(np.median(np.maximum(w, h) /
                                             np.maximum(np.minimum(w, h), 1e-6))),
                "best_frame": obs[best][0],
                "best_box": obs[best][1].tolist(),
            })
    return out


def score(tracks: list[dict]) -> None:
    """Attach a 'suspicion' score and a human-readable reason to each track."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for t in tracks:
        by_label[t["label"]].append(t)
    stats = {}
    for lab, ts in by_label.items():
        s = np.array([t["median_size"] for t in ts])
        a = np.array([t["median_ar"] for t in ts])
        stats[lab] = (np.median(s), np.std(s) + 1e-6, np.median(a), np.std(a) + 1e-6)

    # how often does this label appear inside this directory category?
    freq: dict[tuple[str, str], int] = defaultdict(int)
    per_dir: dict[str, int] = defaultdict(int)
    for t in tracks:
        freq[(t["dir_category"], t["label"])] += 1
        per_dir[t["dir_category"]] += 1

    for t in tracks:
        ms, ss, ma, sa = stats[t["label"]]
        z_size = abs(t["median_size"] - ms) / ss
        z_ar = abs(t["median_ar"] - ma) / sa
        rarity = 1.0 - freq[(t["dir_category"], t["label"])] / per_dir[t["dir_category"]]
        reasons = []
        if t["label"] != t["dir_category"]:
            reasons.append(f"{t['label']} inside {t['dir_category']}/")
        if z_size > 1.5:
            reasons.append(f"size {t['median_size']:.0f}px vs class median {ms:.0f}px")
        if z_ar > 1.5:
            reasons.append(f"aspect {t['median_ar']:.1f} vs {ma:.1f}")
        t["suspicion"] = round(float(2.0 * rarity + z_size + 0.5 * z_ar), 3)
        t["reason"] = "; ".join(reasons)


def _crop(dataset: str, t: dict, ctx: float, tile: int,
          seqs: dict[str, object], path_cache: dict[str, dict]) -> np.ndarray | None:
    seq = seqs[t["video_id"]]
    if seq.image_format != "frames":
        return None
    if t["video_id"] not in path_cache:
        path_cache[t["video_id"]] = dict(_enumerate_frame_paths(seq, MOT_ROOTS[dataset]))
    paths = path_cache[t["video_id"]]
    p = paths.get(t["best_frame"])
    if p is None:
        return None
    im = cv2.imread(str(p))
    if im is None:
        return None
    x1, y1, x2, y2 = t["best_box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(max(x2 - x1, y2 - y1) * ctx / 2, 30)
    X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
    X1, Y1 = int(min(im.shape[1], cx + half)), int(min(im.shape[0], cy + half))
    sub = im[Y0:Y1, X0:X1].copy()
    if sub.size == 0:
        return None
    sc = tile / max(sub.shape[0], sub.shape[1])
    sub = cv2.resize(sub, (int(sub.shape[1] * sc), int(sub.shape[0] * sc)),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((tile, tile, 3), np.uint8)
    canvas[:sub.shape[0], :sub.shape[1]] = sub
    col = COLOURS.get(t["label"], (255, 255, 255))
    cv2.rectangle(canvas, (int((x1 - X0) * sc), int((y1 - Y0) * sc)),
                  (int((x2 - X0) * sc), int((y2 - Y0) * sc)), col, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (tile - 1, 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{t['video_id']} id{t['track_id']}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{t['label']} {t['median_size']:.0f}px s={t['suspicion']:.1f}",
                (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
    return canvas


def emit_overrides(reviewed_csv: Path, out_json: Path) -> None:
    """Turn a reviewed CSV into {seq_id: {track_id: corrected_label}}."""
    ov: dict[str, dict[str, str]] = defaultdict(dict)
    n = 0
    with open(reviewed_csv) as f:
        for r in csv.DictReader(f):
            new = (r.get("corrected_label") or "").strip()
            if not new or new == r["label"]:
                continue
            ov[r["seq_id"]][str(r["track_id"])] = new
            n += 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"description":
                   "Per-track GT category corrections; apply on load, "
                   "source data is unmodified.",
                   "corrections": dict(ov)}, f, indent=1)
    print(f"wrote {out_json}: {n} corrected tracks across {len(ov)} sequences")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(MOT_ROOTS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-label", action="append", default=[],
                    help="skip a label entirely, e.g. --skip-label car")
    ap.add_argument("--top", type=int, default=None,
                    help="render only the N most suspicious tracks")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--tile", type=int, default=180)
    ap.add_argument("--context", type=float, default=4.0,
                    help="crop side = context x box side")
    ap.add_argument("--emit-overrides", type=Path, default=None,
                    help="reviewed CSV -> label_overrides.json, then exit")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.emit_overrides:
        emit_overrides(args.emit_overrides, args.out_dir / "label_overrides.json")
        return

    tracks = collect_tracks(args.dataset, set(args.skip_label))
    score(tracks)
    tracks.sort(key=lambda t: -t["suspicion"])
    print(f"{len(tracks)} tracks to audit (skipped labels: {args.skip_label or 'none'})")

    fields = ["seq_id", "video_id", "dir_category", "label", "track_id", "n_obs",
              "median_size", "median_ar", "best_frame", "suspicion", "reason"]
    with open(args.out_dir / "tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["corrected_label"])
        w.writeheader()
        for t in tracks:
            w.writerow({**{k: t[k] for k in fields}, "corrected_label": ""})
    print(f"wrote {args.out_dir/'tracks.csv'}")

    todo = tracks[:args.top] if args.top else tracks
    man = MOTManifest.load(REPO / "space_tracker" / "space_tracker_mot.json")
    seqs = {s.video_id: s for s in man.sequences if s.dataset == args.dataset}
    path_cache: dict[str, dict] = {}
    per_sheet = args.cols * args.rows
    sheet = []
    n_sheet = 0
    for i, t in enumerate(todo):
        c = _crop(args.dataset, t, args.context, args.tile, seqs, path_cache)
        if c is None:
            continue
        sheet.append(c)
        if len(sheet) == per_sheet or i == len(todo) - 1:
            while len(sheet) < per_sheet:
                sheet.append(np.zeros((args.tile, args.tile, 3), np.uint8))
            grid = np.vstack([np.hstack(sheet[r * args.cols:(r + 1) * args.cols])
                              for r in range(args.rows)])
            out = args.out_dir / f"sheet_{n_sheet:02d}.jpg"
            cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  {out}")
            sheet = []
            n_sheet += 1
    print(f"\nReview the sheets, fill 'corrected_label' in tracks.csv, then:\n"
          f"  python tools/audit_mot_labels.py --dataset {args.dataset} "
          f"--out-dir {args.out_dir} --emit-overrides {args.out_dir/'tracks.csv'}")


if __name__ == "__main__":
    main()
