"""
RAFT-based static-tracklet filter for SAM3 (or any) MOT output.

Pipeline:
  1. For each sequence in <run_dir>/mot_format/<seq>.txt, load the source
     frames via the dataset class.
  2. Compute RAFT optical flow between consecutive frames (cached as
     per-tracklet per-frame median |flow| in <run_dir>/raft_track_motion.json
     so that re-running with a new threshold is cheap).
  3. For each tracklet, aggregate the median |flow| over its lifetime
     (max or 80th percentile).
  4. Drop tracklets whose aggregate is <= tau; emit the survivors to
     <run_dir>/mot_format_filtered/<seq>.txt (same MOTChallenge format).

Why median |flow| inside the box:
  - mean is biased by background pixels along box edges
  - median is robust to ~50% mislabelled pixels (over-large boxes)

Usage:
  python tools/raft_filter_tracklets.py \\
      --run-dir /data/.../sam3_text_rscardata_<TS> \\
      --dataset RsCarData \\
      --dataset-root /data/ESA_DLSTEM_2025/data/trafic/RsCarData \\
      --tau 0.5 --agg p80
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Make the vendored RAFT importable, while ensuring our repo's
# ``datasets/`` package wins over RAFT/core/datasets.py (RAFT ships its
# own training-data loader by that name).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAFT_ROOT = _REPO_ROOT / "RAFT"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_RAFT_ROOT / "core") not in sys.path:
    sys.path.append(str(_RAFT_ROOT / "core"))

from raft import RAFT  # noqa: E402
from utils.utils import InputPadder  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


# --------------------------------------------------------------------
# RAFT loader
# --------------------------------------------------------------------

def _load_raft(ckpt_path: str, small: bool = False) -> torch.nn.Module:
    """Load a RAFT checkpoint. The released weights are saved as a
    DataParallel state dict, so we wrap-then-unwrap before .to(DEVICE).

    RAFT.__init__ uses ``'dropout' in args``, which requires a container
    that supports __contains__ — argparse.Namespace works on Py3.9+.
    """
    import argparse
    args = argparse.Namespace(
        small=small,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.module.to(DEVICE).eval()
    return model


# --------------------------------------------------------------------
# Dataset access
# --------------------------------------------------------------------

def _build_dataset(name: str, root: str):
    from datasets import VISODataset, AIRMOTDataset
    from datasets.rscardata import RsCarDataset
    from datasets.satmtb import SATMTBDataset
    from datasets.sdmcar import SDMCarDataset

    name = name.lower()
    if name == "viso":
        return VISODataset(root=root, split="test")
    if name in ("viso_no_car", "visonocar"):
        return VISODataset(
            root=root, split="test",
            categories=("plane", "ship", "train"),
        )
    if name in ("rscardata", "rscar"):
        return RsCarDataset(root=root, split="test", class_map={"car": 0})
    if name == "airmot":
        return AIRMOTDataset(
            root=root, split="test",
            class_map={"airplane": 0, "ship": 1},
        )
    if name == "satmtb":
        return SATMTBDataset(
            root=root, split="test", task="mot", mode="detection",
            class_map={"airplane": 0, "car": 1, "ship": 2, "train": 3},
        )
    if name == "sdmcar":
        return SDMCarDataset(
            root=root, split="test", mode="detection",
            class_map={"car": 0},
        )
    raise ValueError(f"Unknown dataset: {name}")


def _video_lookup(ds) -> dict[str, "VideoInfo"]:  # noqa: F821
    """Map both raw video_id and safe (slash → underscore) form to VideoInfo."""
    out = {}
    for v in ds.videos:
        out[v.video_id] = v
        out[_safe_video_id(v.video_id)] = v
    return out


# --------------------------------------------------------------------
# Flow → per-tracklet motion
# --------------------------------------------------------------------

@torch.no_grad()
def _flow_pair(model, im_a: np.ndarray, im_b: np.ndarray) -> np.ndarray:
    """Compute RAFT flow from im_a (RGB uint8 HWC) to im_b. Returns
    (H, W) magnitude map in float32 pixels-per-frame."""
    a = torch.from_numpy(im_a).permute(2, 0, 1).float()[None].to(DEVICE)
    b = torch.from_numpy(im_b).permute(2, 0, 1).float()[None].to(DEVICE)
    padder = InputPadder(a.shape)
    a, b = padder.pad(a, b)
    _, flow = model(a, b, iters=20, test_mode=True)
    flow = padder.unpad(flow)[0]                # (2, H, W)
    mag = torch.linalg.norm(flow, dim=0)        # (H, W)
    return mag.cpu().numpy().astype(np.float32)


def _tracklet_motions(
    raft: torch.nn.Module,
    video, ds, tracks: dict[int, list[tuple[int, float, float, float, float, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """For each tracklet, return [(frame_id, median |flow| inside box)].

    Frames are loaded once in a streaming fashion: we keep the previous
    frame in memory, compute flow to the current frame, and read off
    box-medians for every tracklet active on the current frame.
    """
    fids_sorted = sorted(video.frame_ids)
    fid_to_idx = {fid: i for i, fid in enumerate(fids_sorted)}

    # Group track rows by frame for quick lookup.
    rows_by_fid: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for tid, rows in tracks.items():
        for (fid, x1, y1, x2, y2, _sc) in rows:
            rows_by_fid[fid].append((tid, x1, y1, x2, y2))

    out: dict[int, list[tuple[int, float]]] = defaultdict(list)

    prev_img = None
    prev_fid = None
    H = W = None

    for fid in fids_sorted:
        cur = ds._load_frame(video, fid)
        if H is None:
            H, W = cur.shape[:2]

        # Need flow for this frame; use t-1 -> t. For the very first
        # frame of the video, fall back to t -> t+1 once available
        # (handled by caching the result at the next step).
        if prev_img is not None:
            mag = _flow_pair(raft, prev_img, cur)
        else:
            mag = None

        if mag is not None and fid in rows_by_fid:
            for tid, x1, y1, x2, y2 in rows_by_fid[fid]:
                ix1 = max(0, int(round(x1)))
                iy1 = max(0, int(round(y1)))
                ix2 = min(W, int(round(x2)))
                iy2 = min(H, int(round(y2)))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                patch = mag[iy1:iy2, ix1:ix2]
                if patch.size == 0:
                    continue
                out[tid].append((fid, float(np.median(patch))))

        prev_img = cur
        prev_fid = fid

    # Backfill: any tracklet present only at frame_ids_sorted[0] would
    # have no flow sample. Compute t0 -> t1 once and read the same boxes.
    if len(fids_sorted) >= 2:
        first_fid = fids_sorted[0]
        if first_fid in rows_by_fid and any(
            (out.get(tid) and out[tid][0][0] != first_fid) or not out.get(tid)
            for tid, *_ in rows_by_fid[first_fid]
        ):
            im0 = ds._load_frame(video, first_fid)
            im1 = ds._load_frame(video, fids_sorted[1])
            mag01 = _flow_pair(raft, im0, im1)
            for tid, x1, y1, x2, y2 in rows_by_fid[first_fid]:
                ix1 = max(0, int(round(x1)))
                iy1 = max(0, int(round(y1)))
                ix2 = min(W, int(round(x2)))
                iy2 = min(H, int(round(y2)))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                patch = mag01[iy1:iy2, ix1:ix2]
                if patch.size == 0:
                    continue
                # Insert at the correct position so the list stays
                # frame-ordered.
                out[tid].insert(0, (first_fid, float(np.median(patch))))

    return out


# --------------------------------------------------------------------
# MOT IO
# --------------------------------------------------------------------

def _read_mot_file(path: Path) -> dict[int, list[tuple[int, float, float, float, float, float]]]:
    """Parse MOTChallenge text → {track_id: [(frame, x1, y1, x2, y2, score)]}.

    Input is xywh (top-left); we convert to xyxy on read to match what
    RAFT expects.
    """
    tracks: dict[int, list[tuple]] = defaultdict(list)
    if not path.exists():
        return tracks
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.split(",")
        fid = int(parts[0])
        tid = int(parts[1])
        x, y, w, h = (float(parts[i]) for i in (2, 3, 4, 5))
        sc = float(parts[6]) if len(parts) > 6 else 1.0
        tracks[tid].append((fid, x, y, x + w, y + h, sc))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def _write_mot_file(
    path: Path, tracks: dict[int, list[tuple[int, float, float, float, float, float]]],
) -> None:
    rows = []
    for tid, lst in tracks.items():
        for fid, x1, y1, x2, y2, sc in lst:
            w, h = x2 - x1, y2 - y1
            rows.append((fid, tid, x1, y1, w, h, sc))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [
        f"{f},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1"
        for (f, i, x, y, w, h, s) in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------
# Aggregators
# --------------------------------------------------------------------

def _agg_fn(name: str) -> Callable[[list[float]], float]:
    if name == "max":
        return lambda vs: float(max(vs)) if vs else 0.0
    if name == "p80":
        return lambda vs: float(np.percentile(vs, 80)) if vs else 0.0
    if name == "mean":
        return lambda vs: float(np.mean(vs)) if vs else 0.0
    raise ValueError(f"Unknown aggregator: {name}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="SAM3 run dir containing mot_format/<seq>.txt")
    ap.add_argument("--dataset", required=True,
                    help="Dataset name: viso_no_car / viso / rscardata")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--raft-ckpt", default="/work/anon/checkpoints/raft/raft-things.pth")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Per-tracklet motion threshold (pixels/frame).")
    ap.add_argument("--agg", choices=("max", "p80", "mean"), default="p80")
    ap.add_argument("--out-subdir", default="mot_format_filtered")
    ap.add_argument("--cache-name", default="raft_track_motion.json",
                    help="Per-tracklet per-frame median |flow| cache "
                         "(written under --run-dir).")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Ignore the cache and re-run RAFT from scratch.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    mot_in_dir = run_dir / "mot_format"
    if not mot_in_dir.is_dir():
        sys.exit(f"missing {mot_in_dir}")
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = run_dir / args.cache_name
    cache: dict[str, dict[int, list[tuple[int, float]]]] = {}

    if cache_path.exists() and not args.force_recompute:
        raw = json.loads(cache_path.read_text())
        # JSON keys are strings; coerce back to int track ids.
        cache = {
            seq: {int(tid): [tuple(x) for x in entries]
                  for tid, entries in tracks.items()}
            for seq, tracks in raw.items()
        }
        print(f"loaded cache: {len(cache)} sequences from {cache_path}")

    need_compute = [
        f for f in sorted(mot_in_dir.glob("*.txt"))
        if f.stem not in cache
    ]

    if need_compute:
        print(f"loading RAFT from {args.raft_ckpt}")
        raft = _load_raft(args.raft_ckpt)
        ds = _build_dataset(args.dataset, args.dataset_root)
        vlut = _video_lookup(ds)

        for f in need_compute:
            seq = f.stem
            tracks = _read_mot_file(f)
            if seq not in vlut:
                print(f"  skip {seq}: not found in dataset video index")
                continue
            video = vlut[seq]
            print(f"  {seq}: {len(tracks)} tracklets, "
                  f"{len(video.frame_ids)} frames")
            cache[seq] = _tracklet_motions(raft, video, ds, tracks)

        # Persist cache (lists of [fid, mag]).
        serial = {
            seq: {str(tid): [list(e) for e in entries]
                  for tid, entries in tracks.items()}
            for seq, tracks in cache.items()
        }
        cache_path.write_text(json.dumps(serial))
        print(f"wrote cache → {cache_path}")
    else:
        print("cache hit for every sequence — skipping RAFT.")

    # ------ Apply threshold ------
    agg = _agg_fn(args.agg)
    n_kept = n_dropped = 0
    summary_rows: list[dict] = []
    for f in sorted(mot_in_dir.glob("*.txt")):
        seq = f.stem
        tracks = _read_mot_file(f)
        motions = cache.get(seq, {})
        kept: dict[int, list] = {}
        for tid, rows in tracks.items():
            mags = [m for (_fid, m) in motions.get(tid, [])]
            score = agg(mags)
            if score > args.tau:
                kept[tid] = rows
                n_kept += 1
            else:
                n_dropped += 1
            summary_rows.append({
                "seq": seq, "track_id": tid,
                "n_frames": len(rows),
                "agg": args.agg,
                "score": round(score, 4),
                "kept": int(score > args.tau),
            })
        _write_mot_file(out_dir / f.name, kept)

    print(f"\nwrote {out_dir}: kept {n_kept}, dropped {n_dropped} "
          f"(tau={args.tau}, agg={args.agg})")

    summary_path = run_dir / f"raft_filter_summary_{args.agg}_tau{args.tau}.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"wrote per-tracklet decisions → {summary_path}")


if __name__ == "__main__":
    main()
