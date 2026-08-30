"""
MASA (CVPR 2024) on the Space-Tracker MOT release, non-car classes.

MASA is *not* a detector and *not* a joint detect-and-track model: it is a
detector-agnostic association adapter. The MASA-R50 "plug and play" variant
takes whatever boxes you already have and produces instance embeddings plus
track assignments, which is exactly the tracking-by-detection contract used by
SORT / ByteTrack / OC-SORT / BoT-SORT / TrackTrack here. So it reads the SAME
cached Faster R-CNN detections as every other TBD tracker, and any difference in
the numbers is association behaviour alone.

This lives outside the usual eval_tracker_multiclass.py driver because MASA
needs mmdet 3.3.0 / mmcv 2.1.0 / torch 2.1, which cannot coexist with the
project environment's torch 2.10. It runs in the separate `masa` env and writes
results in the same format the other trackers produce, so downstream HOTA /
aggregation is unchanged.

Usage (in the masa env):
    python evaluation/eval_masa_spacetracker.py \
        --det-cache /path/to/spacetracker_nocar_dets_cache/spacetracker_nocar \
        --out-dir   /path/to/out/masa_spacetracker_nocar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "masa"))

def _stub_eval_deps():
    """
    MASA's package __init__ imports its BDD/TAO evaluation metrics, which pull in
    `scalabel` and `teta`. Neither is on PyPI and neither is used for inference --
    we score with our own HOTA pipeline. A plain sys.modules stub is not enough
    because the metrics do `from scalabel.eval.hota import ...`, which requires a
    real package structure, so we install a meta-path finder that fabricates any
    submodule under those two roots on demand.
    """
    import importlib.abc
    import importlib.machinery
    import types
    from unittest.mock import MagicMock

    ROOTS = ("scalabel", "teta")

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []                       # make it importable as a package
            mod.__getattr__ = lambda name: MagicMock()   # PEP 562
            return mod

        def exec_module(self, module):
            pass

    class _StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ROOTS:
                return importlib.machinery.ModuleSpec(
                    fullname, _StubLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _StubFinder())


_stub_eval_deps()

from masa.apis import inference_masa, init_masa, build_test_pipeline  # noqa: E402
from project_paths import DATA_ROOT


def _safe_video_id(video_id: str) -> str:
    return video_id.replace("/", "_")


def _frames_of(release_root: Path, video_name: str, category: str,
               frame_ids: list[int]):
    """Yield (frame_id, BGR image) straight from the release MOTChallenge tree."""
    import configparser
    import cv2

    seq_dir = release_root / "mot" / category / video_name
    ext = ".png"
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        ext = cp.get("Sequence", "imExt", fallback=".png")
    for fid in frame_ids:
        img = cv2.imread(str(seq_dir / "img1" / f"{fid:06d}{ext}"))
        if img is None:
            raise FileNotFoundError(seq_dir / "img1" / f"{fid:06d}{ext}")
        yield fid, img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--release", default=f"{DATA_ROOT}/release/space_tracker")
    ap.add_argument("--masa-config",
                    default=str(_PROJECT_ROOT / "masa" / "configs" / "masa-one"
                                / "masa_r50_plug_and_play.py"))
    ap.add_argument("--masa-checkpoint",
                    default=str(_PROJECT_ROOT / "masa" / "saved_models"
                                / "masa_models" / "masa_r50.pth"))
    ap.add_argument("--score-floor", type=float, default=0.10,
                    help="Detections below this are dropped before association. "
                         "Matches the floor used for the score-cascade trackers.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    det_dir = Path(args.det_cache)
    out_dir = Path(args.out_dir)
    (out_dir / "mot_format").mkdir(parents=True, exist_ok=True)
    release = Path(args.release)

    # Which category subdirectory each released sequence lives in.
    ann = json.loads((release / "mot" / "annotations"
                      / "space_tracker_mot.json").read_text())
    cat_of = {v["name"]: v["category"] for v in ann["videos"]}

    print(f"[setup] masa cfg={args.masa_config}")
    model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    pipeline = build_test_pipeline(model.cfg)

    det_files = sorted(det_dir.glob("*.json"))
    print(f"[setup] {len(det_files)} sequences")

    t_all = time.time()
    for vi, det_path in enumerate(det_files, 1):
        vid = det_path.stem
        d = json.loads(det_path.read_text())
        frame_ids = d["frame_ids"]
        category = cat_of.get(vid)
        if category is None:
            print(f"  [{vi}/{len(det_files)}] {vid}: not in release, skip")
            continue

        rows = []
        t0 = time.time()
        for k, (fid, frame) in enumerate(
                _frames_of(release, vid, category, frame_ids)):
            boxes = np.asarray(d["boxes"][k], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(d["scores"][k], dtype=np.float32)
            # HiEUM is single-class and writes no label array.
            labels = (np.asarray(d["labels"][k], dtype=np.int64)
                      if "labels" in d
                      else np.ones(len(boxes), dtype=np.int64))
            keep = scores >= args.score_floor
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes):
                # inference_masa wants [x1,y1,x2,y2,score] and 0-based labels.
                det_bboxes = torch.from_numpy(
                    np.concatenate([boxes, scores[:, None]], axis=1)
                ).to(args.device)
                det_labels = torch.from_numpy(labels - 1).to(args.device)
            else:
                det_bboxes = torch.zeros((0, 5), device=args.device)
                det_labels = torch.zeros((0,), dtype=torch.long, device=args.device)

            track_result = inference_masa(
                model, frame, frame_id=k, video_len=len(frame_ids),
                test_pipeline=pipeline,
                det_bboxes=det_bboxes, det_labels=det_labels,
            )
            inst = track_result[0].pred_track_instances
            tb = inst.bboxes.cpu().numpy()
            tid = inst.instances_id.cpu().numpy()
            tsc = inst.scores.cpu().numpy()
            tlb = inst.labels.cpu().numpy() + 1        # back to 1-based
            for (x1, y1, x2, y2), i, s, l in zip(tb, tid, tsc, tlb):
                rows.append(f"{fid},{int(i)},{x1:.2f},{y1:.2f},"
                            f"{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},{int(l)},-1")

        (out_dir / "mot_format" / f"{_safe_video_id(vid)}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""))
        print(f"  [{vi}/{len(det_files)}] {vid}: {len(rows)} track rows  "
              f"{time.time() - t0:.1f}s")

    print(f"\ndone in {time.time() - t_all:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
