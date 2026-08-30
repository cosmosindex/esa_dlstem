"""
Export the non-car VALIDATION split into MOTIP's dataset layout.

MOTIP was given only train (87) and test (22); the release's 12 validation
sequences were never exported, which is why its checkpoint was taken as "the
last epoch" rather than selected. This builds the missing split so that
checkpoint choice can be measured on held-out data that is not the test split.

Ground truth is produced by ``compute_hota_multiclass._write_gt_for_class``,
the same function that writes the test-split GT every published number is
scored against, so the two are constructed identically and differ only in which
sequences they contain. Images are symlinked from the JDE export, which is the
only tree holding validation frames.

Usage:
    python tools/export_motip_val.py --out MOTIP/datasets/SpaceTracker/val \
        --workspace /tmp/hota_nocar_val_ws
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import compute_hota_multiclass as chm

JDE_IMAGES = Path("/data/ESA_DLSTEM_2025/data/spacetracker_jde_nocar/"
                  "spacetracker_nocar/images")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workspace", required=True, type=Path,
                    help="TrackEval workspace to leave the GT in, for scoring")
    args = ap.parse_args()

    bench = "spacetracker_nocar_val_all"
    gt_root = args.workspace / "gt" / f"{bench}-test"
    if gt_root.exists():
        shutil.rmtree(gt_root)
    gt_root.mkdir(parents=True)
    seqs, _ = chm._write_gt_for_class("spacetracker_nocar_val", "all", gt_root)
    print(f"{len(seqs)} validation sequences with non-car GT")

    args.out.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    for seq in seqs:
        src_img = JDE_IMAGES / seq
        if not src_img.is_dir():
            raise SystemExit(f"no images for {seq} at {src_img}")
        frames = sorted(src_img.glob("*.png"))
        w, h = Image.open(frames[0]).size

        d = args.out / seq
        (d / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(gt_root / seq / "gt" / "gt.txt", d / "gt" / "gt.txt")
        # MOTIP addresses frames as %08d (data/spacetracker.py); the JDE export
        # names them %06d, so img1 is a directory of per-frame links rather than
        # one link to the source directory. Verified above that GT frame numbers
        # run 1..N contiguously, so the k-th sorted image is frame k.
        img1 = d / "img1"
        if img1.is_symlink():
            img1.unlink()
        if img1.is_dir():
            shutil.rmtree(img1)
        img1.mkdir()
        for k, f in enumerate(frames, start=1):
            (img1 / f"{k:08d}{f.suffix}").symlink_to(f)
        # Real per-sequence size: MOTIP resizes by the longer side, so a wrong
        # imWidth/imHeight would silently change the inference scale.
        (d / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={seq}\n"
            "imDir=img1\n"
            "frameRate=30\n"
            f"seqLength={len(frames)}\n"
            f"imWidth={w}\n"
            f"imHeight={h}\n"
            "imExt=.png\n"
        )
        n_frames += len(frames)
        print(f"  {seq:28s} {len(frames):5d} frames  {w}x{h}")

    seqmap = args.workspace / "seqmaps"
    seqmap.mkdir(parents=True, exist_ok=True)
    (seqmap / f"{bench}-test.txt").write_text("name\n" + "\n".join(seqs) + "\n")
    print(f"[done] {len(seqs)} seqs, {n_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
