"""Stage 2 of the space-tracker release: put the imagery in the tree.

Stage 1 wrote annotations and, alongside them, a staging plan naming every
destination frame and where it comes from.  This script consumes that plan.

Frames are **hardlinked**, not copied.  The release tree and the source datasets
share one filesystem, so 293k links cost inodes and nothing else, while tar still
writes real bytes into the shipped archive.  A copy would cost ~120 GB twice.

SDM-Car ships 99 sequences as ``.avi`` and no frames at all.  MOTChallenge
readers want ``img1/000001.jpg``, so those are decoded once, in order, to JPEG.
That is the only place in the package where a pixel is re-encoded, and it is the
difference between a user running TrackEval and a user first writing a decoder.

Usage::

    python tools/stage_space_tracker_images.py \
        --out /data/<root>/release/space_tracker [--jobs 8] [--verify]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JPEG_QUALITY = 95


def link_one(dest: Path, src: Path) -> str:
    """Hardlink src -> dest; fall back to a copy across filesystems."""
    if dest.exists():
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "link"
    except FileExistsError:
        return "skip"
    except OSError:
        import shutil
        shutil.copy2(src, dest)
        return "copy"


def stage_frames(plan: Path, out: Path, jobs: int) -> dict:
    rows = list(csv.reader(open(plan), delimiter="\t"))
    counts = {"link": 0, "copy": 0, "skip": 0, "missing": 0}

    def work(row):
        dest_rel, src, _seq = row
        src_p = Path(src)
        if not src_p.exists():
            return "missing"
        return link_one(out / dest_rel, src_p)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, res in enumerate(pool.map(work, rows), start=1):
            counts[res] += 1
            if i % 25000 == 0:
                print(f"  {i}/{len(rows)} {counts}", flush=True)
    return counts


def extract_videos(plan: Path, out: Path) -> dict:
    import cv2

    jobs = json.loads(plan.read_text())
    counts = {"sequences": 0, "frames": 0, "short": [], "missing": []}
    for job in jobs:
        dest_dir = out / job["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        want = job["n_frames"]
        have = len(list(dest_dir.glob("*" + job["ext"])))
        if have >= want:
            counts["sequences"] += 1
            counts["frames"] += have
            continue
        src = Path(job["video_path"])
        if not src.exists():
            counts["missing"].append(job["seq"])
            continue
        cap = cv2.VideoCapture(str(src))
        written = 0
        while written < want:
            ok, frame = cap.read()
            if not ok:
                break
            written += 1
            cv2.imwrite(str(dest_dir / f"{written:06d}{job['ext']}"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        cap.release()
        if written < want:
            # The GT indexes frames the container does not hold; the annotation
            # is the authority on sequence length, so say so instead of
            # silently shipping a short sequence.
            counts["short"].append({"seq": job["seq"], "want": want, "got": written})
        counts["sequences"] += 1
        counts["frames"] += written
        if counts["sequences"] % 20 == 0:
            print(f"  video {counts['sequences']}/{len(jobs)}", flush=True)
    return counts


def reconcile(out: Path) -> list[str]:
    """Delete sequence directories the manifests no longer claim.

    A sequence's class folder is derived from the categories its ground truth
    actually contains, so re-deriving it can move a sequence — a car sequence in
    which the review hand-annotated a ship becomes a mixed one. Rebuilding then
    leaves the old directory behind, complete with its staged imagery. Move the
    images to where the manifest now says they live, then remove the stale copy;
    a second img1/ of the same frames is worse than wasted space, because a
    reader who finds it has no way to know which one the annotations describe.
    """
    notes = []
    for task, mani_name, img_name in (("sot", "space_tracker_sot.json", "img"),
                                      ("mot", "space_tracker_mot.json", "img1")):
        mani_path = out / task / mani_name
        if not mani_path.exists():
            continue
        mani = json.loads(mani_path.read_text())
        want = {r["path"]: r for r in mani["sequences"]}
        by_name = {r["name"]: r for r in mani["sequences"]}
        for group_dir in sorted((out / task).iterdir()):
            if not group_dir.is_dir() or group_dir.name == "annotations":
                continue
            for seq_dir in sorted(group_dir.iterdir()):
                rel = f"{task}/{group_dir.name}/{seq_dir.name}"
                if rel in want:
                    continue
                rec = by_name.get(seq_dir.name)
                src_img, moved = seq_dir / img_name, ""
                if rec is not None and src_img.is_dir():
                    dst_img = out / rec["path"] / img_name
                    if not dst_img.exists():
                        dst_img.parent.mkdir(parents=True, exist_ok=True)
                        src_img.rename(dst_img)
                        moved = f" (imagery moved to {rec['path']})"
                import shutil
                shutil.rmtree(seq_dir)
                notes.append(f"removed stale {rel}{moved}")
        for group_dir in sorted((out / task).iterdir()):
            if group_dir.is_dir() and group_dir.name != "annotations" \
                    and not any(group_dir.iterdir()):
                group_dir.rmdir()
                notes.append(f"removed empty folder {task}/{group_dir.name}")
    return notes


def verify(out: Path) -> list[str]:
    """Every sequence's manifest promises N frames; check N files are there."""
    problems = []
    for task, mani_name in (("sot", "space_tracker_sot.json"),
                            ("mot", "space_tracker_mot.json")):
        mani_path = out / task / mani_name
        if not mani_path.exists():
            continue
        mani = json.loads(mani_path.read_text())
        for rec in mani["sequences"]:
            img_dir = out / rec["path"] / ("img" if task == "sot" else "img1")
            n = len(list(img_dir.glob("*" + rec["img_ext"]))) if img_dir.is_dir() else 0
            if n != rec["n_frames"]:
                problems.append(f"{task}/{rec['name']}: {n} images, "
                                f"manifest says {rec['n_frames']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--task", choices=["sot", "mot", "both"], default="both")
    ap.add_argument("--verify", action="store_true",
                    help="only re-check that frame counts match the manifests")
    args = ap.parse_args()

    plan_dir = args.plan_dir or Path(str(args.out) + ".build")

    for note in reconcile(args.out):
        print(note)

    if not args.verify:
        if args.task in ("sot", "both"):
            print("SOT frames...")
            print(" ", stage_frames(plan_dir / "sot_images.tsv", args.out, args.jobs))
        if args.task in ("mot", "both"):
            print("MOT frames...")
            print(" ", stage_frames(plan_dir / "mot_images.tsv", args.out, args.jobs))
            print("MOT videos -> frames...")
            res = extract_videos(plan_dir / "mot_videos.json", args.out)
            print(" ", {k: v for k, v in res.items() if k in ("sequences", "frames")})
            for s in res["short"]:
                print("  SHORT:", s)
            for s in res["missing"]:
                print("  MISSING VIDEO:", s)

    problems = verify(args.out)
    print(f"\nverify: {len(problems)} sequences with a frame-count mismatch")
    for p in problems[:20]:
        print("   ", p)
    if len(problems) > 20:
        print(f"    ... and {len(problems) - 20} more")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
