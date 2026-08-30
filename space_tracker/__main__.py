"""Check a downloaded Space-Tracker package: ``python -m space_tracker <root>``.

Prints what the manifests claim and, with ``--verify``, confirms that the frames
and ground truth they point at are actually on disk.
"""

from __future__ import annotations

import argparse
import os
import sys

from .release import ROOT_ENV_VAR, SpaceTracker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m space_tracker", description=__doc__)
    ap.add_argument(
        "root",
        nargs="?",
        default=os.environ.get(ROOT_ENV_VAR),
        help=f"package root, the directory holding sot/ and mot/ (default: ${ROOT_ENV_VAR})",
    )
    ap.add_argument("--verify", action="store_true", help="also stat every ground-truth file and first/last frame")
    args = ap.parse_args(argv)

    if not args.root:
        ap.error(f"pass the package root or set {ROOT_ENV_VAR}")

    st = SpaceTracker(args.root)
    print(f"Space-Tracker at {st.root}")
    print(st.summary())

    for half in (st.sot, st.mot):
        by_cat: dict[str, int] = {}
        for seq in half:
            by_cat[seq.category] = by_cat.get(seq.category, 0) + 1
        cats = ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]))
        frames = sum(s.n_frames for s in half)
        print(f"  {half.task}: {frames:,} frames; class folders: {cats}")

    if not args.verify:
        return 0

    missing = 0
    for half in (st.sot, st.mot):
        for seq in half:
            for path in (seq.gt_path, seq.image_path(1), seq.image_path(seq.n_frames)):
                if not path.exists():
                    print(f"MISSING {path}", file=sys.stderr)
                    missing += 1
    print(f"verify: {missing} missing path(s)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
