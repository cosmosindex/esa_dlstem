#!/usr/bin/env python
"""Write the release's own README from ``space_tracker/README.md``.

The package on disk and the repository were documented twice, by hand, and they
drifted. There is one manual now: ``space_tracker/README.md``. This script
rewrites it for a reader who has the package but not the repository -- the
repo-relative links have no meaning there, and the sections about the loading
code's neighbours describe files that are not in the download.

Usage::

    python tools/make_release_readme.py --release $DATA_ROOT/release/space_tracker
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import DATA_ROOT, REPO_ROOT  # noqa: E402

SOURCE = Path(REPO_ROOT) / "space_tracker" / "README.md"

#: Sections that describe the repository rather than the package.
DROP_SECTIONS = ("## 13. Also in this directory",)

#: Links that resolve only inside the repository.
LINK_REWRITES = {
    "[`../DATASETS.md`](../DATASETS.md)": "`DATASETS.md` in the code repository",
    "[`space_tracker/README.md`](space_tracker/README.md)": "this file",
}


def convert(text: str) -> str:
    out = []
    skipping = False
    for block in re.split(r"(?m)^(?=## )", text):
        if any(block.startswith(s) for s in DROP_SECTIONS):
            skipping = True
            continue
        skipping = False
        out.append(block)
    text = "".join(out)

    for old, new in LINK_REWRITES.items():
        text = text.replace(old, new)

    text = text.replace(
        "# Space-Tracker — dataset manual",
        "# Space-Tracker")
    text = text.replace(
        "This directory is the loading code. It reads the released package and "
        "nothing\nelse — no source dataset is needed, and no field is recomputed "
        "at read time, so\nwhat you get is exactly what the package ships.",
        "This is the released package. The loading code that reads it, the "
        "annotation\ntool and the evaluation scripts live in the code "
        "repository; nothing here is\nrecomputed at read time, so what you read "
        "is exactly what was written.")
    text = text.replace(
        "Copy this directory next to your own code, or add the repository root to\n"
        "`PYTHONPATH`.",
        "Copy the `space_tracker/` package from the code repository next to your "
        "own\ncode, or add that repository's root to `PYTHONPATH`.")

    # Renumber the section headings left after the drop.
    n = 0
    def _renumber(m):
        nonlocal n
        n += 1
        return f"## {n}. {m.group(2)}"
    text = re.sub(r"(?m)^## (\d+)\. (.+)$", _renumber, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path,
                    default=Path(DATA_ROOT) / "release" / "space_tracker")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = convert(SOURCE.read_text())
    target = args.release / "README.md"
    if args.dry_run:
        print(text)
        return
    target.write_text(text)
    print(f"wrote {target} ({len(text.splitlines())} lines) from "
          f"{SOURCE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
