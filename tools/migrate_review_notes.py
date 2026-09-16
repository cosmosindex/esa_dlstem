"""Turn free-text review notes into tags, without deciding anything.

The first review pass had only one place to put "this video is all-static" or
"single target, usable for SOT": the note box, next to a Flag button. So three
different intentions — park a decision, drop the sequence, describe the video —
all came out as ``flagged`` with a sentence attached.

This recovers the describable part. It reads each note, adds the tags it clearly
implies, and **leaves ``status`` and the note itself exactly as they are**: which
of those sequences were parked and which were merely described is a judgement
only the reviewer can make, and guessing it would quietly decide what ships.

Patterns are deliberately generous on spelling — the notes contain `Occlusion`,
`Occusion` and `Occulssion` for the same thing.

Usage::

    python tools/migrate_review_notes.py --dry-run
    python tools/migrate_review_notes.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_tool.core.paths import DEFAULT_OVERRIDES
from annotation_tool.core.vdecisions import SequenceDecisions

PATTERNS = {
    "static":        r"static|not\s+moving|is\s+not\s+mov",
    "single_object": r"\bSOT\b|single\s+(airplane|object|target|ship)",
    "fast_motion":   r"fast[-\s]?moving|rushed\s+out",
    # One thing, three spellings in the same pass.
    "occlusion":     r"occ\w*s+ion|occlu\w*",
    "sudden_entry":  r"suddenly\s+rushed\s+out|suddenly\s+appear",
    "maneuver":      r"making\s+a\s+turn|turning|\bturn\b",
    "tiny":          r"\btiny\b|\bsmall\b",
    "weather":       r"smogg?y|haze|hazy|cloud|fog",
}


def tags_for(note: str) -> list[str]:
    return sorted(t for t, p in PATTERNS.items() if re.search(p, note, re.I))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    decisions = SequenceDecisions(args.overrides)
    doc = decisions.as_dict()["sequences"]

    changed = 0
    counts: Counter = Counter()
    unmatched = []
    for seq_id, entry in sorted(doc.items()):
        note = (entry.get("note") or "").strip()
        if not note:
            continue
        derived = tags_for(note)
        if not derived:
            unmatched.append((seq_id, note))
            continue
        merged = sorted(set(entry.get("tags") or []) | set(derived))
        if merged != (entry.get("tags") or []):
            changed += 1
            counts.update(derived)
            print(f"  {seq_id:<16} {note!r}")
            print(f"  {'':<16} -> {' '.join(merged)}   (status stays "
                  f"{entry.get('status')!r})")
            if not args.dry_run:
                decisions.set_tags(seq_id, merged)

    if not args.dry_run and changed:
        decisions.save()

    print(f"\n{changed} sequences tagged"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    for t, n in counts.most_common():
        print(f"  {t:<15} {n}")
    if unmatched:
        print(f"\n{len(unmatched)} notes matched no pattern — tag these by hand:")
        for seq_id, note in unmatched:
            print(f"  {seq_id:<16} {note!r}")


if __name__ == "__main__":
    main()
