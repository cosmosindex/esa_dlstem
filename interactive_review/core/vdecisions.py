"""Per-sequence review state: what a human fixed, and what they signed off.

The track-level store (:mod:`.overrides`) keys everything by track because a
track was the unit of decision. Here the unit is a *video*: a reviewer walks one
sequence frame by frame, corrects whatever is wrong anywhere in it, watches it
back, and accepts the sequence as a whole. Keying sign-off by track would let a
sequence be half-accepted, which is exactly the state this workflow exists to
avoid.

Schema (``version: 1``)::

    {
      "version": 1,
      "sequences": {
        "satmtb/airplane/28": {
          "status":   "accepted" | "flagged" | null,
          "note":     "",
          "reviewer": "",
          "updated":  "2026-07-28T09:00:00",
          "watched":  false,
          "boxes":    {"airplane:12": {"41": [x1, y1, x2, y2]}},
          "drawn":    {"ship:9001":   {"1":  [x1, y1, x2, y2]}},
          "deleted":  ["airplane:37"]
        }
      }
    }

``boxes`` corrects existing tracks one frame at a time — sparse on purpose, so
fixing 1 frame of 283 leaves the other 282 on the batch geometry. ``drawn``
holds tracks annotated from nothing, which have no lower layer to fall back to.
``deleted`` removes a track the reviewer judges spurious.

``watched`` and ``status`` are deliberately fragile: any edit clears both, so an
acceptance always refers to exactly the annotation that was played back.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1

#: ``accepted``  ships.
#: ``flagged``   parked — a decision is owed before it can ship.
#: ``excluded``  deliberately dropped from the release; needs a recorded reason,
#:               because a sequence that vanishes without one is indistinguishable
#:               from a sequence that was lost.
STATUSES = ("accepted", "flagged", "excluded")

#: Track ids for hand-drawn tracks start here, far above any dataset's own ids.
DRAWN_ID_BASE = 9001


class SequenceDecisions:
    """Load / mutate / save the per-sequence review document."""

    def __init__(self, path: Path, reviewer: str = ""):
        self.path = Path(path)
        self.reviewer = reviewer
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.is_file():
            data = json.loads(self.path.read_text())
            if data.get("version") != SCHEMA_VERSION:
                raise ValueError(f"{self.path} has schema version "
                                 f"{data.get('version')}, expected {SCHEMA_VERSION}")
            return data
        return {"version": SCHEMA_VERSION, "sequences": {}}

    def save(self) -> None:
        """Atomic write — a crash mid-review must not truncate hours of work."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- access -------------------------------------------------------------

    def _entry(self, seq_id: str) -> dict:
        return self._data["sequences"].setdefault(seq_id, {
            "status": None, "note": "", "reviewer": self.reviewer,
            "updated": None, "watched": False, "tags": [],
            "boxes": {}, "drawn": {}, "deleted": [],
        })

    def get(self, seq_id: str) -> dict:
        return self._data["sequences"].get(seq_id, {})

    def boxes(self, seq_id: str) -> dict[str, dict[int, list[float]]]:
        raw = self.get(seq_id).get("boxes") or {}
        return {k: {int(f): [float(v) for v in b] for f, b in fb.items()}
                for k, fb in raw.items()}

    def drawn(self, seq_id: str) -> dict[str, dict[int, list[float]]]:
        raw = self.get(seq_id).get("drawn") or {}
        return {k: {int(f): [float(v) for v in b] for f, b in fb.items()}
                for k, fb in raw.items()}

    def deleted(self, seq_id: str) -> set[str]:
        return set(self.get(seq_id).get("deleted") or [])

    # -- mutation -----------------------------------------------------------

    def _touch(self, entry: dict) -> None:
        """Any edit invalidates the sign-off — never silently keep it."""
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        entry["reviewer"] = self.reviewer
        entry["status"] = None
        entry["watched"] = False

    def patch_box(self, seq_id: str, key: str, frame_id: int,
                  box: list[float] | None) -> None:
        """Correct one frame of one track. ``box=None`` reverts that frame."""
        e = self._entry(seq_id)
        frames = dict(e["boxes"].get(key) or {})
        if box is None:
            frames.pop(str(frame_id), None)
        else:
            frames[str(frame_id)] = [round(float(v), 2) for v in box]
        if frames:
            e["boxes"][key] = frames
        else:
            e["boxes"].pop(key, None)
        self._touch(e)

    def new_drawn_key(self, seq_id: str, category: str) -> str:
        """Next free key for a hand-drawn track in this sequence."""
        used = {int(k.split(":", 1)[1]) for k in self.get(seq_id).get("drawn", {})}
        n = DRAWN_ID_BASE
        while n in used:
            n += 1
        return f"{category}:{n}"

    def set_drawn(self, seq_id: str, key: str,
                  boxes: dict[int, list[float]] | None) -> None:
        e = self._entry(seq_id)
        if boxes:
            e["drawn"][key] = {str(f): [round(float(v), 2) for v in b]
                               for f, b in sorted(boxes.items())}
        else:
            e["drawn"].pop(key, None)
        self._touch(e)

    def set_deleted(self, seq_id: str, key: str, deleted: bool = True) -> None:
        e = self._entry(seq_id)
        keys = set(e.get("deleted") or [])
        keys.add(key) if deleted else keys.discard(key)
        e["deleted"] = sorted(keys)
        self._touch(e)

    def tags(self, seq_id: str) -> list[str]:
        return list(self.get(seq_id).get("tags") or [])

    def set_tags(self, seq_id: str, tags: list[str] | None) -> None:
        """Attach descriptive labels to a sequence.

        Deliberately does **not** touch ``status`` or ``watched``: "every object
        in this video is static" is a property of the imagery, not a claim about
        whether anyone has reviewed it, and collapsing the two is what made a
        single Flag button carry three different meanings.
        """
        e = self._entry(seq_id)
        e["tags"] = sorted({t.strip() for t in (tags or []) if t.strip()})
        e["updated"] = datetime.now().isoformat(timespec="seconds")
        e["reviewer"] = self.reviewer

    def set_watched(self, seq_id: str) -> None:
        """Record that the whole sequence was played back with the current state."""
        e = self._entry(seq_id)
        e["watched"] = True
        e["updated"] = datetime.now().isoformat(timespec="seconds")

    def set_status(self, seq_id: str, status: str | None, note: str = "") -> None:
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES} or None, got {status!r}")
        e = self._entry(seq_id)
        e["status"] = status
        if note:
            e["note"] = note
        e["reviewer"] = self.reviewer
        e["updated"] = datetime.now().isoformat(timespec="seconds")

    # -- reporting ----------------------------------------------------------

    def status_of(self, seq_id: str) -> str | None:
        return self.get(seq_id).get("status")

    def is_touched(self, seq_id: str) -> bool:
        e = self.get(seq_id)
        return bool(e.get("boxes") or e.get("drawn") or e.get("deleted"))

    def edit_counts(self, seq_id: str) -> dict[str, int]:
        e = self.get(seq_id)
        return {
            "frames_fixed": sum(len(v) for v in (e.get("boxes") or {}).values()),
            "tracks_fixed": len(e.get("boxes") or {}),
            "tracks_drawn": len(e.get("drawn") or {}),
            "boxes_drawn": sum(len(v) for v in (e.get("drawn") or {}).values()),
            "tracks_deleted": len(e.get("deleted") or []),
        }

    def stats(self) -> Counter:
        c: Counter = Counter()
        for e in self._data["sequences"].values():
            if e.get("status"):
                c[e["status"]] += 1
            if e.get("boxes") or e.get("drawn") or e.get("deleted"):
                c["edited"] += 1
            if e.get("tags"):
                c["tagged"] += 1
        return c

    def tag_counts(self) -> Counter:
        return Counter(t for e in self._data["sequences"].values()
                       for t in (e.get("tags") or []))

    def as_dict(self) -> dict:
        return self._data
