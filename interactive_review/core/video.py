"""Whole-sequence playback — the last thing a reviewer sees before signing off.

A grid of crops answers "is this box tight on this frame". It cannot answer
whether the completed annotation holds up *as a video*: whether a recovered
track flickers, whether a box drifts off its object over 200 frames, whether two
tracks label the same aircraft. Those are properties of the sequence in motion,
so the sign-off step plays it.

Boxes are coloured by provenance, exactly as in the still views, so what the
reviewer signs is what they inspected.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2

from .gtsource import MERGED_ROOT, visible_objects
from .paths import MOT_ROOTS, sequence_by_id
from .render import _read_frame
from .vrender import COLOR_BY_PROVENANCE, label_of

#: Written per sequence, reused while nothing about the sequence has changed.
VIDEO_CACHE = Path("/tmp/space_tracker_review")

#: Decoding dominates rendering — a SAT-MTB frame is a 1.3 MB 1600x1200 PNG at
#: ~39 ms, against 0.1 ms to draw its boxes. ``cv2.imread`` releases the GIL, so
#: threads genuinely overlap here.
READ_WORKERS = 8


def _fingerprint(seq_id: str, decisions) -> str:
    """Identifies exactly what a rendered video shows.

    Cheaper and more honest than comparing timestamps: the cache is reused only
    when the ground truth file *and* every human edit are byte-identical to what
    was rendered, so a stale video can never be mistaken for a current one.
    """
    seq = sequence_by_id(seq_id)
    parts = [seq_id]
    for root in (MERGED_ROOT, MOT_ROOTS[seq.dataset]):
        p = root / seq.gt_path
        if p.is_file():
            st = p.stat()
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
    if decisions is not None:
        parts.append(json.dumps(decisions.get(seq_id).get("boxes") or {}, sort_keys=True))
        parts.append(json.dumps(decisions.get(seq_id).get("drawn") or {}, sort_keys=True))
        parts.append(json.dumps(sorted(decisions.deleted(seq_id))))
        # Relabelling changes both the colour-free label drawn on every box and
        # the identity the reviewer is signing off. Leaving it out would reuse a
        # video that still calls three ships `A104`, `A105`, `A109`.
        parts.append(json.dumps(decisions.labels(seq_id), sort_keys=True))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _draw(img, box, color, thickness=1, label=None, ring_below=0.0):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if ring_below and max(x2 - x1, y2 - y1) < ring_below:
        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), int(ring_below), color, 1)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 3, 9)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)


def render_sequence_video(
    seq_id: str,
    decisions=None,
    out_path: Path | None = None,
    fps: int = 10,
    max_side: int = 1000,
    ring_below: float = 14.0,
    labels: bool = True,
    progress: Callable[[float, str], None] | None = None,
    reuse: bool = True,
) -> Path:
    """Render the sequence with its completed ground truth drawn.

    Returns the path to an H.264 MP4 the browser can play inline, named for the
    fingerprint of everything it shows. A video already rendered from identical
    inputs is reused — re-watching a sequence is a normal part of the loop, and
    re-decoding 319 PNGs to produce the same file is pure latency — while any
    edit produces a different name, and so a different URL that no browser cache
    can answer from.
    """
    seq = sequence_by_id(seq_id)
    base = seq.frame_index_base
    frame_ids = list(range(base, base + seq.n_frames))

    # The fingerprint is in the *filename*, not in a sidecar next to a fixed one.
    # A fixed name meant every render of a sequence produced the same path, so
    # Gradio served it under the same URL and the browser replayed whatever it
    # had cached — the reviewer draws a track, the file on disk gains it, and
    # the <video> element keeps showing the version without it, permanently.
    # Server-side invalidation cannot fix that; only a URL that changes can.
    stem = seq_id.replace("/", "_")
    stamp = _fingerprint(seq_id, decisions)
    out_path = out_path or (VIDEO_CACHE / f"{stem}_{stamp}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse and out_path.is_file() and out_path.stat().st_size > 0:
        if progress is not None:
            progress(1.0, "already rendered")
        return out_path

    writer = None
    size = None
    scale = 1.0
    try:
        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            pending: deque = deque()
            upcoming = iter(frame_ids)
            for _ in range(READ_WORKERS * 2):
                fid = next(upcoming, None)
                if fid is None:
                    break
                pending.append((fid, pool.submit(_read_frame, seq_id, fid)))

            n = 0
            while pending:
                fid, future = pending.popleft()
                nxt = next(upcoming, None)
                if nxt is not None:
                    pending.append((nxt, pool.submit(_read_frame, seq_id, nxt)))
                if progress is not None and n % 10 == 0:
                    progress(n / max(len(frame_ids), 1),
                             f"rendering frame {n + 1}/{len(frame_ids)}")
                n += 1
                img = future.result()
                if img is None:
                    continue

                for o in visible_objects(seq_id, fid, decisions):
                    _draw(img, o.box,
                          COLOR_BY_PROVENANCE.get(o.provenance, (200, 200, 200)),
                          1, label_of(o) if labels else None, ring_below)

                cv2.putText(img, f"{seq_id}  frame {fid}", (8, img.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

                if writer is None:
                    h, w = img.shape[:2]
                    scale = min(1.0, max_side / max(h, w))
                    size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
                    writer = cv2.VideoWriter(str(out_path),
                                             cv2.VideoWriter_fourcc(*"avc1"),
                                             fps, size)
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer for {out_path}")
                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                writer.write(img[..., ::-1])
    finally:
        if writer is not None:
            writer.release()

    # Every edit to a sequence leaves one of these behind, and a 300-frame
    # SAT-MTB render is ~1.6 MB. Only this sequence's own stale renders go, and
    # only after the new one is written.
    for old_render in VIDEO_CACHE.glob(f"{stem}_*.mp4"):
        if old_render != out_path:
            old_render.unlink(missing_ok=True)
            old_render.with_suffix(".stamp").unlink(missing_ok=True)
    # Left by the fixed-name scheme this replaced.
    (VIDEO_CACHE / f"{stem}.mp4").unlink(missing_ok=True)
    (VIDEO_CACHE / f"{stem}.stamp").unlink(missing_ok=True)

    if progress is not None:
        progress(1.0, "done")
    return out_path
