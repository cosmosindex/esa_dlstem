"""Does a drawn track move over the ground, or only with the camera?

SAT-MTB annotates ``car`` for movers only, so a sweep that finds every vehicle
in an airport or an interchange finds the wrong half of them: parked cars are
matched just as confidently as moving ones, and shipping them changes what the
category means. This decides which is which.

The camera's own motion is measured **per object, from the background right
around it**, not once for the whole frame and not from the annotations.

Both of the obvious alternatives were tried and rejected on real sequences:

*from the annotations*
    Taking the median displacement of all tracks as the camera's assumes most
    tracks are static, which is the very question being asked. On
    ``satmtb/airplane/14`` (25 of 33 static) it was right; on
    ``satmtb/airplane/34`` (12 of 14 moving) the median was 49x55 px of pure
    traffic and every verdict it gave was wrong.

*one global phase correlation*
    A single translation does not model satellite video, which rotates. The
    residual it left on genuinely static objects was 12-15 px — the same order
    as a slow vehicle, so the two were not separable.

A window around the object moves with whatever the camera is doing *there*,
rotation and parallax included, and leaves a residual near zero: on that same
sequence the static tracks came out at 0.5-3 px and the moving ones at 29-115.

A track can fail to be a moving car in two independent ways, and measuring
against the ground only catches one of them:

*parked*
    Fixed on the ground, so it travels across the image with the camera. Ground
    displacement ~0; image displacement equals the camera's.

*pinned*
    Fixed in the image — the propagation lost the object, usually where it left
    the frame, and the box stayed at the border. Image displacement ~0; ground
    displacement equals the camera's, so the ground test calls it *moving* and
    the further the camera travels the more confidently it is wrong.

They are opposites in both coordinate frames, so neither test finds the other
and both are needed. A parked car is a real object annotated under the wrong
convention; a pinned box is not an object at all.

Known failure: phase correlation locks onto the wrong peak over strongly
periodic texture — rows of parked cars, hangar roofs — and reports motion that
is not there. So a verdict here marks a track deleted, which is reversible and
visible, rather than removing anything.
"""

from __future__ import annotations

import cv2
import numpy as np

from .render import _read_frame

#: Ground-relative displacement, in pixels, below which a track is not moving.
#: Between the two populations measured on airplane/14: static topped out at
#: 3 px and moving started at 29.
STATIC_PX = 15.0
#: Window half-size around the object, as a multiple of its longest side.
WIN_MULT = 6.0
MIN_WIN = 48.0
#: Image-frame displacement, in pixels, below which a track is pinned rather
#: than tracking anything. Well under a car's own travel — the slowest real car
#: measured on rscardata/train/011 covers 1.4 px per frame — and well under the
#: camera motion that a parked car would be carried by.
PINNED_PX = 8.0
#: Too few frames and a short track that happens to start and end in the same
#: place reads as pinned. A propagation that has locked up runs long.
PINNED_FRAMES = 15
#: Frames between samples. The question is displacement over the whole track,
#: not its velocity profile, so a coarse sample is enough and much cheaper.
STEP = 20
#: Frames between samples when looking for the moment motion *starts*. That is a
#: question about the velocity profile, so it needs a fine sample — but not one
#: per frame: a car pulling away covers its own length in a handful of them.
ONSET_STEP = 2
#: Floor under the onset threshold, in pixels. Phase correlation on a genuinely
#: parked object leaves 0.5-3 px of residual, so nothing below this can be
#: distinguished from a stationary car however quiet the measurement looks.
ONSET_MIN_PX = 3.0
#: The onset threshold is this multiple of the track's *own* measured noise,
#: floored by :data:`ONSET_MIN_PX`. Taken from the track rather than fixed
#: because the residual scales with how much texture is around the object.
ONSET_NOISE_MULT = 2.0

_gray_cache: dict[tuple[str, int], np.ndarray | None] = {}


def _gray(seq_id: str, fid: int):
    key = (seq_id, fid)
    if key in _gray_cache:
        # Move to the end so eviction is least-recently-used, not first-in. Every
        # comparison re-reads the reference frame, and a FIFO would drop it and
        # re-decode it once the sampling is dense enough to fill the cache.
        _gray_cache[key] = _gray_cache.pop(key)
        return _gray_cache[key]
    img = _read_frame(seq_id, fid)
    _gray_cache[key] = (None if img is None else
                        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32))
    if len(_gray_cache) > 200:
        _gray_cache.pop(next(iter(_gray_cache)))
    return _gray_cache[key]


def _local_shift(seq_id: str, f0: int, f1: int, cx: float, cy: float, half: float):
    """How far the ground moved between two frames, in a window at (cx, cy)."""
    a, b = _gray(seq_id, f0), _gray(seq_id, f1)
    if a is None or b is None:
        return None
    H, W = a.shape
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    pa, pb = a[y1:y2, x1:x2], b[y1:y2, x1:x2]
    win = cv2.createHanningWindow((pa.shape[1], pa.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(pa, pb, win)
    return np.array([dx, dy])


def ground_motion(seq_id: str, track: dict[int, list[float]],
                  step: int = STEP) -> tuple[float, float, int]:
    """``(net, max, samples)`` displacement of a track relative to the ground.

    ``net`` is first sample to last, ``max`` the furthest it ever got from where
    it started. ``net`` is the more stable of the two — a single mis-correlated
    window inflates ``max`` and leaves ``net`` alone — so ``net`` decides and
    ``max`` is reported alongside it.
    """
    fids = sorted(track)
    if len(fids) < 2:
        return 0.0, 0.0, len(fids)
    sample = fids[::step]
    if sample[-1] != fids[-1]:
        sample.append(fids[-1])

    f0 = sample[0]
    b0 = track[f0]
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    half = max(WIN_MULT * max(b0[2] - b0[0], b0[3] - b0[1]), MIN_WIN)

    rel = [np.zeros(2)]
    for f in sample[1:]:
        b = track[f]
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        shift = _local_shift(seq_id, f0, f, c0[0], c0[1], half)
        if shift is None:
            continue
        rel.append((c - c0) - shift)
    r = np.array(rel)
    return (float(np.linalg.norm(r[-1])),
            float(np.linalg.norm(r, axis=1).max()), len(r))


def image_motion(track: dict[int, list[float]]) -> float:
    """First-to-last displacement of a track in image coordinates.

    No correlation and no imagery: a pinned box is pinned *to the image*, so the
    only thing to measure is where its own corners were. Cheap enough to run on
    every track before deciding which are worth a ground measurement.
    """
    fids = sorted(track)
    if len(fids) < 2:
        return 0.0
    c = lambda f: np.array([(track[f][0] + track[f][2]) / 2,
                            (track[f][1] + track[f][3]) / 2])
    return float(np.linalg.norm(c(fids[-1]) - c(fids[0])))


def static_tracks(seq_id: str, decisions, threshold: float = STATIC_PX,
                  pinned_px: float = PINNED_PX,
                  pinned_frames: int = PINNED_FRAMES,
                  progress=None) -> tuple[list[str], list[tuple]]:
    """Which drawn tracks are not a moving object, by either measure.

    Returns ``(keys, [(key, ground_net, ground_max, image_net, n_frames,
    reason) for every track])`` — the rows so the reviewer can see the margin
    each verdict was decided by, rather than a bare list to take on trust.
    ``reason`` is ``"parked"``, ``"pinned"`` or ``""``.

    The image test runs first because it is free, and because a pinned track
    would otherwise be *cleared* by the ground test: standing still in the image
    while the camera pans reads there as motion, and reads as more of it the
    longer the sequence.
    """
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)
    live = [k for k in sorted(drawn, key=lambda k: (k.split(":")[0],
                                                    int(k.split(":")[1])))
            if k not in deleted]
    out, rows = [], []
    for n, key in enumerate(live):
        if progress is not None:
            progress(n / max(len(live), 1), f"measuring {key}")
        track = drawn[key]
        img = image_motion(track)
        if img < pinned_px and len(track) >= pinned_frames:
            # Cost nothing to skip: the ground number for a pinned box is the
            # camera's own travel and says nothing about the object.
            rows.append((key, float("nan"), float("nan"), img, len(track), "pinned"))
            out.append(key)
            continue
        net, mx, _n = ground_motion(seq_id, track)
        reason = "parked" if net < threshold else ""
        rows.append((key, net, mx, img, len(track), reason))
        if reason:
            out.append(key)
    if progress is not None:
        progress(1.0, f"{len(out)} of {len(live)} are not moving objects")
    return out, rows


def ground_track(seq_id: str, track: dict[int, list[float]],
                 step: int = ONSET_STEP) -> list[tuple[int, float]]:
    """``(frame, ground displacement from the track's first frame)``, sampled.

    The whole profile rather than its endpoints, because the question here is
    *when* the object started moving, not whether it ever did.
    """
    fids = sorted(track)
    if len(fids) < 2:
        return [(f, 0.0) for f in fids]
    f0 = fids[0]
    b0 = track[f0]
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    half = max(WIN_MULT * max(b0[2] - b0[0], b0[3] - b0[1]), MIN_WIN)

    out = []
    for f in fids[::step] + ([fids[-1]] if fids[-1] % step else []):
        b = track[f]
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        shift = _local_shift(seq_id, f0, f, c0[0], c0[1], half)
        if shift is None:
            continue
        out.append((f, float(np.linalg.norm((c - c0) - shift))))
    return out


def motion_onset(seq_id: str, track: dict[int, list[float]]) -> dict:
    """The frame a parked object pulls away, and the evidence for it.

    ``{"onset", "floor", "noise", "profile"}``; ``onset`` is ``None`` when the
    track never clearly moves, and equals its first frame when it was already
    moving. The car sets annotate movers only, so the frames before ``onset``
    are a parked car under a convention that has no category for one — not a
    tracking error, but not ground truth either.

    The threshold comes from the track's own noise, not from a constant. What
    phase correlation leaves on a stationary object depends on how much texture
    surrounds it, so the same fixed number is too tight beside a hedgerow and
    too loose on open tarmac. The noise is measured over the first half of the
    frames that precede any unambiguous motion, which is a stretch the object
    cannot yet have travelled across.

    The onset is the *last* frame under the threshold, not the first frame over
    it: a car pulling away crosses once and never returns, whereas a single
    mis-correlated window crosses and comes straight back, and taking the last
    crossing ignores the second.
    """
    profile = ground_track(seq_id, track)
    if len(profile) < 4:
        return {"onset": None, "floor": 0.0, "noise": 0.0, "profile": profile}

    moving = [i for i, (_f, d) in enumerate(profile) if d > STATIC_PX]
    if not moving:
        return {"onset": None, "floor": 0.0, "noise": 0.0, "profile": profile}

    quiet = [d for _f, d in profile[:max(moving[0] // 2, 2)]]
    noise = float(np.quantile(quiet, 0.95)) if quiet else 0.0
    floor = max(ONSET_NOISE_MULT * noise, ONSET_MIN_PX)

    below = [i for i, (_f, d) in enumerate(profile[:moving[0]]) if d <= floor]
    if not below:
        return {"onset": profile[0][0], "floor": floor, "noise": noise,
                "profile": profile}
    last_quiet = below[-1]
    if last_quiet + 1 >= len(profile):
        return {"onset": None, "floor": floor, "noise": noise, "profile": profile}
    return {"onset": profile[last_quiet + 1][0], "floor": floor,
            "noise": noise, "profile": profile}


def trim_parked_starts(seq_id: str, decisions, progress=None) -> list[str]:
    """Drop the drawn frames from before each object started moving.

    Only drawn frames go: where the dataset itself annotates a frame it has
    already ruled on whether that object counts, and this tool does not overrule
    it. A track that turns out never to move is left alone rather than emptied,
    because "delete the whole thing" is what `Delete static tracks` is for and
    it is reversible, whereas this is not.
    """
    drawn = decisions.drawn(seq_id)
    deleted = decisions.deleted(seq_id)
    live = [k for k in sorted(drawn, key=lambda k: (k.split(":")[0],
                                                    int(k.split(":")[1])))
            if k not in deleted and drawn[k]]
    notes = []
    for n, key in enumerate(live):
        if progress is not None:
            progress(n / max(len(live), 1), f"measuring {key}")
        track = {int(f): [float(v) for v in b] for f, b in drawn[key].items()}
        r = motion_onset(seq_id, track)
        onset = r["onset"]
        if onset is None or onset <= min(track):
            continue
        kept = {f: b for f, b in drawn[key].items() if int(f) >= onset}
        dropped = len(track) - len(kept)
        if not dropped:
            continue
        decisions.set_drawn(seq_id, key, kept or None)
        notes.append(f"{key}: {dropped} frames before {onset} dropped — parked "
                     f"there, moving under {r['floor']:.1f} px over the ground "
                     f"against a {r['noise']:.1f} px measurement noise")
    if notes:
        decisions.save()
    if progress is not None:
        progress(1.0, f"{len(notes)} of {len(live)} had a parked start")
    return notes
