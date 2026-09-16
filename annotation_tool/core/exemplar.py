"""Find every object that looks like one the reviewer already annotated.

Drawing one car by hand takes a click, a Segment and a Propagate. A SAT-MTB
airplane sequence holds dozens of moving cars that no annotation covers, and
doing that dozens of times is the reason those cars are still unlabelled.

SAM 3's detector can be prompted with a *visual exemplar* instead of text: its
geometry encoder RoI-aligns a box over an image's features and pools that into
a prompt token, and the grounding head then finds that concept on any other
image. So one hand-drawn car becomes the query, and the whole sequence is swept
for the rest of them. This is the same mechanism the BIRDSAI cross-image
exemplar probe validated, where a single training exemplar recovered tiny
thermal animals that text prompts and three trained detectors all scored ~0 on.

Two things keep it honest:

*negatives*
    Background boxes from the same sequence are pooled into the bank with a
    negative label, which pushes the prototype away from "any small bright
    blob" — the failure mode on overhead imagery, where roof furniture and
    road markings read like vehicles.
*size gating*
    A candidate whose area is nowhere near the exemplar's is dropped. In an
    ``airplane`` sequence the strongest visual match to a car-shaped prompt is
    often an aircraft, and it is worth rejecting on geometry alone.

Nothing here writes to the review. Candidates come back as boxes for the
reviewer to look at, and only become tracks when they ask for it.
"""

from __future__ import annotations

import numpy as np

from .gtsource import frame_objects, visible_objects
from .paths import sequence_by_id
from .render import _read_frame
from .sam3refine import sam3_autocast

#: Positive exemplar tokens pooled into the bank, sampled along the source track.
K_POS = 6
#: Background tokens. Fewer than the positives, so the pool stays positive-weighted.
M_NEG = 2
#: Class-agnostic NMS on each frame's detections.
NMS_IOU = 0.5
#: A detection this close to something already annotated is that thing.
DUP_IOU = 0.2
#: Candidate area must stay inside this band around the exemplar's median area.
SIZE_LO, SIZE_HI = 0.15, 6.0

_image_model = None
_processor = None


def get_detector(confidence: float = 0.2):
    """Build the SAM 3 *image* model once — separate from the video tracker.

    The tracker shares the detector backbone but exposes no grounding head, so
    exemplar detection needs the image model as well. Both live on the GPU at
    once; together they are well inside a 32 GB card.
    """
    global _image_model, _processor
    if _image_model is None:
        from models.sam3 import SAM3Tracker
        SAM3Tracker._ensure_pkg_resources_shim()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        _image_model = build_sam3_image_model(load_from_HF=True)
        _processor = Sam3Processor(_image_model, confidence_threshold=confidence)
    _processor.confidence_threshold = confidence
    return _image_model, _processor


def unload_detector() -> None:
    global _image_model, _processor
    _image_model = _processor = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass


def _iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _norm_cxcywh(box, W, H):
    x1, y1, x2, y2 = box
    return np.clip([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                    (x2 - x1) / W, (y2 - y1) / H], 0, 1)


def _sample_positives(track: dict[int, list[float]], k: int):
    """``k`` frames spread evenly along the track.

    Evenly rather than the k largest: a bank built from one pose of one car at
    one scale finds that car again and little else.
    """
    fids = sorted(track)
    if not fids:
        return []
    idx = np.unique(np.linspace(0, len(fids) - 1, min(k, len(fids))).round().astype(int))
    return [(fids[i], track[fids[i]]) for i in idx]


def _sample_negatives(seq_id, fids, decisions, exemplar_area, m, rng):
    """Background boxes of roughly the exemplar's size, clear of every object."""
    out = []
    side = max(int(round(float(np.sqrt(exemplar_area)))), 4)
    for fid in fids:
        # Deleted tracks still count as occupied here, unlike everywhere else: a
        # track is often deleted for bad geometry over a real object, and a
        # negative sampled on a real object teaches the bank the wrong thing.
        # Losing a few background samples costs nothing by comparison.
        occupied = np.asarray([o.box for o in frame_objects(seq_id, fid, decisions)],
                              np.float32).reshape(-1, 4)
        img = _read_frame(seq_id, fid)
        if img is None:
            continue
        H, W = img.shape[:2]
        for _ in range(200):
            if len(out) >= m:
                break
            x = int(rng.integers(0, max(1, W - side)))
            y = int(rng.integers(0, max(1, H - side)))
            box = np.array([x, y, x + side, y + side], np.float32)
            if len(occupied) and (_iou_mat(box[None], occupied) > 0.02).any():
                continue
            out.append((fid, [float(v) for v in box]))
        if len(out) >= m:
            break
    return out


def build_bank(seq_id: str, positives, negatives):
    """Pool exemplar boxes into one geometry-prompt bank.

    ``positives``/``negatives`` are ``[(frame_id, xyxy)]``. Only the box token
    of each exemplar is kept — the per-frame CLS token is generic, and keeping
    it lets K frames of the same video outvote the object itself.
    """
    import torch
    from PIL import Image

    model, proc = get_detector()
    feats, masks = [], []
    # Same thread-local autocast problem as the video tracker: SAM 3 opens a
    # bf16 context in its constructor, Gradio runs each callback on a different
    # worker thread, and the weights meet fp32 activations.
    with torch.inference_mode(), sam3_autocast():
        for fid, box, positive in ([(f, b, True) for f, b in positives]
                                   + [(f, b, False) for f, b in negatives]):
            img = _read_frame(seq_id, fid)
            if img is None:
                continue
            pil = Image.fromarray(img)
            W, H = pil.size
            st = proc.set_image(pil)
            proc.reset_all_prompts(st)
            geo = model._get_dummy_prompt()
            cxcywh = _norm_cxcywh(box, W, H)
            geo.append_boxes(
                torch.tensor(cxcywh, dtype=torch.float32,
                             device=proc.device).view(1, 1, 4),
                torch.tensor([positive], device=proc.device).view(1, 1))
            _, img_feats, img_pos, sizes = model._get_img_feats(
                st["backbone_out"], proc.find_stage.img_ids)
            f, m = model.geometry_encoder(geo, img_feats, sizes, img_pos)
            feats.append(f[:1])
            masks.append(m[:, :1])
    if not feats:
        return None
    return torch.cat(feats, 0), torch.cat(masks, 1)


def detect_frame(seq_id: str, frame_id: int, bank, thr: float):
    """Run the exemplar-prompted detector on one frame. Returns (boxes, scores)."""
    import torch
    import torchvision
    from PIL import Image

    model, proc = get_detector(confidence=min(thr, 0.2))
    img = _read_frame(seq_id, frame_id)
    if img is None:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    geo_feats, geo_masks = bank
    with torch.inference_mode(), sam3_autocast():
        st = proc.set_image(Image.fromarray(img))
        proc.reset_all_prompts(st)
        # The text prompt is only there to build the state the grounding head
        # consumes; the geometry encoder is then short-circuited to the bank, so
        # the words never reach the decision.
        st = proc.set_text_prompt("object", st)
        orig = model.geometry_encoder.forward
        model.geometry_encoder.forward = lambda *a, **k: (geo_feats, geo_masks)
        try:
            st["geometric_prompt"] = model._get_dummy_prompt()
            st = proc._forward_grounding(st)
        finally:
            model.geometry_encoder.forward = orig
        boxes, scores = st["boxes"].float(), st["scores"].float()
        if len(boxes):
            keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
            boxes, scores = boxes[keep], scores[keep]
    b = boxes.cpu().numpy().reshape(-1, 4)
    s = scores.cpu().numpy().reshape(-1)
    sel = s >= thr
    return b[sel], s[sel]


def find_similar(seq_id: str, track: dict[int, list[float]], *,
                 decisions=None, thr: float = 0.35, stride: int = 12,
                 max_frames: int = 20, max_new: int = 200, exclude=None,
                 stats: dict | None = None,
                 progress=None) -> list[tuple[int, list[float], float]]:
    """Sweep the sequence for objects that look like ``track``.

    Frames are sampled rather than swept exhaustively: an object that is never
    visible on any sampled frame is one the propagation would not hold anyway,
    and every sampled frame costs a detector forward pass. Sampling also picks
    up objects that enter the video late, which prompting a single frame does
    not.

    ``exclude`` is a ``{frame_id: xyxy}`` track to suppress as well — the
    exemplar itself when it comes from an unsaved draft, which the review
    document does not know about yet and would otherwise be proposed back to
    the reviewer as a discovery.

    ``stats``, if given, is filled with where the detections went. An empty
    result has three quite different causes — the detector found nothing, it
    found things of the wrong size, or it found objects that are already
    annotated — and they call for opposite responses from the reviewer. A bare
    zero tells them which none.

    Returns ``[(frame_id, xyxy, score)]``, strongest first, with anything
    already annotated on its own frame removed.
    """
    if not track:
        return []
    seq = sequence_by_id(seq_id)
    base = seq.frame_index_base
    all_fids = list(range(base, base + seq.n_frames))
    step = max(stride, int(np.ceil(len(all_fids) / max(max_frames, 1))))
    sample = all_fids[::step]

    rng = np.random.default_rng(0)
    pos = _sample_positives(track, K_POS)
    med_area = float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for _, b in pos]))
    neg = _sample_negatives(seq_id, sample[:4], decisions, med_area, M_NEG, rng)

    if progress is not None:
        progress(0.02, f"building an exemplar bank from {len(pos)} boxes")
    bank = build_bank(seq_id, pos, neg)
    if bank is None:
        return []

    counts = {"frames": len(sample), "raw": 0, "wrong_size": 0,
              "already_annotated": 0, "new": 0}
    out: list[tuple[int, list[float], float]] = []
    for n, fid in enumerate(sample):
        if progress is not None:
            progress(0.05 + 0.9 * n / max(len(sample), 1),
                     f"searching frame {fid} ({len(out)} found)")
        boxes, scores = detect_frame(seq_id, fid, bank, thr)
        counts["raw"] += len(boxes)
        if not len(boxes):
            continue
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (area > SIZE_LO * med_area) & (area < SIZE_HI * med_area)
        counts["wrong_size"] += int((~keep).sum())
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            continue
        # Deleted deliberately excluded: a reviewer who deletes a bad track
        # and sweeps again is asking for that object back, and counting it
        # as already annotated is how the sweep would silently refuse.
        known = [o.box for o in visible_objects(seq_id, fid, decisions)]
        if exclude is not None and fid in exclude:
            known.append(exclude[fid])
        known = np.asarray(known, np.float32).reshape(-1, 4)
        if len(known):
            boxes_keep = (_iou_mat(boxes, known) < DUP_IOU).all(axis=1)
            counts["already_annotated"] += int((~boxes_keep).sum())
            boxes, scores = boxes[boxes_keep], scores[boxes_keep]
        for b, sc in zip(boxes, scores):
            out.append((fid, [float(v) for v in b], float(sc)))

    counts["new"] = len(out)
    if stats is not None:
        stats.update(counts)
    out.sort(key=lambda t: -t[2])
    if progress is not None:
        progress(1.0, f"{len(out)} candidates")
    return out[:max_new]


#: Two tracks agreeing this well over the frames they share are one object.
SAME_TRACK_IOU = 0.4
#: ...but only once they share this many frames — two boxes crossing for a frame
#: or two are not the same car.
SAME_TRACK_MIN_SHARED = 3


def _same_object(a: dict, b: dict) -> bool:
    shared = sorted(set(a) & set(b))
    if len(shared) < SAME_TRACK_MIN_SHARED:
        return False
    ious = _iou_mat(np.asarray([a[f] for f in shared], np.float32),
                    np.asarray([b[f] for f in shared], np.float32)).diagonal()
    return float(np.mean(ious)) > SAME_TRACK_IOU


def adopt(seq_id: str, candidates, *, decisions=None, existing=None,
          max_tracks: int = 40, progress=None) -> dict[int, dict[int, list[float]]]:
    """Turn candidate detections into tracks, dropping the ones already covered.

    The same car is found again on every sampled frame it appears in, so the
    candidate list is mostly duplicates of each other. They are resolved the
    only way that is actually reliable: propagate the earliest ones first, then
    test each later candidate against the tracks that already exist on *its own*
    frame. Two detections of one car agree there even when their boxes came from
    frames 40 apart.

    Returns ``{index: {frame_id: xyxy}}``.
    """
    by_frame: dict[int, list[tuple[list[float], float]]] = {}
    for fid, box, sc in candidates:
        by_frame.setdefault(fid, []).append((box, sc))

    from .annotate import propagate_boxes

    adopted: dict[int, dict[int, list[float]]] = {}
    if existing:
        # Pre-seed the overlap test with tracks the reviewer drew earlier, so a
        # sweep run twice does not annotate the same cars a second time.
        adopted.update({-(i + 1): fb for i, fb in enumerate(existing)})
    n = 0
    frames = sorted(by_frame)
    for fi, fid in enumerate(frames):
        if n >= max_tracks:
            break
        taken = np.asarray([fb[fid] for fb in adopted.values() if fid in fb],
                           np.float32).reshape(-1, 4)
        fresh = []
        for box, sc in sorted(by_frame[fid], key=lambda t: -t[1]):
            b = np.asarray(box, np.float32).reshape(1, 4)
            pool = np.concatenate([taken, np.asarray(fresh, np.float32).reshape(-1, 4)]) \
                if len(fresh) or len(taken) else np.zeros((0, 4), np.float32)
            if len(pool) and (_iou_mat(b, pool) > DUP_IOU).any():
                continue
            fresh.append(box)
            if len(fresh) + n >= max_tracks:
                break
        if not fresh:
            continue

        def _p(f, m, _fi=fi, _fid=fid):
            if progress is not None:
                progress((_fi + f) / max(len(frames), 1),
                         f"frame {_fid}: {m}")

        seeds = {n + i: box for i, box in enumerate(fresh)}
        out = propagate_boxes(seq_id, fid, seeds, progress=_p)
        for oid, fb in sorted(out.items()):
            # A candidate found on a frame where an existing track happens to
            # have no box is not a new object — it is that track, seen past the
            # point where it was cut short. Propagation runs both directions, so
            # the duplicate only becomes visible once the whole track exists,
            # which is why this test is here and not on the seed.
            if fb and not any(_same_object(fb, other) for other in adopted.values()):
                adopted[oid] = fb
        n += len(fresh)

    return {k: v for k, v in adopted.items() if k >= 0}


def proposals_view(seq_id: str, candidates, exemplar=None, tile: int = 130,
                   columns: int = 8, context: float = 2.5,
                   max_tiles: int = 96, rejected=None) -> np.ndarray:
    """A wall of every proposal, labelled with its frame and score.

    The canvas can only ever show the proposals that fall on the frame being
    displayed, so a sweep that found 40 objects across 19 frames looks, on the
    canvas, like a sweep that found 10. This is the view that answers "what did
    it actually find" — one tile per proposal, strongest first, each labelled
    with the frame it came from so a doubtful one can be gone to and looked at.

    The exemplar leads the wall, because every score on it is a claim about
    similarity *to that crop*, and judging the claims without seeing what they
    are similar to is guesswork.

    ``rejected`` holds indices into ``candidates`` the reviewer has struck out.
    They stay on the wall, dimmed — removing them would renumber every tile
    after them, so the next click would land on a different object than the one
    under the cursor, and a mis-click would be unrecoverable.
    """
    import cv2

    from .render import _tile_placeholder

    items = list(candidates[:max_tiles])
    if exemplar is not None:
        items = [exemplar] + items
    if not items:
        return _tile_placeholder(tile, "no proposals")

    frames: dict[int, np.ndarray] = {}
    tiles = []
    for n, (fid, box, score) in enumerate(items):
        img = frames.get(fid)
        if img is None:
            img = _read_frame(seq_id, fid)
            frames[fid] = img
        is_exemplar = exemplar is not None and n == 0
        idx = n - (1 if exemplar is not None else 0)
        is_out = (not is_exemplar) and rejected is not None and idx in rejected
        colour = ((60, 130, 255) if is_exemplar
                  else (110, 110, 110) if is_out else (255, 120, 220))
        if img is None:
            tiles.append(_tile_placeholder(tile, f"f{fid}?"))
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        half = max((x2 - x1) * (1 + context) / 2, (y2 - y1) * (1 + context) / 2, 18.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = int(round(cx - half)), int(round(cy - half))
        crop = img[max(oy, 0):min(int(round(cy + half)), H),
                   max(ox, 0):min(int(round(cx + half)), W)]
        if crop.size == 0:
            tiles.append(_tile_placeholder(tile, f"f{fid}"))
            continue
        crop = crop.copy()
        crop = cv2.copyMakeBorder(crop, max(-oy, 0),
                                  max(int(round(cy + half)) - H, 0), max(-ox, 0),
                                  max(int(round(cx + half)) - W, 0),
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
        ch, cw = crop.shape[:2]
        sc = tile / max(ch, cw)
        crop = cv2.resize(crop, (max(int(cw * sc), 1), max(int(ch * sc), 1)),
                          interpolation=cv2.INTER_NEAREST if sc > 1 else cv2.INTER_AREA)
        crop = cv2.copyMakeBorder(crop, 0, tile - crop.shape[0], 0,
                                  tile - crop.shape[1], cv2.BORDER_CONSTANT,
                                  value=(30, 30, 30))
        b = (np.asarray([x1, y1, x2, y2], float)
             - np.array([ox, oy, ox, oy])) * sc
        cv2.rectangle(crop, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colour, 1)
        cv2.rectangle(crop, (0, 0), (tile - 1, tile - 1), colour,
                      3 if is_exemplar else 1)
        if is_out:
            crop = (crop.astype(np.float32) * 0.35).astype(np.uint8)
            cv2.line(crop, (6, 6), (tile - 7, tile - 7), (230, 70, 70), 2)
            cv2.line(crop, (tile - 7, 6), (6, tile - 7), (230, 70, 70), 2)
        # ASCII only: cv2's Hershey fonts have no glyph for anything else and
        # render a multiplication sign as "???".
        label = ("EXEMPLAR" if is_exemplar
                 else f"f{fid}  {score:.2f}" + ("  OUT" if is_out else ""))
        cv2.putText(crop, label, (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (230, 70, 70) if is_out else colour, 1, cv2.LINE_AA)
        tiles.append(crop)

    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i:i + columns]
        while len(row) < columns:
            row.append(np.full((tile, tile, 3), 20, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)
