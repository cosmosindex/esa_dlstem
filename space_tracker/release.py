"""Read the released Space-Tracker package.

This module is the public loading API of the benchmark. It reads the package as
it is downloaded — the two manifests, the per-sequence ground truth, the frames —
and nothing else: no source dataset is needed, and no field is recomputed at read
time, so what you get here is exactly what the package ships.

    from space_tracker import SpaceTracker

    st = SpaceTracker("/path/to/space_tracker")

    for seq in st.sot.filter(split="test", attribute="OCC"):
        for f in seq.frames():
            if f.visible:
                f.box          # (x, y, w, h), absolute pixels, top-left origin
                f.image_path   # pathlib.Path

    for seq in st.mot.filter(split="test", category="car"):
        for frame_id, objects in seq.objects().items():
            for o in objects:
                o.track_id, o.category, o.box

Only the standard library is required. ``load_image`` additionally needs Pillow
or OpenCV, and ``load_coco`` needs ``pycocotools`` unless you pass
``raw=True``.

The sibling module :mod:`space_tracker.manifest` (and its friends) is a
different, internal thing: it indexes the *source* datasets in their original
on-disk formats and is used only to build this package. A reader of the release
wants this module.
"""

from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "SpaceTracker",
    "SOTHalf",
    "MOTHalf",
    "SOTSequence",
    "MOTSequence",
    "SOTFrame",
    "MOTObject",
    "MOTTrack",
]

#: Environment variable consulted when no root is passed explicitly.
ROOT_ENV_VAR = "SPACE_TRACKER_ROOT"

#: Shipped alongside the code so that the splits are available before you have
#: downloaded anything. Overridden by ``<root>/splits.json`` when that exists.
_BUNDLED_SPLITS = Path(__file__).resolve().parent / "splits.json"

_BOX = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SOTFrame:
    """One row of a SOT ``groundtruth.txt``, paired with its image.

    There is exactly one row per frame. When the target is absent from the frame
    the row is written with ``visible = False`` and no geometry, which is how the
    package unifies SatSOT's literal ``none`` with SV248S's separate state file.
    """

    frame_id: int
    image_path: Path
    visible: bool
    box: Optional[_BOX]
    #: SV248S's native flag (0 visible, 1 invisible, 2 occluded); ``None`` elsewhere.
    state: Optional[int]
    #: Oriented-box corners ``(px1, py1, ..., px4, py4)`` where the source
    #: annotates an orientation (OOTB, and SV248S where a polygon was imported);
    #: ``None`` otherwise.
    polygon: Optional[Tuple[float, ...]]

    @property
    def box_xyxy(self) -> Optional[_BOX]:
        if self.box is None:
            return None
        x, y, w, h = self.box
        return (x, y, x + w, y + h)


@dataclass(frozen=True)
class MOTObject:
    """One row of a MOT ``gt/gt.txt``."""

    frame_id: int
    track_id: int
    box: _BOX
    category_id: int
    category: str
    conf: float = 1.0
    visibility: float = 1.0

    @property
    def box_xyxy(self) -> _BOX:
        x, y, w, h = self.box
        return (x, y, x + w, y + h)


@dataclass(frozen=True)
class MOTTrack:
    """A whole MOT identity, as described by the COCO-VID ``tracks`` table.

    ``is_small`` is the benchmark's own size flag (median ``sqrt(area)`` over the
    track at most 32 px). ``motion_state`` is one of ``moving`` /
    ``intermittent`` / ``static`` / ``unknown`` and also sits beside the ground
    truth in ``gt/motion_state.txt``.
    """

    track_id: int
    category_id: int
    category: str
    n_boxes: int
    is_small: bool
    median_sqrt_area_px: float
    motion_state: str = "unknown"
    moving_fraction: Optional[float] = None
    median_speed_mps: Optional[float] = None
    centre_span_over_size: Optional[float] = None


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #
class _Sequence:
    """Shared behaviour of a SOT and a MOT sequence."""

    def __init__(self, record: Dict[str, Any], root: Path, split: Optional[str]):
        self._r = record
        self.root = root
        self.split = split

    # -- identity ----------------------------------------------------------- #
    @property
    def id(self) -> str:
        """Released sequence name, e.g. ``mixed_sdmcar_0030``."""
        return self._r["id"]

    name = id

    @property
    def dataset(self) -> str:
        """Source dataset key, e.g. ``sdmcar``."""
        return self._r["dataset"]

    @property
    def source_sequence_id(self) -> str:
        """The source dataset's own name for this sequence, e.g. ``satmtb/car/01``."""
        return self._r["source_sequence_id"]

    @property
    def category(self) -> str:
        """Class folder this sequence lives in (``mixed`` for multi-class MOT)."""
        return self._r["category"]

    # -- acquisition -------------------------------------------------------- #
    @property
    def platform(self) -> Optional[str]:
        """Imaging platform(s) the source dataset names, or ``None``."""
        return self._r.get("platform")

    @property
    def gsd_m(self) -> Optional[float]:
        """Ground sample distance in metres, or ``None`` where no source states one.

        With :attr:`fps` this converts a displacement in pixels per frame into
        one in metres per second, which is how ``motion_state`` was assigned.
        Known for all 403 MOT sequences; unknown for the 149 SOT sequences whose
        source mixes imaging platforms without saying which filmed what.
        """
        return self._r.get("gsd_m")

    @property
    def fps(self) -> Optional[float]:
        """Acquisition frame rate, or ``None`` where no source states one."""
        return self._r.get("fps")

    @property
    def acquisition(self) -> Dict[str, Any]:
        """:attr:`platform`, :attr:`gsd_m` and :attr:`fps` with their provenance.

        Each value is tagged ``paper`` (stated by the source publication),
        ``derived`` (measured from the distributed files), ``inherited`` (taken
        from a matched parent scene) or ``unverified`` (the value is ``None``).
        Nothing here is guessed.
        """
        return {k: self._r.get(k) for k in
                ("platform", "gsd_m", "gsd_source", "fps", "fps_source")}

    # -- geometry / extent -------------------------------------------------- #
    @property
    def n_frames(self) -> int:
        return self._r["n_frames"]

    @property
    def width(self) -> int:
        return self._r["img_width"]

    @property
    def height(self) -> int:
        return self._r["img_height"]

    # -- paths -------------------------------------------------------------- #
    @property
    def dir(self) -> Path:
        """Absolute path of the sequence directory."""
        return self.root / self._r["path"]

    @property
    def gt_path(self) -> Path:
        return self.root / self._r["gt_path"]

    @property
    def frame_ids(self) -> range:
        """Frame ids, 1-indexed throughout the package."""
        return range(1, self.n_frames + 1)

    def image_path(self, frame_id: int) -> Path:
        """Absolute path of one frame. Frame ids are 1-indexed."""
        return self.root / self._r["image_path_pattern"].format(frame_id=frame_id)

    def load_image(self, frame_id: int):
        """Decode one frame to an RGB ``numpy`` array. Needs Pillow or OpenCV."""
        path = str(self.image_path(frame_id))
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pass
        else:
            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "load_image needs Pillow or OpenCV; the rest of this module does not."
            ) from exc
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img[:, :, ::-1]

    @cached_property
    def seqinfo(self) -> Dict[str, str]:
        """``seqinfo.ini`` as a flat dict. ``frameRate`` is ``-1`` on purpose:
        none of the seven sources publishes a capture rate."""
        parser = configparser.ConfigParser()
        parser.read(self.dir / "seqinfo.ini")
        return dict(parser["Sequence"]) if parser.has_section("Sequence") else {}

    @property
    def record(self) -> Dict[str, Any]:
        """The raw manifest entry, for fields this wrapper does not surface."""
        return self._r

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} {self.id} {self.n_frames}f {self.width}x{self.height}>"


class SOTSequence(_Sequence):
    """One single-object sequence: one target, one box per frame."""

    @property
    def n_visible_frames(self) -> int:
        return self._r["n_visible_frames"]

    @property
    def median_sqrt_area_px(self) -> float:
        """Median ``sqrt(area)`` of the target over the sequence, in pixels."""
        return self._r["median_sqrt_area_px"]

    @property
    def native_attrs(self) -> List[str]:
        """Attribute labels exactly as the source dataset wrote them."""
        return list(self._r["native_attrs"])

    @property
    def unified_attrs(self) -> List[str]:
        """This sequence's *pooled* attributes: the five more than one source
        annotates, which may therefore be scored across the whole benchmark.

        A subset of :attr:`taxonomy_attrs`, never a replacement for it. An
        attribute missing here is still annotated and still evaluable -- it is
        reported on its one annotating source instead of pooled.
        """
        return list(self._r["unified_attrs"])

    pooled_attrs = unified_attrs

    @property
    def taxonomy_attrs(self) -> List[str]:
        """Every taxonomy attribute this sequence carries -- all 18 are reachable,
        plus any occlusion sub-type that applies. This is the complete list."""
        return list(self._r["taxonomy_attrs"])

    def has_attribute(self, attr: str) -> bool:
        return attr in self._r["taxonomy_attrs"]

    @cached_property
    def groundtruth(self) -> List[SOTFrame]:
        """Every row of ``groundtruth.txt``, in frame order.

        The file is 15 columns: ``frame_id, x, y, w, h, visible, state,
        px1..py4``. Absent targets carry all-``-1`` geometry.
        """
        out: List[SOTFrame] = []
        for line in self.gt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            f = line.split(",")
            frame_id = int(float(f[0]))
            x, y, w, h = (float(v) for v in f[1:5])
            visible = bool(int(float(f[5])))
            state = int(float(f[6]))
            poly = tuple(float(v) for v in f[7:15]) if len(f) >= 15 else ()
            out.append(
                SOTFrame(
                    frame_id=frame_id,
                    image_path=self.image_path(frame_id),
                    visible=visible,
                    box=(x, y, w, h) if visible else None,
                    state=state if state >= 0 else None,
                    polygon=poly if poly and poly[0] != -1 else None,
                )
            )
        return out

    def frames(self, visible_only: bool = False) -> Iterator[SOTFrame]:
        """Iterate the ground truth. ``visible_only`` skips absent-target frames."""
        for f in self.groundtruth:
            if visible_only and not f.visible:
                continue
            yield f

    @property
    def init_box(self) -> _BOX:
        """The first visible box — what a single-object tracker is initialised with."""
        for f in self.groundtruth:
            if f.visible and f.box is not None:
                return f.box
        raise ValueError(f"{self.id} has no visible frame")


class MOTSequence(_Sequence):
    """One multi-object sequence: every object in the field of view is annotated."""

    def __init__(self, record, root, split, categories: Dict[int, str], tracks_index):
        super().__init__(record, root, split)
        self._categories = categories
        self._tracks_index = tracks_index

    @property
    def categories_in_sequence(self) -> List[str]:
        """Classes actually present in the boxes — not the folder name."""
        return list(self._r["categories_in_sequence"])

    @property
    def n_tracks(self) -> int:
        return self._r["n_tracks"]

    @property
    def n_small_tracks(self) -> int:
        return self._r["n_small_tracks"]

    @property
    def n_boxes(self) -> int:
        return self._r["n_boxes"]

    @property
    def review(self) -> Dict[str, Any]:
        """This sequence's audit trail from the annotation tool."""
        return dict(self._r.get("review", {}))

    @cached_property
    def objects_by_frame(self) -> Dict[int, List[MOTObject]]:
        """``{frame_id: [MOTObject, ...]}`` parsed from ``gt/gt.txt``.

        MOTChallenge 9 columns: ``frame, id, x, y, w, h, conf, class,
        visibility``. ``conf`` and ``visibility`` are 1 on every row — the
        sources annotate neither, and the columns are kept so that MOTChallenge
        parsers read the file unmodified.
        """
        out: Dict[int, List[MOTObject]] = {}
        for line in self.gt_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            frame_id = int(float(f[0]))
            cat_id = int(float(f[7]))
            out.setdefault(frame_id, []).append(
                MOTObject(
                    frame_id=frame_id,
                    track_id=int(float(f[1])),
                    box=(float(f[2]), float(f[3]), float(f[4]), float(f[5])),
                    category_id=cat_id,
                    category=self._categories.get(cat_id, str(cat_id)),
                    conf=float(f[6]),
                    visibility=float(f[8]) if len(f) > 8 else 1.0,
                )
            )
        return out

    def objects(self, frame_id: Optional[int] = None):
        """All objects, or just those in one frame."""
        if frame_id is None:
            return self.objects_by_frame
        return self.objects_by_frame.get(frame_id, [])

    def frames(self) -> Iterator[Tuple[int, List[MOTObject]]]:
        """``(frame_id, objects)`` for every frame, including empty ones."""
        by_frame = self.objects_by_frame
        for frame_id in self.frame_ids:
            yield frame_id, by_frame.get(frame_id, [])

    @cached_property
    def tracks(self) -> Dict[int, MOTTrack]:
        """``{track_id: MOTTrack}`` — size flag and motion state per identity.

        Read from ``gt/motion_state.txt``, which is written from the same pass as
        the COCO-VID ``tracks`` table and carries the same values.
        """
        path = self.dir / "gt" / "motion_state.txt"
        motion: Dict[int, Tuple[str, float, float, float]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                f = line.split(",")
                motion[int(float(f[0]))] = (
                    f[1],
                    _maybe_float(f[2]),
                    _maybe_float(f[3]),
                    _maybe_float(f[4]),
                )
        out: Dict[int, MOTTrack] = {}
        for rec in self._tracks_index.get(self.id, []):
            tid = rec["local_track_id"]
            m = motion.get(tid, (rec.get("motion_state", "unknown"), None, None, None))
            out[tid] = MOTTrack(
                track_id=tid,
                category_id=rec["category_id"],
                category=self._categories.get(rec["category_id"], str(rec["category_id"])),
                n_boxes=rec["n_boxes"],
                is_small=bool(rec["is_small"]),
                median_sqrt_area_px=rec["median_sqrt_area_px"],
                motion_state=m[0],
                moving_fraction=m[1],
                median_speed_mps=m[2],
                centre_span_over_size=m[3],
            )
        return out


def _maybe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# halves
# --------------------------------------------------------------------------- #
class _Half:
    """One task half of the package (``sot/`` or ``mot/``)."""

    task = ""

    def __init__(self, root: Path, splits: Dict[str, str]):
        self.root = Path(root)
        self.manifest_path = self.root / self.task / f"space_tracker_{self.task}.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path} not found — is {self.root} the root of the "
                f"downloaded package (the directory holding sot/ and mot/)?"
            )
        self.manifest: Dict[str, Any] = json.loads(self.manifest_path.read_text())
        self._splits = splits
        self._seqs = [self._wrap(r) for r in self.manifest["sequences"]]
        self._by_id = {s.id: s for s in self._seqs}

    # -- construction ------------------------------------------------------- #
    def _wrap(self, record):  # pragma: no cover - overridden
        raise NotImplementedError

    # -- container protocol ------------------------------------------------- #
    @property
    def sequences(self) -> List[_Sequence]:
        return list(self._seqs)

    @property
    def ids(self) -> List[str]:
        return [s.id for s in self._seqs]

    def __len__(self) -> int:
        return len(self._seqs)

    def __iter__(self):
        return iter(self._seqs)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._by_id[key]
            except KeyError:
                raise KeyError(f"no sequence {key!r} in the {self.task.upper()} half") from None
        return self._seqs[key]

    def __contains__(self, key: str) -> bool:
        return key in self._by_id

    def get(self, key: str, default=None):
        return self._by_id.get(key, default)

    # -- metadata ----------------------------------------------------------- #
    @property
    def categories(self) -> Dict[str, int]:
        """``{name: category_id}``, 1-indexed like everything else."""
        return dict(self.manifest["categories"])

    @property
    def coco_path(self) -> Path:
        """The COCO-VID annotation file covering every sequence in this half."""
        return self.root / self.manifest["coco_annotations"]

    def per_class_coco_path(self, category: str) -> Path:
        return self.root / self.task / "annotations" / "per_class" / f"{category}.json"

    def load_coco(self, category: Optional[str] = None, raw: bool = False):
        """Open the COCO-VID annotations, whole or for one class.

        ``raw=True`` returns the parsed JSON — the video extension fields
        (``videos``, ``tracks``, ``video_id``, ``frame_id``, ``track_id``) that
        ``pycocotools`` does not model live there.
        """
        path = self.coco_path if category is None else self.per_class_coco_path(category)
        if raw:
            return json.loads(path.read_text())
        from pycocotools.coco import COCO

        return COCO(str(path))

    # -- selection ---------------------------------------------------------- #
    def filter(
        self,
        split: Optional[str] = None,
        category: Optional[str] = None,
        dataset: Optional[str] = None,
        ids: Optional[Iterable[str]] = None,
        predicate: Optional[Callable[[Any], bool]] = None,
        **extra,
    ) -> List[_Sequence]:
        """Select sequences. Every argument accepts a string or a list of them.

        ``split``    ``train`` / ``val`` / ``test``.
        ``category`` class folder; for MOT use ``contains`` to ask what a
                     sequence actually holds rather than what it is filed under.
        ``dataset``  source dataset key.
        ``predicate`` an arbitrary callable, for anything not covered here.
        """
        out = self._seqs
        if split is not None:
            want = _as_set(split)
            out = [s for s in out if s.split in want]
        if category is not None:
            want = _as_set(category)
            out = [s for s in out if s.category in want]
        if dataset is not None:
            want = _as_set(dataset)
            out = [s for s in out if s.dataset in want]
        if ids is not None:
            want = _as_set(ids)
            out = [s for s in out if s.id in want]
        out = self._filter_extra(out, **extra)
        if predicate is not None:
            out = [s for s in out if predicate(s)]
        return out

    def _filter_extra(self, seqs, **extra):
        if extra:
            raise TypeError(f"unexpected filter argument(s): {', '.join(sorted(extra))}")
        return seqs

    def counts(self) -> Dict[str, int]:
        """Sequences per split, for a quick sanity check after downloading."""
        out: Dict[str, int] = {}
        for s in self._seqs:
            out[s.split or "unassigned"] = out.get(s.split or "unassigned", 0) + 1
        return out

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} {len(self._seqs)} sequences at {self.root}>"


class SOTHalf(_Half):
    """``sot/`` — 395 sequences, one annotated target each."""

    task = "sot"

    def _wrap(self, record):
        return SOTSequence(record, self.root, self._splits.get(f"sot/{record['id']}"))

    @property
    def attribute_taxonomy(self) -> Dict[str, Any]:
        """Every attribute key, with its group, full name, definition and the
        source labels it was merged from. This is the single definition.

        It holds the 18 taxonomy attributes plus the five sub-types that drill
        into the pooled ``OCC`` row; use :attr:`attributes` for the 18 alone.
        Each entry carries ``pooled``: whether the row may be scored across
        sources. Every entry is annotated and filterable either way.
        """
        attrs = self.manifest["attribute_taxonomy"]["attributes"]
        return attrs[0] if isinstance(attrs, list) else attrs

    @property
    def attributes(self) -> List[str]:
        """All 18 attributes of the taxonomy, in taxonomy order.

        Every one of them is annotated on the released sequences and every one
        is filterable through ``filter(attribute=...)``. Five of them are
        additionally *pooled* (see :attr:`pooled_attributes`); the rest are
        reported on their single annotating source. That distinction is about
        how a row is scored, not about what is attached to a sequence.
        """
        return [
            k
            for k, v in self.attribute_taxonomy.items()
            if v.get("group") != "occlusion_subtypes"
        ]

    @property
    def pooled_attributes(self) -> List[str]:
        """The five attributes more than one source annotates among the released
        sequences, so that a score over them spans datasets rather than one.

        ``DEF`` and ``ARC`` are defined by two source datasets each, but the
        32 px size filter leaves DEF in OOTB alone and ARC in SatSOT alone, so
        neither can be pooled here. Both remain in :attr:`attributes`.
        """
        return [k for k, v in self.attribute_taxonomy.items() if v.get("pooled")]

    @property
    def occlusion_subtypes(self) -> List[str]:
        """The five source labels the unified ``OCC`` row is made of. They are a
        drill-down, not extra rows of the taxonomy."""
        return [
            k
            for k, v in self.attribute_taxonomy.items()
            if v.get("group") == "occlusion_subtypes"
        ]

    def _filter_extra(
        self,
        seqs,
        attribute: Optional[str] = None,
        unified_attribute: Optional[str] = None,
        max_size: Optional[float] = None,
        min_size: Optional[float] = None,
        **rest,
    ):
        if attribute is not None:
            want = _as_set(attribute)
            known = set(self.attributes) | set(self.occlusion_subtypes)
            unknown = want - known
            if unknown:
                raise ValueError(f"unknown attribute(s): {sorted(unknown)}. "
                                 f"Known: {sorted(known)}")
            seqs = [s for s in seqs if want & set(s.taxonomy_attrs)]
        if unified_attribute is not None:
            want = _as_set(unified_attribute)
            unknown = want - set(self.pooled_attributes)
            if unknown:
                raise ValueError(
                    f"not pooled: {sorted(unknown)}. Pooled attributes are "
                    f"{self.pooled_attributes}; every other attribute is "
                    f"annotated by one source and is reached with "
                    f"attribute=... instead.")
            seqs = [s for s in seqs if want & set(s.unified_attrs)]
        if max_size is not None:
            seqs = [s for s in seqs if s.median_sqrt_area_px <= max_size]
        if min_size is not None:
            seqs = [s for s in seqs if s.median_sqrt_area_px > min_size]
        return super()._filter_extra(seqs, **rest)


class MOTHalf(_Half):
    """``mot/`` — 403 sequences, every object in frame annotated."""

    task = "mot"

    def __init__(self, root: Path, splits: Dict[str, str]):
        # Both of these are handed to every sequence by reference and filled in
        # afterwards: the category map once the manifest is read, the track index
        # only if some sequence actually asks for its tracks.
        self._id_to_cat: Dict[int, str] = {}
        self._tracks_index = _LazyTracks(self)
        super().__init__(root, splits)
        self._id_to_cat.update({v: k for k, v in self.categories.items()})

    def _wrap(self, record):
        return MOTSequence(
            record,
            self.root,
            self._splits.get(f"mot/{record['id']}"),
            self._id_to_cat,
            self._tracks_index,
        )

    @cached_property
    def _coco_tracks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Lazily index the COCO-VID ``tracks`` table by sequence name.

        Only touched the first time a sequence's ``.tracks`` is asked for, since
        it parses the whole annotation file.
        """
        data = json.loads(self.coco_path.read_text())
        video_name = {v["id"]: v["name"] for v in data["videos"]}
        index: Dict[str, List[Dict[str, Any]]] = {}
        for t in data["tracks"]:
            index.setdefault(video_name[t["video_id"]], []).append(t)
        self._tracks_index.update(index)
        return index

    def _filter_extra(
        self,
        seqs,
        contains: Optional[str] = None,
        has_small_tracks: Optional[bool] = None,
        **rest,
    ):
        if contains is not None:
            want = _as_set(contains)
            seqs = [s for s in seqs if want & set(s.categories_in_sequence)]
        if has_small_tracks is not None:
            seqs = [s for s in seqs if (s.n_small_tracks > 0) == has_small_tracks]
        return super()._filter_extra(seqs, **rest)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
class SpaceTracker:
    """The downloaded package, both halves.

    ``root`` is the directory that holds ``sot/`` and ``mot/``. It defaults to
    ``$SPACE_TRACKER_ROOT``.
    """

    def __init__(self, root: Optional[os.PathLike] = None, splits: Optional[os.PathLike] = None):
        root = root or os.environ.get(ROOT_ENV_VAR)
        if not root:
            raise ValueError(
                f"pass the package root, or set {ROOT_ENV_VAR} to the directory "
                f"holding sot/ and mot/"
            )
        self.root = Path(root).expanduser().resolve()
        self.splits = _load_splits(self.root, splits)
        self.sot = SOTHalf(self.root, self.splits)
        self.mot = MOTHalf(self.root, self.splits)

    def __getitem__(self, task: str) -> _Half:
        return {"sot": self.sot, "mot": self.mot}[task]

    def sequence(self, seq_id: str) -> _Sequence:
        """Look a sequence up by name in whichever half holds it."""
        for half in (self.sot, self.mot):
            if seq_id in half:
                return half[seq_id]
        raise KeyError(f"no sequence {seq_id!r} in either half")

    def summary(self) -> str:
        """One line per half — run this first to check the download."""
        lines = []
        for half in (self.sot, self.mot):
            counts = half.counts()
            order = ("train", "val", "test", "unassigned")
            parts = ", ".join(f"{k} {counts[k]}" for k in order if k in counts)
            lines.append(f"{half.task.upper()}: {len(half)} sequences ({parts})")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<SpaceTracker root={self.root} sot={len(self.sot)} mot={len(self.mot)}>"


class _LazyTracks(dict):
    """Defers parsing the COCO-VID file until a sequence's tracks are asked for."""

    def __init__(self, half: MOTHalf):
        super().__init__()
        self._half = half

    def get(self, key, default=None):
        if not self:
            self.update(self._half._coco_tracks)
        return super().get(key, default)


def _load_splits(root: Path, explicit: Optional[os.PathLike]) -> Dict[str, str]:
    """``{'sot/<id>': 'train', 'mot/<id>': 'test', ...}``.

    Preferred order: the file you pass, then ``<root>/splits.json`` if the
    package you downloaded carries one, then the copy shipped with this code.
    """
    for path in (explicit, root / "splits.json", _BUNDLED_SPLITS):
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            out: Dict[str, str] = {}
            for task, mapping in data["splits"].items():
                for key, value in mapping.items():
                    out[key if "/" in key else f"{task}/{key}"] = value
            return out
    return {}


def _as_set(value) -> set:
    if isinstance(value, str):
        return {value}
    return set(value)
