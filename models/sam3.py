"""
SAM3 tracker wrapper.

Thin wrapper around the SAM 3 video tracker that exposes the same stateful
interface as SAM2Tracker, so it can be used interchangeably with
`VideoTrackerEvaluationModule`:

    init_video → add_prompts → propagate → reset_state

Internally it uses the `tracker` sub-model of SAM 3 (which is API-compatible
with SAM 2's `SAM2VideoPredictor`), sharing the detector backbone.

Notes
-----
* SAM 3's `add_new_points_or_box` expects **relative coordinates in [0, 1]**
  (with `rel_coordinates=True`). We convert xyxy pixel boxes accordingly.
* `init_state` only accepts a JPEG folder or video path, so we dump frames
  to a temporary directory (same trick as SAM2Tracker).
* `propagate_in_video` yields 5-tuples in SAM3
  `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)`,
  whereas SAM2 yielded only `(frame_idx, obj_ids, mask_logits)`. We use
  `video_res_masks` (full-resolution mask logits) and `obj_scores` for the
  per-object confidence.
* The first call to `propagate_in_video` needs `propagate_preflight=True`.
"""

import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from obb_utils import mask_to_obb, mask_to_aabb


class SAM3Tracker(nn.Module):
    """
    SAM3 video tracker wrapped as an nn.Module, mirroring `SAM2Tracker`'s API.

    Args:
        checkpoint_path:        Optional path to a local SAM 3 checkpoint.
                                If None, the model is fetched from Hugging Face
                                (requires `hf auth login` and SAM 3 access).
        apply_temporal_disambiguation:
                                Forwarded to `build_sam3_video_model`.
                                True enables the temporal disambiguation
                                heuristics used in the official SAM 3 release.
        offload_video_to_cpu:   Offload decoded frames to CPU RAM to save GPU.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        apply_temporal_disambiguation: bool = True,
        offload_video_to_cpu: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.apply_temporal_disambiguation = apply_temporal_disambiguation
        self.offload_video_to_cpu = offload_video_to_cpu

        self.predictor = self._build_predictor()

        # Internal video state (reset between sequences)
        self._inference_state = None
        self._tmp_dir: str | None = None
        self._video_h: int | None = None
        self._video_w: int | None = None
        # Mapping from obj_id → class label (set during add_prompts)
        self._obj_id_to_label: dict[int, int] = {}
        # Earliest frame index that received a prompt — used as propagation start
        self._min_prompt_frame: int | None = None
        # Highest frame index seen in init_video
        self._num_frames: int = 0

    def _build_predictor(self):
        """Build the SAM3 video tracker, sharing the detector backbone."""
        # setuptools >= 81 dropped `pkg_resources`, but sam3.model_builder still
        # imports it at module top to locate the BPE asset. Inject a minimal
        # shim before importing so we don't need to patch sam3 or downgrade
        # setuptools.
        self._ensure_pkg_resources_shim()

        # Locate the BPE asset path explicitly (also used by the shim fallback).
        import sam3 as _sam3_pkg
        from pathlib import Path
        sam3_pkg_dir = Path(_sam3_pkg.__path__[0])
        bpe_path = str(sam3_pkg_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz")

        from sam3.model_builder import build_sam3_video_model

        sam3_model = build_sam3_video_model(
            checkpoint_path=self.checkpoint_path,
            bpe_path=bpe_path,
            apply_temporal_disambiguation=self.apply_temporal_disambiguation,
        )
        tracker = sam3_model.tracker
        # The tracker re-uses the detector's backbone for feature extraction
        tracker.backbone = sam3_model.detector.backbone
        tracker.eval()
        return tracker

    @staticmethod
    def _ensure_pkg_resources_shim():
        """Inject a minimal `pkg_resources` replacement if it's missing.

        setuptools >= 81 removed `pkg_resources`. sam3.model_builder still
        uses `pkg_resources.resource_filename(pkg, path)` to locate asset
        files, so we provide a small shim backed by `importlib`.
        """
        import sys
        try:
            import pkg_resources  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        import types
        import importlib
        from pathlib import Path

        shim = types.ModuleType("pkg_resources")

        def resource_filename(package: str, resource: str) -> str:
            mod = importlib.import_module(package)
            base = Path(mod.__path__[0])
            return str(base / resource)

        shim.resource_filename = resource_filename
        sys.modules["pkg_resources"] = shim

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """
        Initialise SAM3 video state from a list of frames.

        Args:
            frames: list of H×W×3 uint8 numpy arrays (RGB).
        """
        # SAM 3 expects a directory of JPEG frames or a video path
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._video_h, self._video_w = frames[0].shape[:2]
        self._num_frames = len(frames)

        self._inference_state = self.predictor.init_state(
            video_path=self._tmp_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
        )
        self._obj_id_to_label.clear()
        self._min_prompt_frame = None

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """
        Add bounding-box prompts for objects on a given frame.

        Args:
            frame_idx: Index of the frame (0-based within the clip).
            boxes:     (N, 4) xyxy absolute-pixel boxes.
            labels:    (N,) int class labels. Stored for output assignment.
            obj_ids:   Optional list of N integer object IDs; defaults to 1…N.
        """
        if len(boxes) == 0:
            return

        if obj_ids is None:
            obj_ids = list(range(1, len(boxes) + 1))

        if labels is not None:
            for oid, lbl in zip(obj_ids, labels):
                self._obj_id_to_label[int(oid)] = int(lbl)

        W, H = self._video_w, self._video_h
        for obj_id, box in zip(obj_ids, boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            rel_box = torch.tensor(
                [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32
            )
            self.predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=rel_box,
                rel_coordinates=True,
                clear_old_points=True,
            )

        # Track the earliest prompt frame (propagation starts there)
        if self._min_prompt_frame is None or frame_idx < self._min_prompt_frame:
            self._min_prompt_frame = frame_idx

    def propagate(self) -> list[dict]:
        """
        Propagate masks through the video after prompts have been added.

        Runs forward (and, if prompts are placed past frame 0, also backward)
        from the earliest prompted frame.

        Returns:
            List of per-frame dicts (ordered by frame index):
            {
                'boxes':     (N, 4) xyxy float tensor,
                'obb':       (N, 8) OBB corner float tensor (from minAreaRect),
                'labels':    (N,)   long  tensor,
                'scores':    (N,)   float tensor (object score sigmoid),
                'track_ids': (N,)   long  tensor,
            }
        """
        if self._min_prompt_frame is None:
            return [self._empty_output() for _ in range(self._num_frames)]

        frame_outputs: dict[int, dict] = {}

        # Forward pass from the earliest prompt frame
        for it in self.predictor.propagate_in_video(
            self._inference_state,
            start_frame_idx=self._min_prompt_frame,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            self._collect_frame(it, frame_outputs)

        # Backward pass if prompts aren't already at frame 0
        if self._min_prompt_frame > 0:
            for it in self.predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=self._min_prompt_frame,
                max_frame_num_to_track=None,
                reverse=True,
                propagate_preflight=False,
                tqdm_disable=True,
            ):
                self._collect_frame(it, frame_outputs)

        max_idx = max(frame_outputs.keys()) if frame_outputs else -1
        total = max(max_idx + 1, self._num_frames)
        return [frame_outputs.get(i, self._empty_output()) for i in range(total)]

    def _collect_frame(self, iteration_out, frame_outputs: dict):
        """Parse one `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)` tuple."""
        frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores = iteration_out

        # video_res_masks: (N_obj, 1, H, W) float — threshold at 0 (logit)
        masks = (video_res_masks > 0.0).cpu()
        # obj_scores: (N_obj, 1) logit; sigmoid → confidence
        scores_tensor = torch.sigmoid(obj_scores.float().cpu()).flatten()

        boxes_list, obb_list, scores_list, ids_list, labels_list = [], [], [], [], []
        for i, obj_id in enumerate(obj_ids):
            mask_2d = masks[i, 0].numpy()  # (H, W) bool
            if not mask_2d.any():
                continue
            # Tight AABB from the mask itself (best for HBB GT datasets)
            box_xyxy = mask_to_aabb(mask_2d)
            if box_xyxy is None:
                continue
            # OBB from cv2.minAreaRect (best for OBB GT datasets)
            obb_8 = mask_to_obb(mask_2d)
            if obb_8 is None:
                continue

            boxes_list.append(torch.from_numpy(box_xyxy))
            obb_list.append(torch.from_numpy(obb_8))
            scores_list.append(float(scores_tensor[i]))
            ids_list.append(int(obj_id))
            labels_list.append(self._obj_id_to_label.get(int(obj_id), 0))

        if boxes_list:
            frame_outputs[int(frame_idx)] = {
                "boxes":     torch.stack(boxes_list),
                "obb":       torch.stack(obb_list),
                "labels":    torch.tensor(labels_list, dtype=torch.long),
                "scores":    torch.tensor(scores_list, dtype=torch.float32),
                "track_ids": torch.tensor(ids_list, dtype=torch.long),
            }
        else:
            frame_outputs[int(frame_idx)] = self._empty_output()

    def reset_state(self):
        """Reset video state and clean up temporary files."""
        if self._inference_state is not None:
            try:
                self.predictor.clear_all_points_in_video(self._inference_state)
            except Exception:
                pass
            self._inference_state = None
        self._obj_id_to_label.clear()
        self._min_prompt_frame = None
        self._num_frames = 0

        if self._tmp_dir is not None:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_output() -> dict:
        return {
            "boxes":     torch.zeros((0, 4), dtype=torch.float32),
            "obb":       torch.zeros((0, 8), dtype=torch.float32),
            "labels":    torch.zeros(0, dtype=torch.long),
            "scores":    torch.zeros(0, dtype=torch.float32),
            "track_ids": torch.zeros(0, dtype=torch.long),
        }


class SAM3TextTracker(nn.Module):
    """
    Text-prompted SAM3 for open-vocabulary video MOT.

    Instead of taking per-object box prompts (SOT style), this wrapper takes a
    single text prompt *per video* (set via :meth:`set_text_prompt` before
    :meth:`propagate`) and runs SAM 3's combined detector+tracker pipeline once
    with that noun phrase.  For AIR-MOT-style datasets where each video has a
    known dominant class (``"airplane"`` or ``"car"``), this means one SAM 3
    invocation per clip — SAM 3 detects every instance of that class in every
    frame and its temporal-disambiguation tracker links them with consistent
    IDs.

    If no text prompt is set before ``propagate()``, the wrapper falls back to
    looping over ``class_names`` and merging the results — useful when the
    dataset does not expose a per-video dominant class.

    The underlying pattern is the benchmark-eval path implemented in
    ``Sam3VideoInferenceWithInstanceInteractivity.forward()``
    (sam3/model/sam3_video_inference.py:909) — for each noun phrase, call
    ``add_prompt(text_str=name)`` → ``propagate_in_video`` → ``reset_state``.

    Track IDs are made globally unique across classes by offsetting each
    class's local obj_ids by the running max across previously-processed
    classes.  Track IDs are *not* stable across clips of the same video —
    SAM 3 is re-run from scratch per clip, so configure ``clip_len`` large
    enough to cover full sequences when possible.

    Args:
        class_names:    Ordered list of text prompts (one per class).
        label_to_id:    Map from class name → integer label id, used to fill
                        the ``labels`` field of each per-frame output dict.
        checkpoint_path / apply_temporal_disambiguation / offload_video_to_cpu:
                        Same meaning as :class:`SAM3Tracker`.
    """

    def __init__(
        self,
        class_names: list[str],
        label_to_id: dict[str, int],
        checkpoint_path: str | None = None,
        apply_temporal_disambiguation: bool = True,
        offload_video_to_cpu: bool = True,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.label_to_id = dict(label_to_id)
        self.checkpoint_path = checkpoint_path
        self.apply_temporal_disambiguation = apply_temporal_disambiguation
        self.offload_video_to_cpu = offload_video_to_cpu

        self.predictor = self._build_predictor()

        self._inference_state = None
        self._tmp_dir: str | None = None
        self._video_h: int | None = None
        self._video_w: int | None = None
        self._num_frames: int = 0
        # Text prompt set externally per clip; None → fall back to all classes
        self._current_text: str | None = None

    def set_text_prompt(self, text: str | None):
        """Set the text prompt to use for the next :meth:`propagate` call."""
        self._current_text = text if text else None

    def _build_predictor(self):
        """Build the full SAM3 video model (detector + tracker combined)."""
        SAM3Tracker._ensure_pkg_resources_shim()

        import sam3 as _sam3_pkg
        from pathlib import Path
        sam3_pkg_dir = Path(_sam3_pkg.__path__[0])
        bpe_path = str(sam3_pkg_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz")

        from sam3.model_builder import build_sam3_video_model

        model = build_sam3_video_model(
            checkpoint_path=self.checkpoint_path,
            bpe_path=bpe_path,
            apply_temporal_disambiguation=self.apply_temporal_disambiguation,
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Dump RGB frames to a tmp dir and initialise SAM3 inference state."""
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3txt_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._video_h, self._video_w = frames[0].shape[:2]
        self._num_frames = len(frames)

        self._inference_state = self.predictor.init_state(
            resource_path=self._tmp_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
        )

    def add_prompts(self, frame_idx, boxes, labels=None, obj_ids=None):
        """No-op. Text prompts are added internally during ``propagate()``.

        Kept as a method so ``VideoTrackerEvaluationModule`` can call it
        without a type check; any boxes passed by the eval module are
        ignored.
        """
        return

    def propagate(self) -> list[dict]:
        """Run detector+tracker with the current text prompt.

        If a per-clip prompt was set via :meth:`set_text_prompt`, only that
        single noun phrase is used (one SAM 3 invocation).  Otherwise the
        wrapper falls back to looping over all ``class_names`` and merging
        the results.
        """
        if self._inference_state is None or self._num_frames == 0:
            return [self._empty_output() for _ in range(self._num_frames)]

        W, H = self._video_w, self._video_h
        frame_accum: dict[int, list[dict]] = {i: [] for i in range(self._num_frames)}

        if self._current_text is not None:
            prompts_to_run = [self._current_text]
        else:
            prompts_to_run = self.class_names

        next_global_id = 1
        for class_name in prompts_to_run:
            label_int = self.label_to_id.get(class_name, 0)

            # add_prompt auto-resets state before adding the new text prompt.
            # frame_idx is a formal argument; text prompts apply to all frames.
            self.predictor.add_prompt(
                self._inference_state,
                frame_idx=0,
                text_str=class_name,
            )

            start_offset = next_global_id
            max_local_oid = -1

            for frame_idx, out in self.predictor.propagate_in_video(
                self._inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=self._num_frames,
                reverse=False,
            ):
                obj_ids = out["out_obj_ids"]         # (N,) int64
                probs = out["out_probs"]             # (N,) float
                boxes_xywh = out["out_boxes_xywh"]   # (N, 4) normalized xywh
                masks = out["out_binary_masks"]      # (N, H, W) bool

                for i in range(len(obj_ids)):
                    mask_2d = masks[i]
                    if not mask_2d.any():
                        continue

                    x, y, w, h = [float(v) for v in boxes_xywh[i]]
                    x1, y1 = x * W, y * H
                    x2, y2 = x1 + w * W, y1 + h * H

                    obb_8 = mask_to_obb(mask_2d)
                    if obb_8 is None:
                        continue

                    local_oid = int(obj_ids[i])
                    max_local_oid = max(max_local_oid, local_oid)

                    frame_accum[int(frame_idx)].append({
                        "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "obb": obb_8,
                        "score": float(probs[i]),
                        "label": label_int,
                        "track_id": local_oid + start_offset,
                    })

            if max_local_oid >= 0:
                next_global_id += max_local_oid + 1

            # Reset so the next class starts with a clean state
            self.predictor.reset_state(self._inference_state)

        out_list = []
        for i in range(self._num_frames):
            objs = frame_accum[i]
            if not objs:
                out_list.append(self._empty_output())
                continue
            out_list.append({
                "boxes":     torch.from_numpy(np.stack([o["box"] for o in objs])),
                "obb":       torch.from_numpy(np.stack([o["obb"] for o in objs])),
                "labels":    torch.tensor([o["label"] for o in objs], dtype=torch.long),
                "scores":    torch.tensor([o["score"] for o in objs], dtype=torch.float32),
                "track_ids": torch.tensor([o["track_id"] for o in objs], dtype=torch.long),
            })
        return out_list

    def reset_state(self):
        if self._inference_state is not None:
            try:
                self.predictor.reset_state(self._inference_state)
            except Exception:
                pass
            self._inference_state = None
        self._num_frames = 0

        if self._tmp_dir is not None:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

        self._current_text = None

    @staticmethod
    def _empty_output() -> dict:
        return SAM3Tracker._empty_output()


class SAM31TextTracker(nn.Module):
    """
    Text-prompted SAM 3.1 (multiplex) MOT wrapper.

    Same surface as :class:`SAM3TextTracker` (init_video / add_prompts /
    propagate / reset_state / set_text_prompt) so that
    ``VideoTrackerEvaluationModule`` and downstream callbacks
    (``MOTFormatDumpCallback``, RAFT filter) can be reused without
    modification. Internally it drives the multiplex predictor via the
    higher-level handle_request / handle_stream_request API exposed by
    ``Sam3MultiplexVideoPredictor``.

    The per-frame output dict matches the one returned by SAM 3 base
    (``boxes`` xyxy, ``obb``, ``labels``, ``scores``, ``track_ids``),
    so no eval-side change is needed when swapping versions.

    Args:
        class_names: Ordered list of text prompts (one per class). Used
            as a fallback when no per-clip dominant class is provided.
        label_to_id: Map from raw class name → integer label id, used to
            populate the ``labels`` field of each per-frame output dict.
        checkpoint_path: Optional local checkpoint path; if None the
            model is downloaded from `facebook/sam3.1` on Hugging Face.
        max_num_objects: SAM 3.1 multiplex bucket size (default 16).
        compile: Forward to `build_sam3_predictor`. Off by default since
            torch.compile compilation can take minutes per first run.
    """

    def __init__(
        self,
        class_names: list[str],
        label_to_id: dict[str, int],
        checkpoint_path: str | None = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        compile: bool = False,
        use_fa3: bool = False,
        use_rope_real: bool = False,
        apply_temporal_disambiguation: bool = True,  # accepted for cfg parity; multiplex always on
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.label_to_id = dict(label_to_id)
        self.checkpoint_path = checkpoint_path
        self.max_num_objects = max_num_objects
        self.multiplex_count = multiplex_count
        self.compile = compile
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real

        self.predictor = self._build_predictor()

        self._tmp_dir: str | None = None
        self._video_h: int | None = None
        self._video_w: int | None = None
        self._num_frames: int = 0
        self._session_id: str | None = None
        self._current_text: str | None = None

    def set_text_prompt(self, text: str | None):
        self._current_text = text if text else None

    def _build_predictor(self):
        SAM3Tracker._ensure_pkg_resources_shim()
        from sam3.model_builder import build_sam3_predictor

        # FA3 is gated behind `flash_attn_interface`, which we do not have
        # installed (and for which there is no prebuilt wheel for our
        # cu12 + torch combo). Force it off so the multiplex predictor
        # falls back to PyTorch SDPA. ``use_rope_real`` only matters for
        # torch.compile compatibility; with compile=False we can leave it
        # off too.
        return build_sam3_predictor(
            checkpoint_path=self.checkpoint_path,
            version="sam3.1",
            compile=self.compile,
            max_num_objects=self.max_num_objects,
            multiplex_count=self.multiplex_count,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
        )

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        # Close any leftover session before starting a new one
        self.reset_state()

        self._tmp_dir = tempfile.mkdtemp(prefix="sam3p1txt_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        self._video_h, self._video_w = frames[0].shape[:2]
        self._num_frames = len(frames)

        resp = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=self._tmp_dir,
            )
        )
        self._session_id = resp["session_id"]

    def add_prompts(self, frame_idx, boxes, labels=None, obj_ids=None):
        """No-op — SAM 3.1 text mode injects prompts inside ``propagate``."""
        return

    def propagate(self) -> list[dict]:
        if self._session_id is None or self._num_frames == 0:
            return [self._empty_output() for _ in range(self._num_frames)]

        W, H = self._video_w, self._video_h
        frame_accum: dict[int, list[dict]] = {i: [] for i in range(self._num_frames)}

        if self._current_text is not None:
            prompts_to_run = [self._current_text]
        else:
            prompts_to_run = self.class_names

        next_global_id = 1
        for class_name in prompts_to_run:
            label_int = self.label_to_id.get(class_name, 0)

            # Reset between text prompts (per the SAM 3.1 example notebook —
            # otherwise add_prompt accumulates with the previous text).
            self.predictor.handle_request(
                request=dict(type="reset_session", session_id=self._session_id)
            )

            self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self._session_id,
                    frame_index=0,
                    text=class_name,
                )
            )

            start_offset = next_global_id
            max_local_oid = -1

            # propagation_direction="forward" — text prompt is at frame 0,
            # we don't need backward.
            #
            # SAM 3.1 multiplex raises
            #   RuntimeError("No points are provided; please add points first")
            # mid-clip when its inner SAM2-style tracker tries to propagate
            # but the outer detector found zero matching objects on every
            # conditioning frame so far. AirMOT in particular triggers this
            # on clips where the open-vocab detector misses every airplane.
            # We catch and break: any frames already yielded are kept,
            # the rest fall back to _empty_output() at assembly time.
            try:
                for response in self.predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=self._session_id,
                        propagation_direction="forward",
                    )
                ):
                    frame_idx = response["frame_index"]
                    out = response["outputs"]
                    obj_ids = out["out_obj_ids"]            # (N,) np int64
                    probs = out["out_probs"]                # (N,) np float
                    boxes_xywh = out["out_boxes_xywh"]      # (N, 4) normalized xywh
                    masks = out["out_binary_masks"]         # (N, H, W) bool

                    for i in range(len(obj_ids)):
                        mask_2d = masks[i]
                        if not mask_2d.any():
                            continue
                        x, y, w, h = [float(v) for v in boxes_xywh[i]]
                        x1, y1 = x * W, y * H
                        x2, y2 = x1 + w * W, y1 + h * H

                        obb_8 = mask_to_obb(mask_2d)
                        if obb_8 is None:
                            continue

                        local_oid = int(obj_ids[i])
                        max_local_oid = max(max_local_oid, local_oid)

                        frame_accum[int(frame_idx)].append({
                            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                            "obb": obb_8,
                            "score": float(probs[i]),
                            "label": label_int,
                            "track_id": local_oid + start_offset,
                        })
            except RuntimeError as exc:
                # Most common cause: outer detector found no matching
                # objects on any conditioning frame, so the inner
                # SAM2-style tracker has nothing to propagate. Drop the
                # rest of this clip's predictions for the current text
                # prompt; per-frame outputs already in `frame_accum`
                # are kept, unprocessed frames fall through to empty.
                print(f"[SAM31TextTracker] propagate aborted for prompt "
                      f"{class_name!r}: {exc}")

            if max_local_oid >= 0:
                next_global_id += max_local_oid + 1

        out_list = []
        for i in range(self._num_frames):
            objs = frame_accum[i]
            if not objs:
                out_list.append(self._empty_output())
                continue
            out_list.append({
                "boxes":     torch.from_numpy(np.stack([o["box"] for o in objs])),
                "obb":       torch.from_numpy(np.stack([o["obb"] for o in objs])),
                "labels":    torch.tensor([o["label"] for o in objs], dtype=torch.long),
                "scores":    torch.tensor([o["score"] for o in objs], dtype=torch.float32),
                "track_ids": torch.tensor([o["track_id"] for o in objs], dtype=torch.long),
            })
        return out_list

    def reset_state(self):
        if self._session_id is not None:
            try:
                self.predictor.handle_request(
                    request=dict(type="close_session", session_id=self._session_id)
                )
            except Exception:
                pass
            self._session_id = None
        self._num_frames = 0
        if self._tmp_dir is not None:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        self._current_text = None

    @staticmethod
    def _empty_output() -> dict:
        return SAM3Tracker._empty_output()
