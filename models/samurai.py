"""
SAMURAI tracker wrapper.

SAMURAI = SAM 2.1 + Kalman-filter motion-aware memory (Yang et al., 2024).
Source: https://github.com/yangchris11/samurai

The upstream repo ships a forked `sam2` package (with samurai_mode hooks and
Kalman filtering) under `samurai/sam2/sam2/`, using the *same* top-level
package name as facebookresearch/sam2. That makes them mutually exclusive
within a single Python process.

To avoid breaking the other eval scripts (which import the pip-installed
`sam2`), we isolate samurai's fork by:

  1. Purging any already-imported `sam2*` modules from sys.modules.
  2. Prepending `samurai/sam2/` to sys.path so its `sam2` wins resolution.
  3. Importing `sam2.build_sam` — this calls `initialize_config_module("sam2")`
     on samurai's package, locking Hydra onto samurai's config tree
     (which contains `configs/samurai/*.yaml`).

For that reason this module is NOT re-exported from `models/__init__.py`.
Import it directly (`from models.samurai import SamuraiTracker`) and do so
*before* any other code touches the installed `sam2`.

Usage mirrors SAM2Tracker:
    tracker = SamuraiTracker(model_name="large")
    tracker.init_video(frames)
    tracker.add_prompts(frame_idx=0, boxes=..., labels=..., obj_ids=...)
    outputs = tracker.propagate()
    tracker.reset_state()
"""

import contextlib
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from obb_utils import mask_to_obb, mask_to_aabb


_SAMURAI_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "samurai", "sam2")
)

_MODEL_NAME_TO_CFG = {
    "tiny":      ("configs/samurai/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt",       "facebook/sam2.1-hiera-tiny"),
    "small":     ("configs/samurai/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt",      "facebook/sam2.1-hiera-small"),
    "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt",  "facebook/sam2.1-hiera-base-plus"),
    "large":     ("configs/samurai/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt",      "facebook/sam2.1-hiera-large"),
}


def _activate_samurai_sam2():
    """Make samurai's bundled sam2 fork the active `sam2` package.

    Idempotent: safe to call multiple times in the same process.
    """
    # Drop any previously imported sam2.* (e.g. from the pip-installed fork)
    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2."):
            del sys.modules[name]

    # Also drop Hydra global state so re-initializing the config module works
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        pass

    if _SAMURAI_ROOT not in sys.path:
        sys.path.insert(0, _SAMURAI_ROOT)
    else:
        # Ensure it wins against site-packages
        sys.path.remove(_SAMURAI_ROOT)
        sys.path.insert(0, _SAMURAI_ROOT)


class SamuraiTracker(nn.Module):
    """
    SAMURAI video tracker wrapped as an nn.Module, mirroring SAM2Tracker's API.

    Args:
        model_name:    One of {"tiny", "small", "base_plus", "large"}.
        ckpt_path:     Optional explicit path to the SAM 2.1 .pt checkpoint.
                       If None, the file is downloaded from HuggingFace
                       (`facebook/sam2.1-hiera-<name>`) and cached locally.
        device:        Torch device for predictor construction.
    """

    def __init__(
        self,
        model_name: str = "large",
        ckpt_path: str | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        if model_name not in _MODEL_NAME_TO_CFG:
            raise ValueError(
                f"Unknown samurai model_name={model_name!r}; "
                f"expected one of {list(_MODEL_NAME_TO_CFG)}"
            )
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.device = device

        self.predictor = self._build_predictor()

        # Internal video state (reset between sequences)
        self._inference_state = None
        self._tmp_dir: str | None = None
        self._obj_id_to_label: dict[int, int] = {}
        # Autocast/inference_mode stack held open for a whole clip so that
        # init_state, add_new_points_or_box, and propagate_in_video all run
        # under the same fp16 autocast (matches samurai/main_inference.py).
        self._ctx_stack: contextlib.ExitStack | None = None

    def _build_predictor(self):
        _activate_samurai_sam2()

        from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

        cfg_rel, ckpt_name, hf_repo = _MODEL_NAME_TO_CFG[self.model_name]

        ckpt_path = self.ckpt_path
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_name)

        return build_sam2_video_predictor(cfg_rel, ckpt_path, device=self.device)

    # ------------------------------------------------------------------
    # Stateful video-level API
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Initialise SAMURAI video state from a list of H×W×3 uint8 RGB frames."""
        self._tmp_dir = tempfile.mkdtemp(prefix="samurai_")
        for i, f in enumerate(frames):
            path = os.path.join(self._tmp_dir, f"{i:05d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        # Open a persistent inference_mode + autocast(fp16) context spanning
        # init_state → add_prompts → propagate. Matches the official samurai
        # main_inference.py, and ensures SDPA sees fp16 q/k/v instead of fp32.
        self._ctx_stack = contextlib.ExitStack()
        self._ctx_stack.enter_context(torch.inference_mode())
        if self.device.startswith("cuda"):
            self._ctx_stack.enter_context(
                torch.autocast("cuda", dtype=torch.float16)
            )

        self._inference_state = self.predictor.init_state(
            video_path=self._tmp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        self._obj_id_to_label.clear()

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """Add xyxy pixel-coordinate box prompts on `frame_idx`."""
        if len(boxes) == 0:
            return

        if obj_ids is None:
            obj_ids = list(range(1, len(boxes) + 1))

        if labels is not None:
            for oid, lbl in zip(obj_ids, labels):
                self._obj_id_to_label[int(oid)] = int(lbl)

        for obj_id, box in zip(obj_ids, boxes):
            self.predictor.add_new_points_or_box(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=np.asarray(box, dtype=np.float32),
            )

    def propagate(self) -> list[dict]:
        """
        Propagate masks through the video after prompts have been added.

        Returns a list (ordered by frame index) of per-frame dicts with keys
        boxes / obb / labels / scores / track_ids — same schema as SAM2Tracker.
        """
        frame_outputs: dict[int, dict] = {}

        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            self._inference_state
        ):
            masks = (mask_logits > 0.0).cpu()

            boxes_list, obb_list, scores_list, ids_list, labels_list = [], [], [], [], []
            for obj_id, mask in zip(obj_ids, masks):
                mask_2d = mask[0].numpy()
                box_xyxy = mask_to_aabb(mask_2d)
                if box_xyxy is None:
                    continue
                obb_8 = mask_to_obb(mask_2d)
                if obb_8 is None:
                    continue

                mask_area = float(mask_2d.sum())
                obb_area = float(cv2.contourArea(obb_8.reshape(4, 2)))
                score = mask_area / max(obb_area, 1.0)

                boxes_list.append(torch.from_numpy(box_xyxy))
                obb_list.append(torch.from_numpy(obb_8))
                scores_list.append(min(score, 1.0))
                ids_list.append(int(obj_id))
                labels_list.append(self._obj_id_to_label.get(int(obj_id), 0))

            if boxes_list:
                frame_outputs[frame_idx] = {
                    "boxes":     torch.stack(boxes_list),
                    "obb":       torch.stack(obb_list),
                    "labels":    torch.tensor(labels_list, dtype=torch.long),
                    "scores":    torch.tensor(scores_list, dtype=torch.float32),
                    "track_ids": torch.tensor(ids_list, dtype=torch.long),
                }
            else:
                frame_outputs[frame_idx] = self._empty_output()

        if not frame_outputs:
            return []
        return [frame_outputs.get(i, self._empty_output())
                for i in range(max(frame_outputs.keys()) + 1)]

    def reset_state(self):
        """Reset video state and clean up temporary files."""
        if self._inference_state is not None:
            self.predictor.reset_state(self._inference_state)
            self._inference_state = None
        self._obj_id_to_label.clear()

        if self._ctx_stack is not None:
            self._ctx_stack.close()
            self._ctx_stack = None

        if self._tmp_dir is not None:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

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
