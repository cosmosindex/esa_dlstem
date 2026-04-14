"""
LoRAT v1 tracker wrapper.

LoRAT (ECCV 2024) is a DINOv2-backbone one-stream SOT tracker that uses LoRA
adapters on top of a frozen ViT backbone. Inference builds a template crop on
the init frame, then for every subsequent frame builds a search-region crop
around the last prediction and runs a single joint forward pass through the
ViT to get a score map + per-cell bounding box.

Source: `/home/ziwen/code/esa_dlstem/LoRAT` (ECCV 2024 official repo).

Integration quirks
------------------
* LoRAT ships as the `trackit` framework, which tightly couples its per-frame
  tracker to its own batched data pipeline. There is no drop-in
  `tracker.initialize(frame, bbox) / tracker.track(frame)` API. We bypass
  `trackit` by building the bare `LoRAT_DINOv2` nn.Module and driving it
  directly, reusing trackit's siamfc crop helpers and config loader.
* Configs use custom yaml tags (`!include` / `!combine` / `!concat`) — must
  use `trackit.core.runtime.utils.custom_yaml_loader.load_yaml`, not
  `yaml.safe_load`.
* The model's `forward(z, x, z_feat_mask)` takes already-cropped & normalised
  template / search tensors. Head output is `{score_map: (B,H,W),
  boxes: (B,H,W,4)}` where boxes are [0,1]-normalised in search-crop space.
* Per-variant hparams (template_area_factor=2.0, search_area_factor=4.0) are
  taken from `LoRAT/config/LoRAT/run.yaml` eval section. We hardcode them
  since they're stable across all LoRAT v1 variants (B/L/g, 224/378).
* `optimize_for_inference=True` on the builder asks it to skip instantiating
  LoRA adapters and instead attach a state_dict hook that merges LoRA deltas
  into the base layers at load time — exactly what we want for pretrained
  weights.

Usage mirrors OSTrackTracker:
    tracker = LoRATTracker(
        lorat_root="/home/ziwen/code/esa_dlstem/LoRAT",
        method_name="LoRAT",
        config_name="g-378",
        ckpt_path="/work/ziwen/checkpoints/lorat/lorat_g378.bin",
    )
    tracker.init_video(frames)
    tracker.add_prompts(frame_idx=0, boxes=..., labels=..., obj_ids=...)
    outputs = tracker.propagate()
    tracker.reset_state()
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


_DEFAULT_LORAT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "LoRAT")
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# From LoRAT/config/LoRAT/run.yaml — stable across variants.
_TEMPLATE_AREA_FACTOR = 2.0
_SEARCH_AREA_FACTOR = 4.0


def _activate_lorat_root(lorat_root: str):
    """Prepend LoRAT root to sys.path so `import trackit.*` resolves."""
    if lorat_root not in sys.path:
        sys.path.insert(0, lorat_root)


def _load_resolved_config(lorat_root: str, method_name: str, config_name: str) -> dict:
    """Load a LoRAT yaml config, resolving !include / !combine / !concat / !const.

    `run.yaml` uses `!const` tags that look up values in `LoRAT/consts.yaml`
    (auto-copied from `consts.yaml.template` on first access). We force-init
    the global constants module, then feed its mapping into the yaml loader.
    """
    from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
    from trackit.core.runtime import global_constant as _gc

    if _gc._global_constants is None:
        _gc._initialize_global_constants()

    config_path = os.path.join(
        lorat_root, "config", method_name, config_name, "config.yaml"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"LoRAT config not found: {config_path}")
    return load_yaml(config_path, _gc._global_constants)


class LoRATTracker(nn.Module):
    """
    LoRAT v1 single-object tracker wrapped as an nn.Module, mirroring
    OSTrackTracker's API.

    Args:
        lorat_root:   Absolute path to the LoRAT repo (the one that contains
                      `trackit/`). Defaults to `../LoRAT` relative to this file.
        method_name:  LoRAT config group (usually "LoRAT").
        config_name:  Variant directory name (e.g. "B-224", "L-378", "g-378").
        ckpt_path:    Absolute path to the .bin/.safetensors weight file.
        device:       Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        ckpt_path: str,
        config_name: str = "g-378",
        method_name: str = "LoRAT",
        lorat_root: str | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        if lorat_root is None:
            lorat_root = _DEFAULT_LORAT_ROOT
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"LoRAT checkpoint not found: {ckpt_path}")

        self.lorat_root = lorat_root
        self.method_name = method_name
        self.config_name = config_name
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)

        _activate_lorat_root(self.lorat_root)
        self.config = _load_resolved_config(lorat_root, method_name, config_name)

        common = self.config["common"]
        # LoRAT stores sizes as (W, H).
        self.template_size = tuple(common["template_size"])
        self.search_size = tuple(common["search_region_size"])
        self.template_feat_size = tuple(common["template_feat_size"])
        self.search_feat_size = tuple(common["search_region_feat_size"])
        self.interpolation_mode = common.get("interpolation_mode", "bilinear")
        self.interpolation_align_corners = common.get(
            "interpolation_align_corners", False
        )
        self.normalization = common.get("normalization", "imagenet")
        if self.normalization != "imagenet":
            raise NotImplementedError(
                f"Only imagenet normalization is supported, got {self.normalization}"
            )

        self._template_stride = (
            self.template_size[0] / self.template_feat_size[0],
            self.template_size[1] / self.template_feat_size[1],
        )

        self.predictor = self._build_predictor()

        # Per-sequence state.
        self._frames: list[np.ndarray] | None = None
        self._prompt_frame: int | None = None
        self._prompt_label: int = 0
        self._z: torch.Tensor | None = None            # (1, 3, Hz, Wz) normalised
        self._z_feat_mask: torch.Tensor | None = None  # (1, Hz_f*Wz_f) long
        self._image_mean: torch.Tensor | None = None   # (3,)
        self._last_bbox_xyxy: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Model build
    # ------------------------------------------------------------------

    def _build_predictor(self) -> nn.Module:
        import safetensors.torch
        from trackit.models import ModelImplementationSuggestions
        from trackit.models.methods.LoRAT.builder import build_LoRAT_model

        # load_pretrained=True so build_backbone downloads the DINOv2 ViT backbone
        # from HF / torch hub (cached). The .bin checkpoint only contains the LoRA
        # adapters + head — the frozen backbone is never saved in LoRAT checkpoints.
        # optimize_for_inference=True: LoRA deltas are merged into base weights via
        # state_dict hooks at load time, so no LoRA layers exist at runtime.
        suggestions = ModelImplementationSuggestions(
            device=self.device,
            dtype=torch.float32,
            load_pretrained=True,
            optimize_for_inference=True,
        )
        model = build_LoRAT_model(self.config, suggestions)

        state_dict = safetensors.torch.load_file(self.ckpt_path, device=str(self.device))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in LoRAT checkpoint: {unexpected[:5]}..."
            )
        # "Missing" keys are expected: LoRAT checkpoints only store LoRA adapters +
        # head weights. The frozen DINOv2 backbone is supplied separately by
        # build_backbone(load_pretrained=True), so ~400 backbone keys are absent
        # from the .bin but already present in the live model.

        model.to(self.device).eval()
        return model

    # ------------------------------------------------------------------
    # Stateful video-level API (mirrors OSTrackTracker)
    # ------------------------------------------------------------------

    def init_video(self, frames: list[np.ndarray]):
        """Cache raw H×W×3 uint8 RGB frames for this sequence."""
        self._frames = [np.ascontiguousarray(f) for f in frames]
        self.reset_state(keep_frames=True)

    def add_prompts(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        labels: np.ndarray | None = None,
        obj_ids: list[int] | None = None,
    ):
        """Initialise the template from a single xyxy bbox (N=1 only)."""
        if len(boxes) == 0:
            return
        if self._frames is None:
            raise RuntimeError("init_video() must be called before add_prompts().")
        if frame_idx < 0 or frame_idx >= len(self._frames):
            raise IndexError(f"prompt frame_idx {frame_idx} out of range")

        if labels is not None and len(labels) > 0:
            self._prompt_label = int(labels[0])

        init_bbox_xyxy = np.asarray(boxes[0], dtype=np.float64)
        self._prompt_frame = frame_idx
        self._last_bbox_xyxy = init_bbox_xyxy.copy()

        frame = self._frames[frame_idx]
        z, z_feat_mask, image_mean = self._build_template(frame, init_bbox_xyxy)
        self._z = z
        self._z_feat_mask = z_feat_mask
        self._image_mean = image_mean

    def propagate(self) -> list[dict]:
        """Propagate the template forward through the video."""
        if self._frames is None or self._prompt_frame is None:
            return []

        n = len(self._frames)
        outputs: list[dict] = [self._empty_output() for _ in range(n)]

        p = self._prompt_frame
        outputs[p] = self._bbox_to_output(self._last_bbox_xyxy, score=1.0)

        with torch.inference_mode():
            for i in range(p + 1, n):
                pred_xyxy, score = self._track_frame(self._frames[i])
                outputs[i] = self._bbox_to_output(pred_xyxy, score=score)

        return outputs

    def reset_state(self, keep_frames: bool = False):
        """Clear per-sequence caches. Model weights stay resident."""
        if not keep_frames:
            self._frames = None
        self._prompt_frame = None
        self._prompt_label = 0
        self._z = None
        self._z_feat_mask = None
        self._image_mean = None
        self._last_bbox_xyxy = None

    # ------------------------------------------------------------------
    # Internal: cropping + forward
    # ------------------------------------------------------------------

    def _build_template(self, frame: np.ndarray, init_bbox_xyxy: np.ndarray):
        """Crop the template at init_bbox with template_area_factor.

        Returns (z_normalised, z_feat_mask, image_mean).
        """
        from trackit.core.utils.siamfc_cropping import (
            apply_siamfc_cropping,
            get_siamfc_cropping_params,
        )
        from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import (
            get_foreground_bounding_box,
        )

        image_t = self._frame_to_tensor(frame)  # (3, H, W) float32 on device

        template_output_size = np.array(self.template_size, dtype=np.int64)
        cropping_params = get_siamfc_cropping_params(
            init_bbox_xyxy, _TEMPLATE_AREA_FACTOR, template_output_size
        )

        cropped, image_mean, adjusted_params = apply_siamfc_cropping(
            image_t,
            template_output_size,
            cropping_params,
            self.interpolation_mode,
            self.interpolation_align_corners,
        )
        z = self._normalise(cropped.unsqueeze(0))  # (1, 3, Hz, Wz)

        # Feature-space foreground mask (values 0 or 1).
        feat_bbox = get_foreground_bounding_box(
            init_bbox_xyxy, adjusted_params, self._template_stride
        )
        feat_W, feat_H = self.template_feat_size
        feat_bbox = np.clip(
            feat_bbox,
            a_min=[0, 0, 0, 0],
            a_max=[feat_W, feat_H, feat_W, feat_H],
        ).astype(np.int64)
        if not (feat_bbox[2] > feat_bbox[0] and feat_bbox[3] > feat_bbox[1]):
            # Degenerate init — make the whole template foreground so the model
            # at least has non-zero tokens. This is a rare safety net.
            mask = torch.ones((feat_H, feat_W), dtype=torch.long)
        else:
            mask = torch.zeros((feat_H, feat_W), dtype=torch.long)
            x1, y1, x2, y2 = feat_bbox.tolist()
            mask[y1:y2, x1:x2] = 1
        z_feat_mask = mask.reshape(1, feat_H * feat_W).to(self.device)

        return z, z_feat_mask, image_mean

    def _track_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one search-region forward pass, return (pred_xyxy, score)."""
        from trackit.core.utils.siamfc_cropping import (
            apply_siamfc_cropping,
            apply_siamfc_cropping_to_boxes,
            get_siamfc_cropping_params,
            reverse_siamfc_cropping_params,
        )

        image_t = self._frame_to_tensor(frame)
        H, W = image_t.shape[-2:]

        search_output_size = np.array(self.search_size, dtype=np.int64)
        cropping_params = get_siamfc_cropping_params(
            self._last_bbox_xyxy, _SEARCH_AREA_FACTOR, search_output_size
        )

        cropped, _, adjusted_params = apply_siamfc_cropping(
            image_t,
            search_output_size,
            cropping_params,
            self.interpolation_mode,
            self.interpolation_align_corners,
            image_mean=self._image_mean,
        )
        x = self._normalise(cropped.unsqueeze(0))  # (1, 3, Hx, Wx)

        head = self.predictor(self._z, x, self._z_feat_mask)
        score_map = head["score_map"].float().sigmoid()  # (1, H, W)
        boxes = head["boxes"].float()                    # (1, H, W, 4) in [0,1]

        B, Hf, Wf = score_map.shape
        flat_scores = score_map.view(B, Hf * Wf)
        score, best_idx = flat_scores.max(dim=1, keepdim=True)
        flat_boxes = boxes.view(B, Hf * Wf, 4)
        best_box = torch.gather(
            flat_boxes, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)
        ).squeeze(1)  # (1, 4)

        # Convert normalised [0,1] → search-crop pixel coords.
        sw, sh = self.search_size
        scale = torch.tensor([sw, sh, sw, sh], device=best_box.device)
        pred_search = (best_box * scale).squeeze(0).detach().cpu().numpy().astype(np.float64)

        # Reverse cropping back to full-image coords.
        pred_image = apply_siamfc_cropping_to_boxes(
            pred_search, reverse_siamfc_cropping_params(adjusted_params)
        )
        pred_image = self._clip_bbox(pred_image, W, H)

        if self._bbox_is_valid(pred_image):
            self._last_bbox_xyxy = pred_image.copy()
        # else: keep the previous bbox — next frame searches the same area.

        return pred_image, float(score.item())

    # ------------------------------------------------------------------
    # Internal: small helpers
    # ------------------------------------------------------------------

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """H×W×3 uint8 RGB → (3, H, W) float32 in [0, 255] on device."""
        t = torch.from_numpy(frame).to(self.device)
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        return t.to(torch.float32)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) uint8-range float → imagenet-normalised."""
        x = x / 255.0
        return TF.normalize(x, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))

    @staticmethod
    def _clip_bbox(bbox_xyxy: np.ndarray, W: int, H: int) -> np.ndarray:
        out = bbox_xyxy.astype(np.float64).copy()
        out[0] = np.clip(out[0], 0, W - 1)
        out[1] = np.clip(out[1], 0, H - 1)
        out[2] = np.clip(out[2], 0, W)
        out[3] = np.clip(out[3], 0, H)
        return out

    @staticmethod
    def _bbox_is_valid(bbox_xyxy: np.ndarray) -> bool:
        return bool(bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1])

    def _bbox_to_output(self, xyxy: np.ndarray, score: float = 1.0) -> dict:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        box_xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        obb = torch.tensor(
            [[x1, y1, x2, y1, x2, y2, x1, y2]], dtype=torch.float32
        )
        return {
            "boxes":     box_xyxy,
            "obb":       obb,
            "labels":    torch.tensor([self._prompt_label], dtype=torch.long),
            "scores":    torch.tensor([score], dtype=torch.float32),
            "track_ids": torch.tensor([1], dtype=torch.long),
        }

    @staticmethod
    def _empty_output() -> dict:
        return {
            "boxes":     torch.zeros((0, 4), dtype=torch.float32),
            "obb":       torch.zeros((0, 8), dtype=torch.float32),
            "labels":    torch.zeros(0, dtype=torch.long),
            "scores":    torch.zeros(0, dtype=torch.float32),
            "track_ids": torch.zeros(0, dtype=torch.long),
        }
