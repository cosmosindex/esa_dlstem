"""
DINOv3 detector: DINOv3 backbone (ViT or ConvNeXt) + DETR-style detection head.

DINOv3 is Meta AI's latest self-supervised vision foundation model.
It supports both Vision Transformer (ViT) and ConvNeXt backbone families,
pretrained on LVD-1689M (web images) or SAT-493M (satellite imagery).

References:
    Repository : https://github.com/facebookresearch/dinov3
    HuggingFace: facebook/dinov3-{arch}-pretrain-{dataset}

Loading methods (in priority order):
  1. HuggingFace AutoModel (transformers >= 4.56.0)  <- recommended
  2. torch.hub local (requires cloning the repo)

Architecture:
    DINOv3 backbone  ->  feature tokens (B, N, d_model)
    (ViT)                [CLS token excluded; N = H/p * W/p patch tokens]
    (ConvNeXt)           [spatial map flattened: N = H' * W']
                     ->  DETR-style transformer decoder
                     ->  class logits (B, Q, num_classes+1)
                       + box coords   (B, Q, 4)  cxcywh normalised

Image normalisation:
    LVD-1689M (web)      : mean=(0.485,0.456,0.406)  std=(0.229,0.224,0.225)
    SAT-493M  (satellite): mean=(0.430,0.411,0.296)  std=(0.213,0.156,0.143)

Use with ObjectDetectionModule(has_tracking=False).
"""

from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Detection head (DETR-style)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DINOv3DetectionHead(nn.Module):
    """Lightweight DETR-style detection head on top of DINOv3 feature tokens."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        num_queries: int = 100,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.input_proj  = nn.Linear(d_model, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.class_embed = nn.Linear(d_model, num_classes + 1)   # +1 for no-object
        self.bbox_embed  = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, d_model)  patch / spatial tokens from DINOv3 backbone.
        Returns:
            logits: (B, Q, num_classes+1)
            boxes:  (B, Q, 4)  cxcywh normalised [0, 1]
        """
        B = tokens.shape[0]
        memory  = self.input_proj(tokens)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs      = self.decoder(queries, memory)
        return self.class_embed(hs), self.bbox_embed(hs).sigmoid()


class DINOv3FCOSHead(nn.Module):
    """Dense anchor-free (FCOS-style) head on a single DINOv3 feature map.

    The patch tokens are reshaped to a (B, d, Hp, Wp) stride-`patch` feature
    map; a shared-design conv tower feeds parallel classification /
    box-regression / centerness branches. Far easier to train than the DETR
    head — sigmoid focal loss handles the foreground/background imbalance
    without the no-object collapse, and dense per-location regression starts
    localising from the first epoch.

    Returns raw maps:
        cls_logits : (B, num_classes, H, W)   per-location class logits
        ltrb       : (B, 4, H, W)             l,t,r,b distances in STRIDE units
        centerness : (B, 1, H, W)             per-location centerness logit
    """

    def __init__(self, d_model: int, num_classes: int,
                 num_convs: int = 4, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.input_proj = nn.Conv2d(d_model, hidden, kernel_size=1)

        cls_tower, reg_tower = [], []
        for _ in range(num_convs):
            cls_tower += [nn.Conv2d(hidden, hidden, 3, padding=1),
                          nn.GroupNorm(32, hidden), nn.ReLU(inplace=True)]
            reg_tower += [nn.Conv2d(hidden, hidden, 3, padding=1),
                          nn.GroupNorm(32, hidden), nn.ReLU(inplace=True)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_logits = nn.Conv2d(hidden, num_classes, 3, padding=1)
        self.bbox_pred  = nn.Conv2d(hidden, 4, 3, padding=1)
        self.centerness = nn.Conv2d(hidden, 1, 3, padding=1)
        self.scale = nn.Parameter(torch.ones(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # focal-loss prior on the classification bias (start with low fg prob)
        prior = 0.01
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, feat: torch.Tensor):
        x = self.input_proj(feat)
        cls_logits = self.cls_logits(self.cls_tower(x))
        reg_feat   = self.reg_tower(x)
        ltrb       = F.relu(self.scale * self.bbox_pred(reg_feat))
        centerness = self.centerness(reg_feat)
        return cls_logits, ltrb, centerness


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DINOv3Detector(nn.Module):
    """
    DINOv3 backbone + DETR-style detection head.

    Supports ViT and ConvNeXt backbone families from the DINOv3 repository.
    Class is named DINOv2Detector for backward compatibility with __init__.py;
    use DINOv3Detector as an alias.

    During training  : returns {'loss', 'loss_ce', 'loss_bbox', 'loss_giou'}.
    During inference : returns list of dicts per image:
                       {'boxes' (N,4 xyxy abs), 'labels' (N,), 'scores' (N,)}.
    """

    # HuggingFace model IDs  ->  output feature dimension.
    # NOTE: the real repo ids use the compact arch tag (vitb16, not vit-base);
    # these are the actual gated repos under facebook/ on the HF hub.
    HF_MODEL_DIMS: dict[str, int] = {
        # ViT family (LVD-1689M web pretrain)
        "facebook/dinov3-vits16-pretrain-lvd1689m":      384,
        "facebook/dinov3-vits16plus-pretrain-lvd1689m":  384,
        "facebook/dinov3-vitb16-pretrain-lvd1689m":      768,
        "facebook/dinov3-vitl16-pretrain-lvd1689m":      1024,
        "facebook/dinov3-vith16plus-pretrain-lvd1689m":  1280,
        "facebook/dinov3-vit7b16-pretrain-lvd1689m":     4096,
        # ViT family (SAT-493M satellite pretrain)
        "facebook/dinov3-vitl16-pretrain-sat493m":       1024,
        "facebook/dinov3-vit7b16-pretrain-sat493m":      4096,
        # ConvNeXt family (LVD-1689M)
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m":   768,
        "facebook/dinov3-convnext-small-pretrain-lvd1689m":  768,
        "facebook/dinov3-convnext-base-pretrain-lvd1689m":   1024,
        "facebook/dinov3-convnext-large-pretrain-lvd1689m":  1536,
    }

    # torch.hub entry-point names  ->  feature dimension
    HUB_MODEL_DIMS: dict[str, int] = {
        "dinov3_vits16":           384,
        "dinov3_vitb16":           768,
        "dinov3_vitl16":           1024,
        "dinov3_convnext_tiny":    768,
        "dinov3_convnext_small":   768,
        "dinov3_convnext_base":    1024,
        "dinov3_convnext_large":   1536,
    }

    def __init__(
        self,
        num_classes: int,
        # HuggingFace model name (preferred) or None to use torch.hub
        hf_model_name: str | None = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        # torch.hub fallback (used only when hf_model_name is None)
        hub_repo_dir: str | None = None,
        hub_model_name: str = "dinov3_vitb16",
        # Fine-tuning control
        freeze_backbone: bool = True,
        # Head selection: 'fcos' (dense anchor-free, recommended) or 'detr'
        head_type: str = "fcos",
        # DETR head config
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        # FCOS head config
        fcos_num_convs: int = 4,
        fcos_hidden: int = 256,
        fcos_center_radius: float = 1.5,
        # Effective feature stride for the FCOS head. ViT-B/16 gives a stride-16
        # token map (one cell per 16 px) — far too coarse for ~12 px objects, which
        # then get zero positive locations. Set e.g. 8 to bilinearly upsample the
        # token map 2x before the head → stride-8 grid, so tiny objects get sampled.
        # None / patch_size → no upsample (original behaviour).
        fcos_feat_stride: int | None = None,
        nms_thresh: float = 0.6,
        max_dets: int = 100,
        # Inference
        conf_thresh: float = 0.5,
    ):
        """
        Args:
            num_classes:        Number of foreground classes.
            hf_model_name:      HuggingFace model ID (transformers >= 4.56.0).
                                Set to None to use the local torch.hub path instead.
            hub_repo_dir:       Path to a local clone of the DINOv3 repo.
                                Required when hf_model_name is None.
            hub_model_name:     torch.hub entry-point, e.g. 'dinov3_vitl16'.
            freeze_backbone:    Freeze backbone weights (only train the head).
            num_queries:        Number of DETR object queries.
            num_decoder_layers: Transformer decoder depth.
            nhead:              Attention heads in the decoder.
            conf_thresh:        Minimum score to keep a prediction at inference.
        """
        super().__init__()

        self.num_classes = num_classes
        self.conf_thresh = conf_thresh

        # ------------------------------------------------------------------ #
        # Backbone                                                             #
        # ------------------------------------------------------------------ #
        if hf_model_name is not None:
            self.backbone, d_model = self._load_hf(hf_model_name)
            self._is_convnext = "convnext" in hf_model_name
            model_tag = hf_model_name
        else:
            assert hub_repo_dir is not None, (
                "Provide hub_repo_dir when hf_model_name is None."
            )
            self.backbone, d_model = self._load_hub(hub_repo_dir, hub_model_name)
            self._is_convnext = "convnext" in hub_model_name
            model_tag = hub_model_name

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        self._freeze_backbone = freeze_backbone

        # Patch size for ViT spatial-token slicing (used to skip CLS/register
        # tokens, whose count varies by model). ConvNeXt has no patch tokens.
        self._patch_size = int(getattr(getattr(self.backbone, "config", None),
                                       "patch_size", 16) or 16)

        # Image normalisation stats baked into the model so it accepts the same
        # [0,1] RGB input as FasterRCNN/YOLO (no Normalize transform needed).
        # SAT-493M pretrain uses satellite stats; everything else uses the
        # ImageNet/LVD web stats.
        if "sat493m" in model_tag:
            mean = (0.430, 0.411, 0.296); std = (0.213, 0.156, 0.143)
        else:
            mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",  torch.tensor(std).view(1, 3, 1, 1))

        # ------------------------------------------------------------------ #
        # Detection head                                                       #
        # ------------------------------------------------------------------ #
        self.head_type = head_type
        self.stride = self._patch_size            # single ViT feature level
        self._fcos_upsample = 1
        if head_type == "fcos":
            if fcos_feat_stride is not None and fcos_feat_stride != self._patch_size:
                if self._patch_size % fcos_feat_stride != 0:
                    raise ValueError(
                        f"fcos_feat_stride ({fcos_feat_stride}) must divide the "
                        f"backbone patch size ({self._patch_size})")
                self._fcos_upsample = self._patch_size // fcos_feat_stride
                self.stride = fcos_feat_stride
            self.fcos_center_radius = fcos_center_radius
            self.nms_thresh = nms_thresh
            self.max_dets   = max_dets
            self.head = DINOv3FCOSHead(
                d_model=d_model,
                num_classes=num_classes,
                num_convs=fcos_num_convs,
                hidden=fcos_hidden,
            )
        elif head_type == "detr":
            self.head = DINOv3DetectionHead(
                d_model=d_model,
                num_classes=num_classes,
                num_queries=num_queries,
                num_decoder_layers=num_decoder_layers,
                nhead=nhead,
            )
            # Hungarian loss weights
            self.cost_class = 1.0;  self.loss_ce_w   = 1.0
            self.cost_bbox  = 5.0;  self.loss_bbox_w  = 5.0
            self.cost_giou  = 2.0;  self.loss_giou_w  = 2.0
        else:
            raise ValueError(f"Unknown head_type '{head_type}' (use 'fcos' or 'detr').")

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode regardless of module.train().

        Lightning flips the whole module to train() each epoch; without this a
        frozen DINOv3 backbone would re-enable stochastic depth / dropout and
        inject noise into features that are supposed to be fixed.
        """
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Backbone loading helpers
    # ------------------------------------------------------------------

    def _load_hf(self, model_name: str) -> tuple[nn.Module, int]:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_name)
        d_model  = self.HF_MODEL_DIMS.get(model_name)
        if d_model is None:
            cfg     = backbone.config
            d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "embed_dim", None)
            assert d_model is not None, (
                f"Cannot infer feature dim for '{model_name}'. "
                "Add it to HF_MODEL_DIMS manually."
            )
        return backbone, d_model

    def _load_hub(self, repo_dir: str, model_name: str) -> tuple[nn.Module, int]:
        backbone = torch.hub.load(
            repo_dir, model_name, source="local", pretrained=True
        )
        d_model = self.HUB_MODEL_DIMS.get(model_name)
        assert d_model is not None, (
            f"Unknown hub model '{model_name}'. Add it to HUB_MODEL_DIMS."
        )
        return backbone, d_model

    # ------------------------------------------------------------------
    # Feature extraction (unified for ViT and ConvNeXt)
    # ------------------------------------------------------------------

    def _extract_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial tokens from the backbone.

        torch.hub ViT  : forward_features() -> dict['x_norm_patchtokens'] (B, N, d)
        HF ViT         : last_hidden_state (B, P+N, d) — DINOv3 prefixes CLS AND
                         register tokens, so take the LAST N=(H/p)*(W/p) tokens
                         (robust to however many prefix tokens the model uses).
        HF ConvNeXt    : last_hidden_state (B, d, H, W) -> flatten -> (B, H*W, d)

        Returns: (B, N, d_model)
        """
        H, W = batch.shape[-2], batch.shape[-1]

        # Frozen backbone: run under no_grad and keep it in eval mode so its
        # norm/dropout layers stay deterministic during fine-tuning of the head.
        ctx = torch.no_grad() if self._freeze_backbone else _nullcontext()
        with ctx:
            if hasattr(self.backbone, "forward_features"):
                # torch.hub DINOv3 ViT interface
                out = self.backbone.forward_features(batch)
                return out["x_norm_patchtokens"]       # (B, N, d)

            # HuggingFace interface
            out = self.backbone(pixel_values=batch)
            h   = out.last_hidden_state

        if h.ndim == 4:
            # ConvNeXt: (B, d, H, W) -> (B, H*W, d)
            B, d, Hf, Wf = h.shape
            return h.permute(0, 2, 3, 1).reshape(B, Hf * Wf, d)
        # ViT: keep only the (H/p)*(W/p) patch tokens at the tail, dropping the
        # leading CLS + register tokens.
        n_patches = (H // self._patch_size) * (W // self._patch_size)
        return h[:, -n_patches:, :]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ):
        """
        Args:
            images:  list of (C, H, W) float tensors in [0, 1] RGB (same format
                     as FasterRCNN/YOLO; ImageNet/sat normalisation is applied
                     internally — do NOT pre-normalise).
            targets: list of dicts {'boxes' (N,4) xyxy ABS px, 'labels' (N,)}.
                     Required during training. Boxes are converted to cxcywh
                     normalised internally.

        Returns (train): {'loss', 'loss_ce', 'loss_bbox', 'loss_giou'}
        Returns (eval):  [{'boxes' (N,4) xyxy abs, 'labels' (N,), 'scores' (N,)}]
        """
        batch = torch.stack(images)             # (B, C, H, W)
        batch = (batch - self.pixel_mean) / self.pixel_std
        H, W  = batch.shape[-2], batch.shape[-1]
        tokens = self._extract_tokens(batch)    # (B, N, d_model)

        if self.head_type == "fcos":
            Hp, Wp = H // self._patch_size, W // self._patch_size
            feat = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, Hp, Wp)
            if self._fcos_upsample > 1:
                # stride-16 token map → finer grid (e.g. stride-8) so small objects
                # get positive sample locations. self.stride is set accordingly.
                feat = F.interpolate(feat, scale_factor=self._fcos_upsample,
                                     mode="bilinear", align_corners=False)
                Hp, Wp = Hp * self._fcos_upsample, Wp * self._fcos_upsample
            cls_logits, ltrb, ctr = self.head(feat)
            if self.training:
                assert targets is not None
                return self._fcos_loss(cls_logits, ltrb, ctr, targets, Hp, Wp)
            return self._fcos_postprocess(cls_logits, ltrb, ctr, images, Hp, Wp)

        logits, pred_boxes = self.head(tokens)
        if self.training:
            assert targets is not None
            return self._compute_loss(logits, pred_boxes, targets, H, W)
        return self._postprocess(logits, pred_boxes, images)

    # ------------------------------------------------------------------
    # Loss (Hungarian bipartite matching)
    # ------------------------------------------------------------------

    def _compute_loss(self, logits, pred_boxes, targets, img_h, img_w):
        B = logits.shape[0]
        total = ce_total = bbox_total = giou_total = logits.new_zeros(1).squeeze()

        for b in range(B):
            log_b     = logits[b]
            box_b     = pred_boxes[b]
            # Targets arrive as xyxy absolute pixels (FasterRCNN format); convert
            # to cxcywh normalised [0,1] to match the head's sigmoid box output.
            gt_xyxy   = targets[b]["boxes"].to(log_b.device)
            gt_boxes  = self._xyxy_abs_to_cxcywh_norm(gt_xyxy, img_w, img_h)
            gt_labels = targets[b]["labels"].to(log_b.device)

            src_idx, tgt_idx = self._hungarian_match(log_b, box_b, gt_labels, gt_boxes)

            tgt_cls = torch.full(
                (log_b.shape[0],), self.num_classes,
                dtype=torch.long, device=log_b.device,
            )
            if len(src_idx):
                tgt_cls[src_idx] = gt_labels[tgt_idx]
            ce = F.cross_entropy(log_b, tgt_cls)

            bbox_loss = giou_loss = log_b.new_zeros(1).squeeze()
            if len(src_idx):
                pm = box_b[src_idx];  gm = gt_boxes[tgt_idx]
                bbox_loss = F.l1_loss(pm, gm)
                giou_loss = (1 - self._batch_giou_cxcywh(pm, gm)).mean()

            loss       = self.loss_ce_w * ce + self.loss_bbox_w * bbox_loss + self.loss_giou_w * giou_loss
            total      = total      + loss
            ce_total   = ce_total   + ce
            bbox_total = bbox_total + bbox_loss
            giou_total = giou_total + giou_loss

        return {
            "loss":      total      / B,
            "loss_ce":   ce_total   / B,
            "loss_bbox": bbox_total / B,
            "loss_giou": giou_total / B,
        }

    @torch.no_grad()
    def _hungarian_match(self, logits, pred_boxes, gt_labels, gt_boxes):
        if len(gt_labels) == 0:
            return [], []
        probs      = logits.softmax(-1)
        cost_class = -probs[:, gt_labels]
        cost_bbox  = torch.cdist(pred_boxes, gt_boxes, p=1)
        cost_giou  = -self._pairwise_giou_cxcywh(pred_boxes, gt_boxes)
        C          = (self.cost_class * cost_class
                      + self.cost_bbox * cost_bbox
                      + self.cost_giou * cost_giou)
        r, c = linear_sum_assignment(C.cpu().detach().numpy())
        return (torch.as_tensor(r, dtype=torch.long),
                torch.as_tensor(c, dtype=torch.long))

    # ------------------------------------------------------------------
    # Post-processing (inference)
    # ------------------------------------------------------------------

    def _postprocess(self, logits, pred_boxes, images):
        outputs = []
        for log_b, box_b, img in zip(logits, pred_boxes, images):
            scores, labels = log_b.softmax(-1)[:, :-1].max(-1)
            keep  = scores > self.conf_thresh
            H, W  = img.shape[-2], img.shape[-1]
            boxes = self._cxcywh_norm_to_xyxy_abs(box_b[keep], W, H)
            # Keep preds on the model device — the eval module mixes them with
            # CUDA targets in raw torch ops (FasterRCNN/YOLO do the same).
            outputs.append({
                "boxes":  boxes,
                "labels": labels[keep],
                "scores": scores[keep],
            })
        return outputs

    # ------------------------------------------------------------------
    # FCOS dense head: loss, target assignment, post-processing
    # ------------------------------------------------------------------

    def _fcos_locations(self, Hp: int, Wp: int, device) -> torch.Tensor:
        """Pixel centres of each feature-map cell, row-major (matches reshape)."""
        sx = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) * self.stride
        sy = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) * self.stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)   # (N, 2)

    @staticmethod
    def _fcos_decode(locations: torch.Tensor, ltrb_px: torch.Tensor) -> torch.Tensor:
        """locations (P,2) px + l,t,r,b distances px -> xyxy px."""
        x, y = locations[:, 0], locations[:, 1]
        return torch.stack([x - ltrb_px[:, 0], y - ltrb_px[:, 1],
                            x + ltrb_px[:, 2], y + ltrb_px[:, 3]], dim=-1)

    def _fcos_target_single(self, locations, gt_boxes, gt_labels):
        """Assign each location a class (-1 = background), l/t/r/b px target and
        centerness, using FCOS center-sampling + min-area ambiguity resolution."""
        N = locations.shape[0]
        cls_t = locations.new_full((N,), -1, dtype=torch.long)
        reg_t = locations.new_zeros((N, 4))
        ctr_t = locations.new_zeros((N,))
        if gt_boxes.numel() == 0:
            return cls_t, reg_t, ctr_t

        xs, ys = locations[:, 0], locations[:, 1]
        x1, y1, x2, y2 = gt_boxes.unbind(-1)
        l = xs[:, None] - x1[None];  t = ys[:, None] - y1[None]
        r = x2[None] - xs[:, None];  b = y2[None] - ys[:, None]
        ltrb = torch.stack([l, t, r, b], dim=-1)            # (N, G, 4)
        inside = ltrb.min(dim=-1).values > 0                # (N, G)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rad = self.fcos_center_radius * self.stride
        in_center = ((xs[:, None] > cx[None] - rad) & (xs[:, None] < cx[None] + rad) &
                     (ys[:, None] > cy[None] - rad) & (ys[:, None] < cy[None] + rad))
        valid = inside & in_center                          # (N, G)

        areas = ((x2 - x1) * (y2 - y1))[None].expand(N, -1).clone()
        areas[~valid] = 1e9
        min_area, gt_idx = areas.min(dim=-1)                # (N,)
        has = min_area < 1e9
        if has.any():
            hi = has.nonzero(as_tuple=True)[0]
            sel = gt_idx[hi]
            cls_t[hi] = gt_labels[sel]
            reg_t[hi] = ltrb[hi, sel]
            lr = reg_t[hi][:, [0, 2]];  tb = reg_t[hi][:, [1, 3]]
            ctr_t[hi] = torch.sqrt(
                (lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
                (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6))
            )
        return cls_t, reg_t, ctr_t

    def _fcos_loss(self, cls_logits, ltrb, centerness, targets, Hp, Wp):
        B, C = cls_logits.shape[0], self.num_classes
        dev = cls_logits.device
        locations = self._fcos_locations(Hp, Wp, dev)       # (N, 2)

        cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, C)
        reg_flat = ltrb.permute(0, 2, 3, 1).reshape(B, -1, 4)        # stride units
        ctr_flat = centerness.permute(0, 2, 3, 1).reshape(B, -1)

        cls_ts, reg_ts, ctr_ts = [], [], []
        for b in range(B):
            ct, rt, cn = self._fcos_target_single(
                locations,
                targets[b]["boxes"].to(dev).float(),
                targets[b]["labels"].to(dev),
            )
            cls_ts.append(ct); reg_ts.append(rt); ctr_ts.append(cn)
        cls_t = torch.stack(cls_ts)        # (B, N)
        reg_t = torch.stack(reg_ts)        # (B, N, 4) px
        ctr_t = torch.stack(ctr_ts)        # (B, N)

        pos = cls_t >= 0
        num_pos = pos.sum().clamp(min=1)

        cls_onehot = torch.zeros_like(cls_flat)
        if pos.any():
            pi = pos.nonzero(as_tuple=False)
            cls_onehot[pi[:, 0], pi[:, 1], cls_t[pos]] = 1.0
        loss_cls = self._sigmoid_focal_loss(cls_flat, cls_onehot).sum() / num_pos

        if pos.any():
            loc_pos = locations[None].expand(B, -1, -1)[pos]        # (P, 2)
            pred_box = self._fcos_decode(loc_pos, reg_flat[pos] * self.stride)
            tgt_box  = self._fcos_decode(loc_pos, reg_t[pos])
            ctr_pos  = ctr_t[pos]
            giou = self._batch_giou_xyxy(pred_box, tgt_box)
            loss_reg = ((1 - giou) * ctr_pos).sum() / ctr_pos.sum().clamp(min=1e-6)
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_flat[pos], ctr_pos, reduction="mean")
        else:
            loss_reg = reg_flat.sum() * 0.0
            loss_ctr = ctr_flat.sum() * 0.0

        loss = loss_cls + loss_reg + loss_ctr
        return {"loss": loss, "loss_cls": loss_cls,
                "loss_reg": loss_reg, "loss_ctr": loss_ctr}

    def _fcos_postprocess(self, cls_logits, ltrb, centerness, images, Hp, Wp):
        from torchvision.ops import batched_nms
        B, C = cls_logits.shape[0], self.num_classes
        dev = cls_logits.device
        locations = self._fcos_locations(Hp, Wp, dev)
        cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, C).sigmoid()
        ctr_flat = centerness.permute(0, 2, 3, 1).reshape(B, -1).sigmoid()
        reg_flat = ltrb.permute(0, 2, 3, 1).reshape(B, -1, 4) * self.stride

        outs = []
        for b in range(B):
            scores = cls_flat[b] * ctr_flat[b][:, None]            # (N, C)
            boxes_all = self._fcos_decode(locations, reg_flat[b])  # (N, 4)
            H, W = images[b].shape[-2], images[b].shape[-1]
            boxes_all[:, 0::2] = boxes_all[:, 0::2].clamp(0, W)
            boxes_all[:, 1::2] = boxes_all[:, 1::2].clamp(0, H)

            keep = scores > self.conf_thresh
            loc_i, cls_i = keep.nonzero(as_tuple=True)
            if loc_i.numel() == 0:
                outs.append({"boxes": boxes_all.new_zeros((0, 4)),
                             "labels": loc_i.new_zeros((0,)),
                             "scores": boxes_all.new_zeros((0,))})
                continue
            b_boxes = boxes_all[loc_i]; b_scores = scores[loc_i, cls_i]; b_labels = cls_i
            nms_keep = batched_nms(b_boxes, b_scores, b_labels, self.nms_thresh)[:self.max_dets]
            outs.append({"boxes": b_boxes[nms_keep],
                         "labels": b_labels[nms_keep],
                         "scores": b_scores[nms_keep]})
        return outs

    @staticmethod
    def _sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
        p = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** gamma)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return alpha_t * loss

    @staticmethod
    def _batch_giou_xyxy(a, b):
        x1 = torch.max(a[:, 0], b[:, 0]); y1 = torch.max(a[:, 1], b[:, 1])
        x2 = torch.min(a[:, 2], b[:, 2]); y2 = torch.min(a[:, 3], b[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        aa = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
        ab = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
        union = aa + ab - inter
        iou = inter / union.clamp(1e-6)
        ex1 = torch.min(a[:, 0], b[:, 0]); ey1 = torch.min(a[:, 1], b[:, 1])
        ex2 = torch.max(a[:, 2], b[:, 2]); ey2 = torch.max(a[:, 3], b[:, 3])
        enc = ((ex2 - ex1) * (ey2 - ey1)).clamp(1e-6)
        return iou - (enc - union) / enc

    # ------------------------------------------------------------------
    # Box geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)

    @staticmethod
    def _xyxy_abs_to_cxcywh_norm(b, W, H):
        if b.numel() == 0:
            return b.reshape(0, 4)
        x1, y1, x2, y2 = b.unbind(-1)
        return torch.stack([
            (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
            (x2 - x1) / W,     (y2 - y1) / H,
        ], -1)

    @staticmethod
    def _cxcywh_norm_to_xyxy_abs(b, W, H):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], -1)

    def _pairwise_giou_cxcywh(self, pred, gt):
        pa = self._cxcywh_to_xyxy(pred)
        ga = self._cxcywh_to_xyxy(gt)
        x1 = torch.max(pa[:,None,0], ga[None,:,0]);  y1 = torch.max(pa[:,None,1], ga[None,:,1])
        x2 = torch.min(pa[:,None,2], ga[None,:,2]);  y2 = torch.min(pa[:,None,3], ga[None,:,3])
        inter  = (x2-x1).clamp(0) * (y2-y1).clamp(0)
        area_p = (pa[:,2]-pa[:,0]) * (pa[:,3]-pa[:,1])
        area_g = (ga[:,2]-ga[:,0]) * (ga[:,3]-ga[:,1])
        union  = area_p[:,None] + area_g[None,:] - inter
        iou    = inter / union.clamp(1e-6)
        ex1=torch.min(pa[:,None,0],ga[None,:,0]); ey1=torch.min(pa[:,None,1],ga[None,:,1])
        ex2=torch.max(pa[:,None,2],ga[None,:,2]); ey2=torch.max(pa[:,None,3],ga[None,:,3])
        enc = ((ex2-ex1)*(ey2-ey1)).clamp(1e-6)
        return iou - (enc - union) / enc

    def _batch_giou_cxcywh(self, pred, gt):
        pa = self._cxcywh_to_xyxy(pred); ga = self._cxcywh_to_xyxy(gt)
        x1=torch.max(pa[:,0],ga[:,0]); y1=torch.max(pa[:,1],ga[:,1])
        x2=torch.min(pa[:,2],ga[:,2]); y2=torch.min(pa[:,3],ga[:,3])
        inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
        ap=(pa[:,2]-pa[:,0])*(pa[:,3]-pa[:,1]); ag=(ga[:,2]-ga[:,0])*(ga[:,3]-ga[:,1])
        union=ap+ag-inter; iou=inter/union.clamp(1e-6)
        ex1=torch.min(pa[:,0],ga[:,0]); ey1=torch.min(pa[:,1],ga[:,1])
        ex2=torch.max(pa[:,2],ga[:,2]); ey2=torch.max(pa[:,3],ga[:,3])
        enc=((ex2-ex1)*(ey2-ey1)).clamp(1e-6)
        return iou-(enc-union)/enc
