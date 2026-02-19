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

    # HuggingFace model IDs  ->  output feature dimension
    HF_MODEL_DIMS: dict[str, int] = {
        # ViT family
        "facebook/dinov3-vit-small-pretrain-lvd1689m":  384,
        "facebook/dinov3-vit-base-pretrain-lvd1689m":   768,
        "facebook/dinov3-vit-large-pretrain-lvd1689m":  1024,
        "facebook/dinov3-vit-large-pretrain-sat493m":   1024,
        # ConvNeXt family
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
        hf_model_name: str | None = "facebook/dinov3-vit-base-pretrain-lvd1689m",
        # torch.hub fallback (used only when hf_model_name is None)
        hub_repo_dir: str | None = None,
        hub_model_name: str = "dinov3_vitb16",
        # Fine-tuning control
        freeze_backbone: bool = True,
        # Detection head config
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        nhead: int = 8,
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
        else:
            assert hub_repo_dir is not None, (
                "Provide hub_repo_dir when hf_model_name is None."
            )
            self.backbone, d_model = self._load_hub(hub_repo_dir, hub_model_name)
            self._is_convnext = "convnext" in hub_model_name

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Detection head                                                       #
        # ------------------------------------------------------------------ #
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
        HF ViT         : last_hidden_state (B, 1+N, d) — drop CLS -> (B, N, d)
        HF ConvNeXt    : last_hidden_state (B, d, H, W) -> flatten -> (B, H*W, d)

        Returns: (B, N, d_model)
        """
        if hasattr(self.backbone, "forward_features"):
            # torch.hub DINOv3 ViT interface
            out = self.backbone.forward_features(batch)
            return out["x_norm_patchtokens"]       # (B, N, d)

        # HuggingFace interface
        out = self.backbone(pixel_values=batch)
        h   = out.last_hidden_state

        if h.ndim == 4:
            # ConvNeXt: (B, d, H, W) -> (B, H*W, d)
            B, d, H, W = h.shape
            return h.permute(0, 2, 3, 1).reshape(B, H * W, d)
        else:
            # ViT: (B, 1+N, d) — skip CLS token at position 0
            return h[:, 1:, :]

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
            images:  list of (C, H, W) float tensors, normalised with the
                     appropriate mean/std for the pretrained dataset.
            targets: list of dicts {'boxes' (N,4) cxcywh norm, 'labels' (N,)}.
                     Required during training.

        Returns (train): {'loss', 'loss_ce', 'loss_bbox', 'loss_giou'}
        Returns (eval):  [{'boxes' (N,4) xyxy abs, 'labels' (N,), 'scores' (N,)}]
        """
        batch  = torch.stack(images)            # (B, C, H, W)
        tokens = self._extract_tokens(batch)    # (B, N, d_model)
        logits, pred_boxes = self.head(tokens)

        if self.training:
            assert targets is not None
            return self._compute_loss(logits, pred_boxes, targets)
        return self._postprocess(logits, pred_boxes, images)

    # ------------------------------------------------------------------
    # Loss (Hungarian bipartite matching)
    # ------------------------------------------------------------------

    def _compute_loss(self, logits, pred_boxes, targets):
        B = logits.shape[0]
        total = ce_total = bbox_total = giou_total = logits.new_zeros(1).squeeze()

        for b in range(B):
            log_b     = logits[b]
            box_b     = pred_boxes[b]
            gt_boxes  = targets[b]["boxes"].to(log_b.device)
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
            outputs.append({
                "boxes":  boxes.cpu(),
                "labels": labels[keep].cpu(),
                "scores": scores[keep].cpu(),
            })
        return outputs

    # ------------------------------------------------------------------
    # Box geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)

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
