"""Diagnose the "train loss -> 0, model stops learning" collapse on YOLO fine-tune.

Manual train+eval loop over the real FireRGBT data with the real YOLODetector,
running enough epochs (with warmup->cosine, peak LR) to reach the collapse.
Per epoch it logs mean box/cls/dfl loss + #empty-target batches, then evaluates
on a val slice counting total PREDICTED boxes (at conf 0.05 and 0.25) vs total
GT boxes and a crude TP@0.5 recall. If loss heads to 0 while predicted-box
count collapses to ~0, the model is degenerating to "predict nothing".
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import YOLODetector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_train_transform, build_eval_transform

DEVICE = "cuda"
IMG = 640
import os
EPOCHS = int(os.environ.get("PROBE_EPOCHS", 15))
WARMUP = int(os.environ.get("PROBE_WARMUP", 2))
PEAK_LR = float(os.environ.get("PROBE_LR", 0.01))
MOMENTUM = float(os.environ.get("PROBE_MOMENTUM", 0.937))
VAL_BATCHES = 40


def box_iou(a, b):
    # a:(N,4) b:(M,4) xyxy
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main():
    dm_cfg = DataModuleConfig(
        datasets={"FireRGBT": "/data/ESA_DLSTEM_2025/data/fire/RGBT-3M"},
        class_map={"smoke": 0, "fire": 1, "person": 2},
        batch_size=16, num_workers=0, img_size=(IMG, IMG),
    )
    dm = DetectionDataModule(cfg=dm_cfg,
                             train_transform=build_train_transform((IMG, IMG)),
                             eval_transform=build_eval_transform((IMG, IMG)))
    dm.setup("fit")

    torch.manual_seed(0)
    model = YOLODetector(model_name="/work/anon/checkpoints/yolo11l.pt",
                         num_classes=3, enable_tracking=False,
                         conf_thresh=0.05, iou_thresh=0.5, img_size=IMG).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=PEAK_LR, momentum=MOMENTUM, nesterov=True)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, total_iters=WARMUP),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP)],
        milestones=[WARMUP])

    @torch.no_grad()
    def evaluate():
        model.eval()
        n_pred_05 = n_pred_25 = n_gt = tp = 0
        for bi, (images, targets) in enumerate(dm.val_dataloader()):
            if bi >= VAL_BATCHES:
                break
            images = [im.to(DEVICE) for im in images]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(images)
            for p, t in zip(preds, targets):
                pb, ps = p["boxes"].float().cpu(), p["scores"].float().cpu()
                gb = t["boxes"].float()
                n_gt += len(gb)
                n_pred_05 += int((ps >= 0.05).sum())
                n_pred_25 += int((ps >= 0.25).sum())
                keep = ps >= 0.25
                if keep.any() and len(gb):
                    iou = box_iou(pb[keep], gb)
                    tp += int((iou.max(dim=1).values >= 0.5).sum())
        rec = tp / max(n_gt, 1)
        return n_pred_05, n_pred_25, n_gt, rec

    print(f"{'ep':>2} {'lr':>8} {'box':>7} {'cls':>7} {'dfl':>7} {'total':>7} "
          f"{'empty':>5} | {'pred@.05':>8} {'pred@.25':>8} {'gt':>6} {'recall':>6}")
    for epoch in range(EPOCHS):
        model.train()
        sums = torch.zeros(3)
        n = 0
        empty = 0
        max_step_loss = 0.0
        lr = opt.param_groups[0]["lr"]
        for images, targets in dm.train_dataloader():
            images = [im.to(DEVICE) for im in images]
            targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()}
                       for t in targets]
            empty += sum(1 for t in targets if len(t["labels"]) == 0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(images, targets)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            max_step_loss = max(max_step_loss, loss.item())
            sums += torch.tensor([out["loss_box"].item(), out["loss_cls"].item(), out["loss_dfl"].item()])
            n += 1
        sched.step()
        m = sums / max(n, 1)
        p05, p25, gt, rec = evaluate()
        print(f"{epoch:>2} {lr:>8.5f} {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} "
              f"{m.sum():>7.3f} {empty:>5} | {p05:>8} {p25:>8} {gt:>6} {rec:>6.3f} "
              f"| max_step_loss={max_step_loss:.3e}",
              flush=True)


if __name__ == "__main__":
    main()
