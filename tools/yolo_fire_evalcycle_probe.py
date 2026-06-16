"""Decisive test: does interleaving an eval/validation forward poison training?

Mimics Lightning's train-epoch -> validation -> train-epoch cycle by running
the detection eval path (_inference_forward) between short bursts of training,
at peak LR. If the post-eval training step diverges to NaN, the bug is the
eval forward corrupting model state (cf. yolo_issues #10/#11 cleanup that only
_track_forward does, not _inference_forward).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import YOLODetector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_train_transform, build_eval_transform

DEVICE = "cuda"
IMG = 640


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
    model = YOLODetector(model_name="/work/ziwen/checkpoints/yolo11l.pt",
                         num_classes=3, enable_tracking=False, img_size=IMG).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.937, nesterov=True, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    train_batches = []
    for i, b in enumerate(dm.train_dataloader()):
        train_batches.append(b);
        if i >= 30: break
    val_batch = next(iter(dm.val_dataloader()))

    def train_step(images, targets, tag):
        model.train()
        images = [im.to(DEVICE) for im in images]
        targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(images, targets)
            loss = out["loss"]
        lv = loss.item()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gfin = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt); scaler.update()
        rg = sum(int(p.requires_grad) for p in params)
        wfin = all(torch.isfinite(p).all() for p in params)
        print(f"  [{tag}] loss={lv:8.3f} grad_finite={gfin} weights_finite={wfin} "
              f"#requires_grad={rg}/{len(params)} head.shape={getattr(model.model.model[-1],'shape',None)}")
        return wfin

    @torch.no_grad()
    def val_forward(tag):
        model.eval()
        images = [im.to(DEVICE) for im in val_batch[0]]
        out = model(images)
        print(f"  [{tag}] eval forward done: {len(out)} imgs, "
              f"head.shape={getattr(model.model.model[-1],'shape',None)} "
              f"#requires_grad={sum(int(p.requires_grad) for p in params)}/{len(params)}")

    print("=== phase 1: train 10 steps (no eval) ===")
    for i in range(10):
        if not train_step(*train_batches[i], f"train{i}"):
            print("  diverged in phase 1"); return

    print("=== phase 2: run eval/validation forward (like Lightning) ===")
    val_forward("val")

    print("=== phase 3: resume training — does it diverge now? ===")
    for i in range(10, 25):
        if not train_step(*train_batches[i % len(train_batches)], f"train{i}"):
            print(f"  >>> DIVERGED right after eval, at step {i}  <<<")
            return

    print("=== phase 4: another eval then more training ===")
    val_forward("val2")
    for i in range(25, 30):
        if not train_step(*train_batches[i % len(train_batches)], f"train{i}"):
            print(f"  >>> DIVERGED after 2nd eval, at step {i}  <<<")
            return

    print("\nNO divergence reproduced via eval-cycle.")


if __name__ == "__main__":
    main()
