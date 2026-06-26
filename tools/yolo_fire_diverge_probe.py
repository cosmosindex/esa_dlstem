"""Controlled stress test for YOLO11 FireRGBT NaN divergence (yolo_issues #15).

Trains the real YOLODetector on real FireRGBT batches at PEAK LR (no warmup),
logging per-step total loss + global grad norm, and reporting the first step
whose loss/grad becomes non-finite. Runs several configs back-to-back so we can
see which knob actually prevents the NaN without burning a 26-min full run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from models import YOLODetector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_train_transform

DEVICE = "cuda"
N_STEPS = 200
BATCH = 16
IMG = 640


def get_batches(n_batches: int):
    dm_cfg = DataModuleConfig(
        datasets={"FireRGBT": "/data/ESA_DLSTEM_2025/data/fire/RGBT-3M"},
        class_map={"smoke": 0, "fire": 1, "person": 2},
        batch_size=BATCH,
        num_workers=0,
        img_size=(IMG, IMG),
    )
    dm = DetectionDataModule(cfg=dm_cfg, train_transform=build_train_transform((IMG, IMG)))
    dm.setup("fit")
    loader = dm.train_dataloader()
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i + 1 >= n_batches:
            break
    return batches


def to_device(batch):
    images, targets = batch
    images = [im.to(DEVICE) for im in images]
    targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    return images, targets


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                return float("nan")
            total += g.float().norm() ** 2
    return float(total ** 0.5)


def run_config(name, batches, *, opt_kind, lr, amp_dtype, grad_clip, nan_skip):
    print(f"\n{'='*70}\nCONFIG: {name}\n  opt={opt_kind} lr={lr} amp={amp_dtype} clip={grad_clip} nan_skip={nan_skip}\n{'='*70}")
    torch.manual_seed(0)
    model = YOLODetector(model_name="/work/anon/checkpoints/yolo11l.pt",
                         num_classes=3, enable_tracking=False, img_size=IMG).to(DEVICE)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_kind == "sgd":
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=lr, momentum=0.937, nesterov=True)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)

    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    first_nan = None
    skipped = 0
    for step in range(N_STEPS):
        images, targets = to_device(batches[step % len(batches)])
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(images, targets)
            loss = out["loss"]
        loss_val = loss.item()
        if not torch.isfinite(loss):
            if first_nan is None:
                first_nan = step
            if nan_skip:
                skipped += 1
                continue
            else:
                print(f"  step {step:3d}: loss=NaN -> DIVERGED (no skip)")
                break
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip).item()
        else:
            scaler.unscale_(opt)
            gn = grad_norm(params)
        # If grad is non-finite, skipping protects weights
        if nan_skip and not (gn == gn):  # nan check
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if first_nan is None:
                first_nan = step
            continue
        scaler.step(opt)
        scaler.update()
        if step < 12 or step % 20 == 0:
            print(f"  step {step:3d}: loss={loss_val:8.3f}  grad_norm={gn:10.3f}")

    if first_nan is None:
        print(f"  RESULT: STABLE through {N_STEPS} steps ✅ (skipped {skipped})")
    else:
        print(f"  RESULT: first non-finite at step {first_nan} (skipped {skipped} total)")
    del model, opt
    torch.cuda.empty_cache()
    return first_nan


def main():
    print("loading batches...")
    batches = get_batches(20)
    print(f"got {len(batches)} batches")

    # 1. Reproduce: SGD@0.01 fp16, loose clip (the run that just failed)
    run_config("repro-sgd-fp16-clip10", batches,
               opt_kind="sgd", lr=0.01, amp_dtype=torch.float16, grad_clip=10.0, nan_skip=False)

    # 2. bf16 instead of fp16 (removes overflow as a cause)
    run_config("sgd-bf16-clip10", batches,
               opt_kind="sgd", lr=0.01, amp_dtype=torch.bfloat16, grad_clip=10.0, nan_skip=False)

    # 3. lower LR + tight clip, fp16
    run_config("sgd-lr0.0025-fp16-clip1", batches,
               opt_kind="sgd", lr=0.0025, amp_dtype=torch.float16, grad_clip=1.0, nan_skip=False)

    # 4. robust combo: bf16 + tight clip + NaN-skip, full peak LR
    run_config("ROBUST-sgd-bf16-clip1-skip", batches,
               opt_kind="sgd", lr=0.01, amp_dtype=torch.bfloat16, grad_clip=1.0, nan_skip=True)


if __name__ == "__main__":
    main()
