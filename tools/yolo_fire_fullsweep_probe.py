"""Full-dataset sweep at peak LR to catch the rare bad batch behind the NaN.

Iterates the ENTIRE FireRGBT train loader (with real augmentation) for a few
epochs at peak LR using the exact config that diverged in the real run
(SGD lr=0.01, fp16, clip=10). Reports the first batch whose loss/grad is
non-finite and dumps that batch's target stats so we can see what triggers it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import YOLODetector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_train_transform

DEVICE = "cuda"
IMG = 640
EPOCHS = 3


def main():
    dm_cfg = DataModuleConfig(
        datasets={"FireRGBT": "/data/ESA_DLSTEM_2025/data/fire/RGBT-3M"},
        class_map={"smoke": 0, "fire": 1, "person": 2},
        batch_size=16, num_workers=0, img_size=(IMG, IMG),
    )
    dm = DetectionDataModule(cfg=dm_cfg, train_transform=build_train_transform((IMG, IMG)))
    dm.setup("fit")

    torch.manual_seed(0)
    model = YOLODetector(model_name="/work/anon/checkpoints/yolo11l.pt",
                         num_classes=3, enable_tracking=False, img_size=IMG).to(DEVICE)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=0.01, momentum=0.937, nesterov=True)
    scaler = torch.cuda.amp.GradScaler()

    bad_batches = 0
    global_step = 0
    for epoch in range(EPOCHS):
        loader = dm.train_dataloader()
        running_max = 0.0
        for bi, (images, targets) in enumerate(loader):
            images = [im.to(DEVICE) for im in images]
            targets = [{k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()}
                       for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(images, targets)
                loss = out["loss"]
            lv = loss.item()
            running_max = max(running_max, lv if lv == lv else 1e9)
            if not torch.isfinite(loss):
                nb = [len(t["labels"]) for t in targets]
                print(f"  !! NON-FINITE LOSS at epoch {epoch} batch {bi} (gstep {global_step}): "
                      f"loss={lv} box={out['loss_box'].item()} cls={out['loss_cls'].item()} "
                      f"dfl={out['loss_dfl'].item()} | #boxes/img={nb}")
                bad_batches += 1
                global_step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # check grad finiteness BEFORE clip
            gfin = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            if not gfin:
                print(f"  !! NON-FINITE GRAD at epoch {epoch} batch {bi} (gstep {global_step}), "
                      f"loss was {lv:.3f}")
                bad_batches += 1
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            # check weights still finite
            if global_step % 50 == 0:
                wfin = all(torch.isfinite(p).all() for p in params)
                print(f"  epoch {epoch} batch {bi:3d} gstep {global_step:4d}: loss={lv:7.3f} "
                      f"weights_finite={wfin}")
                if not wfin:
                    print("  !!! WEIGHTS POISONED -> permanent NaN. stopping.")
                    return
            global_step += 1
        print(f"== epoch {epoch} done: max_loss={running_max:.2f} bad_batches_total={bad_batches}")

    print(f"\nFINISHED {EPOCHS} epochs. total non-finite events={bad_batches}. weights finite="
          f"{all(torch.isfinite(p).all() for p in params)}")


if __name__ == "__main__":
    main()
