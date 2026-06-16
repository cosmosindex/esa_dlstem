"""Decisive test: is the YOLO explosion caused by Lightning's validation pass?

Runs the REAL Lightning ObjectDetectionModule + DetectionDataModule training
loop but with validation/sanity-check DISABLED, at peak LR via warmup=1. A
GradNorm callback prints per-step pre-clip grad norm + the running max train
loss. If train loss stays bounded (~<15) with no validation, the eval/inference
forward between epochs is the culprit. If it still explodes, validation is
exonerated and the cause is elsewhere in the Lightning loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import lightning as L

from models import YOLODetector
from lightning_modules import ObjectDetectionModule, DetectionDataModule, DataModuleConfig
from transforms import build_train_transform, build_eval_transform

IMG = 640


class LossWatch(L.Callback):
    def __init__(self):
        self.max_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        v = float(loss)
        self.max_loss = max(self.max_loss, v)
        if trainer.global_step % 50 == 0:
            print(f"[watch] gstep {trainer.global_step}: loss={v:.3f} max_so_far={self.max_loss:.3e}",
                  flush=True)


def main():
    dm_cfg = DataModuleConfig(
        datasets={"FireRGBT": "/data/ESA_DLSTEM_2025/data/fire/RGBT-3M"},
        class_map={"smoke": 0, "fire": 1, "person": 2},
        batch_size=16, num_workers=0, img_size=(IMG, IMG),
    )
    dm = DetectionDataModule(cfg=dm_cfg,
                             train_transform=build_train_transform((IMG, IMG)),
                             eval_transform=build_eval_transform((IMG, IMG)))

    model = YOLODetector(model_name="/work/ziwen/checkpoints/yolo11l.pt",
                         num_classes=3, enable_tracking=False,
                         conf_thresh=0.05, iou_thresh=0.5, img_size=IMG)
    module = ObjectDetectionModule(
        model=model, has_tracking=False, lr=0.01, weight_decay=5e-4,
        lr_scheduler="cosine", warmup_epochs=1, total_epochs=3,
        optimizer="sgd", momentum=0.937,
    )

    trainer = L.Trainer(
        max_epochs=3, accelerator="auto", devices=1, precision="bf16-mixed",
        gradient_clip_val=10.0, logger=False,
        callbacks=[LossWatch()],
        num_sanity_val_steps=0,        # no sanity-check validation
        limit_val_batches=0.0,         # NO validation at all
        enable_checkpointing=False, log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=dm)
    print("DONE: no-validation run finished without crashing.")


if __name__ == "__main__":
    main()
