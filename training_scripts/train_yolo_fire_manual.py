"""
YOLOv11 fine-tuning on RGBT-3M (FireRGBT) — MANUAL training loop.

WHY a manual loop instead of the Lightning ObjectDetectionModule: the YOLO
fine-tune explodes to NaN / loss=0 under Lightning's training loop. Root cause
isolated (yolo_issues #15): the warmup learning-rate schedule does NOT take
effect under Lightning (SequentialLR + `interval="epoch"` interaction) — epoch 0
runs at full LR, the weights move fast, and the YOLO DFL/CIoU loss explodes at
peak LR; gradient clipping cannot stop the runaway (it clips magnitude, not a
bad-direction step). A plain training loop applies the same warmup correctly and
trains cleanly (verified: recall 0.11 -> 0.70 over a few epochs, no explosion).

This script reproduces the Lightning module's data + eval semantics exactly:
  - same FireRGBT splits, same 640 input, same transforms
  - same metric: torchmetrics MeanAveragePrecision(iou=0.5, class_metrics=True)
  - bf16 autocast (bf16, NOT fp16 — fp16 overflows at the explosion; see #15)
so the numbers are comparable to the FasterRCNN run (test/mAP 0.903).

Usage:
    CUDA_VISIBLE_DEVICES=1 python training_scripts/train_yolo_fire_manual.py \\
        --config configs/Detection/yolo11_fire.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision

from models import YOLODetector
from lightning_modules import DetectionDataModule, DataModuleConfig
from transforms import build_train_transform, build_eval_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_device_targets(targets):
    return [
        {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()}
        for t in targets
    ]


def build_optimizer(model, lr, momentum, weight_decay):
    """Ultralytics-style SGD: no weight decay on biases / norm (1-D) params."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, momentum=momentum, nesterov=True,
    )


def build_scheduler(opt, warmup_epochs, total_epochs):
    """LinearLR warmup -> CosineAnnealing. Stepped once per epoch (this works
    correctly in a plain loop, unlike under Lightning — see module note)."""
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_epochs - warmup_epochs)),
        ],
        milestones=[warmup_epochs],
    )


@torch.no_grad()
def evaluate(model, loader, num_classes, amp_dtype, score_thresh=0.5):
    """Compute mAP@0.5 (+ per-class AP) and a simple precision/recall@0.5."""
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    tp = fp = fn = 0
    for images, targets in loader:
        images = [im.to(DEVICE) for im in images]
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            preds = model(images)
        preds_cpu, tgts_cpu = [], []
        for p, t in zip(preds, targets):
            pb = p["boxes"].float().cpu()
            ps = p["scores"].float().cpu()
            pl = p["labels"].long().cpu()
            gb = t["boxes"].float()
            gl = t["labels"].long()
            preds_cpu.append({"boxes": pb, "scores": ps, "labels": pl})
            tgts_cpu.append({"boxes": gb, "labels": gl})
            # precision/recall at score>=thresh, IoU>=0.5, greedy class-aware
            keep = ps >= score_thresh
            pbk, plk = pb[keep], pl[keep]
            matched_gt = set()
            for bi in range(len(pbk)):
                best_iou, best_j = 0.0, -1
                for j in range(len(gb)):
                    if j in matched_gt or gl[j] != plk[bi]:
                        continue
                    iou = _iou(pbk[bi], gb[j])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.5:
                    tp += 1; matched_gt.add(best_j)
                else:
                    fp += 1
            fn += len(gb) - len(matched_gt)
        metric.update(preds_cpu, tgts_cpu)
    res = metric.compute()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {
        "mAP": float(res["map_50"]) if float(res["map_50"]) >= 0 else 0.0,
        "precision": prec,
        "recall": rec,
        "per_class_ap": {},
    }
    # per-class AP
    if "map_per_class" in res and res["map_per_class"].numel() > 0:
        classes = res.get("classes", torch.arange(num_classes))
        for c, ap in zip(classes.tolist(), res["map_per_class"].tolist()):
            out["per_class_ap"][int(c)] = float(ap) if ap >= 0 else 0.0
    return out


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 FireRGBT manual-loop training")
    parser.add_argument("--config", default="configs/Detection/yolo11_fire.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision("high")

    run_name = cfg["run_name"] + "_manual"
    # EXPERIMENT_ROOT env overrides the (anonymised) config path at runtime.
    exp_root = os.environ.get("EXPERIMENT_ROOT") or cfg.get(
        "experiment_root", "/work/anon/experiments")
    experiment_dir = Path(f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}")
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    img = cfg.get("img_size", 640)
    amp_dtype = torch.bfloat16 if "bf16" in str(cfg.get("precision", "bf16-mixed")) else torch.float16
    num_classes = cfg["num_classes"]
    max_epochs = cfg.get("max_epochs", 50)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    clip_val = cfg.get("gradient_clip_val", 10.0)
    patience = cfg.get("patience", 10)

    # ---- data ----
    dm_cfg = DataModuleConfig(
        datasets=cfg["datasets"], class_map=cfg["class_map"],
        batch_size=cfg.get("batch_size", 16), num_workers=cfg.get("num_workers", 0),
        img_size=(img, img),
        per_dataset_kwargs=cfg.get("per_dataset_kwargs", {}),
    )
    dm = DetectionDataModule(
        cfg=dm_cfg,
        train_transform=build_train_transform((img, img)),
        eval_transform=build_eval_transform((img, img)),
    )
    dm.setup("fit")
    dm.setup("test")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # ---- model / optim ----
    model = YOLODetector(
        model_name=os.environ.get("YOLO_CKPT") or cfg.get("model_name", "yolo11l.pt"),
        num_classes=num_classes,
        enable_tracking=False, conf_thresh=cfg.get("conf_thresh", 0.05),
        iou_thresh=cfg.get("iou_thresh", 0.5), img_size=img,
    ).to(DEVICE)
    opt = build_optimizer(model, cfg.get("lr", 0.01), cfg.get("momentum", 0.937),
                          cfg.get("weight_decay", 5e-4))
    sched = build_scheduler(opt, warmup_epochs, max_epochs)

    # ---- wandb ----
    use_wandb = not args.no_wandb
    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb.init(
                project=cfg.get("wandb_project", "esa-dlstem"),
                # WANDB_ENTITY env overrides the (anonymised) config entity at runtime.
                entity=os.environ.get("WANDB_ENTITY") or cfg.get("wandb_entity", "anonymous"),
                name=run_name, config=cfg,
            )
        except Exception as e:
            print(f"[wandb] disabled ({e})"); wb = None

    class_names = {v: k for k, v in cfg["class_map"].items()}
    print("=" * 72)
    print(f"YOLOv11 FireRGBT MANUAL training: {run_name}")
    print(f"  precision={amp_dtype} epochs={max_epochs} warmup={warmup_epochs} "
          f"lr={cfg.get('lr')} clip={clip_val}")
    print(f"  output: {experiment_dir}")
    print("=" * 72, flush=True)

    best_map = -1.0
    best_path = experiment_dir / "checkpoints" / "best.pt"
    last_path = experiment_dir / "checkpoints" / "last.pt"
    epochs_no_improve = 0
    nan_skips = 0

    for epoch in range(max_epochs):
        model.train()
        lr = opt.param_groups[0]["lr"]
        run_loss = torch.zeros(3)
        n = 0
        for images, targets in train_loader:
            images = [im.to(DEVICE) for im in images]
            targets = to_device_targets(targets)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(images, targets)
                loss = out["loss"]
            if not torch.isfinite(loss):
                nan_skips += 1
                continue  # skip non-finite batch, do not poison weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None], clip_val)
            opt.step()
            run_loss += torch.tensor(
                [out["loss_box"].item(), out["loss_cls"].item(), out["loss_dfl"].item()])
            n += 1
        sched.step()
        mean_loss = (run_loss / max(n, 1))
        total_loss = float(mean_loss.sum())

        val = evaluate(model, val_loader, num_classes, amp_dtype,
                       cfg.get("visualization_score_thresh", 0.5))
        improved = val["mAP"] > best_map
        if improved:
            best_map = val["mAP"]
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_mAP": best_map, "num_classes": num_classes}, best_path)
        else:
            epochs_no_improve += 1
        # always keep the latest epoch's weights (guards against a too-early best)
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val_mAP": val["mAP"], "num_classes": num_classes}, last_path)

        pc = " ".join(f"{class_names.get(c, c)}={ap:.3f}"
                      for c, ap in sorted(val["per_class_ap"].items()))
        print(f"epoch {epoch:>2}/{max_epochs-1} lr={lr:.5f} "
              f"loss(box/cls/dfl)={mean_loss[0]:.3f}/{mean_loss[1]:.3f}/{mean_loss[2]:.3f} "
              f"total={total_loss:.3f} | val/mAP={val['mAP']:.4f} "
              f"P={val['precision']:.3f} R={val['recall']:.3f} [{pc}]"
              f"{' *best*' if improved else ''}", flush=True)

        if wb is not None:
            log = {"epoch": epoch, "lr": lr, "train/loss": total_loss,
                   "train/loss_box": float(mean_loss[0]), "train/loss_cls": float(mean_loss[1]),
                   "train/loss_dfl": float(mean_loss[2]), "val/mAP": val["mAP"],
                   "val/Precision": val["precision"], "val/Recall": val["recall"]}
            for c, ap in val["per_class_ap"].items():
                log[f"val/AP_{class_names.get(c, c)}"] = ap
            wb.log(log)

        if cfg.get("early_stopping", True) and epochs_no_improve >= patience:
            print(f"[early stop] no val/mAP improvement for {patience} epochs.", flush=True)
            break

    print(f"\nTraining done. best val/mAP={best_map:.4f} (skipped {nan_skips} non-finite batches).")

    # ---- test on best ----
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best (epoch {ckpt['epoch']}, val/mAP {ckpt['val_mAP']:.4f}) for test.")
    test = evaluate(model, test_loader, num_classes, amp_dtype,
                    cfg.get("visualization_score_thresh", 0.5))
    pc = {class_names.get(c, c): ap for c, ap in test["per_class_ap"].items()}
    print("=" * 72)
    print(f"TEST  mAP@0.5={test['mAP']:.4f}  P={test['precision']:.3f}  R={test['recall']:.3f}")
    print(f"  per-class AP: {pc}")
    print("=" * 72)

    test_out = {"test/mAP": test["mAP"], "test/Precision": test["precision"],
                "test/Recall": test["recall"], "test/per_class_ap": pc,
                "best_val_mAP": best_map, "best_ckpt": str(best_path)}
    with open(experiment_dir / "test_metrics.json", "w") as f:
        json.dump(test_out, f, indent=2)
    if wb is not None:
        wb.log({"test/mAP": test["mAP"], "test/Precision": test["precision"],
                "test/Recall": test["recall"]})
        wb.finish()
    print(f"Wrote {experiment_dir/'test_metrics.json'}")


if __name__ == "__main__":
    main()
