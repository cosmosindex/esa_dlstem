"""
Smoke test: FasterRCNN + ObjectDetectionModule on uav1 frames.

Tests that train / val / test steps all run without errors.
Run from the project root:
    python test_smoke.py
"""

import os
import re
import glob

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import lightning as L

from models import FasterRCNNDetector
from lightning_modules.module import ObjectDetectionModule


# ---------------------------------------------------------------------------
# 1. Parse MOT-format ground truth
# ---------------------------------------------------------------------------

def parse_mot_gt(gt_path: str) -> dict[int, list]:
    """
    Read uav1/labels/gt.txt (MOT format).

    Returns:
        dict mapping frame_number -> list of (x1, y1, x2, y2, track_id)
        Only rows with consider_in_eval == 1 are kept.
    """
    gt: dict[int, list] = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame      = int(float(parts[0]))
            track_id   = int(float(parts[1]))
            x          = float(parts[2])
            y          = float(parts[3])
            w          = float(parts[4])
            h          = float(parts[5])
            consider   = int(float(parts[6]))   # 1 = count in eval, 0 = skip
            if consider == 0:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            gt.setdefault(frame, []).append((x1, y1, x2, y2, track_id))
    return gt


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class UAV1Dataset(Dataset):
    """
    Pairs each image frame with its MOT ground-truth annotations.
    Only frames that have at least one GT box are included.

    Each item returned:
        image  : (C, H, W) float32 tensor, values in [0, 1]
        target : dict with keys
                    'boxes'     – (N, 4) xyxy float32
                    'labels'    – (N,)   int64, all 1 (single foreground class)
                    'track_ids' – (N,)   int64
    """

    def __init__(self, img_dir: str, gt_path: str):
        all_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")), key=_natural_key
        )
        gt = parse_mot_gt(gt_path)

        # Keep only frames that appear in the GT
        # Image filename stem (e.g. "0000098") -> int 98 -> matches GT frame number
        self.samples = []
        for fp in all_paths:
            stem = os.path.splitext(os.path.basename(fp))[0]
            frame_num = int(stem)
            if frame_num in gt:
                self.samples.append((fp, gt[frame_num]))

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fp, annots = self.samples[idx]

        img_bgr = cv2.imread(fp)
        H, W    = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image   = self.to_tensor(img_rgb)          # (C, H, W) float [0, 1]

        # Build target tensors
        boxes_list, tid_list = [], []
        for (x1, y1, x2, y2, tid) in annots:
            # Clip to image bounds and skip degenerate boxes
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(0.0, min(x2, W - 1))
            y2 = max(0.0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_list.append([x1, y1, x2, y2])
            tid_list.append(tid)

        if boxes_list:
            boxes     = torch.tensor(boxes_list, dtype=torch.float32)
            labels    = torch.ones(len(boxes_list), dtype=torch.long)   # class 1
            track_ids = torch.tensor(tid_list, dtype=torch.long)
        else:
            boxes     = torch.zeros((0, 4), dtype=torch.float32)
            labels    = torch.zeros(0, dtype=torch.long)
            track_ids = torch.zeros(0, dtype=torch.long)

        target = {"boxes": boxes, "labels": labels, "track_ids": track_ids}
        return image, target


def collate_fn(batch):
    """Keep images as a list and targets as a list (required by FasterRCNN)."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    IMG_DIR = "uav1/img"
    GT_PATH = "uav1/labels/gt.txt"

    # --- Dataset & splits ---
    dataset = UAV1Dataset(IMG_DIR, GT_PATH)
    print(f"Frames with GT annotations: {len(dataset)}")

    n       = len(dataset)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=2, shuffle=False, collate_fn=collate_fn)

    # --- Model ---
    # num_classes = 1 foreground class + 1 background = 2
    detector = FasterRCNNDetector(num_classes=2)

    lit_model = ObjectDetectionModule(
        model=detector,
        has_tracking=True,
        lr=1e-4,
    )

    # --- Trainer (limit batches so it finishes quickly) ---
    trainer = L.Trainer(
        # max_epochs=1,
        # limit_train_batches=3,   # run only 3 batches per phase
        # limit_val_batches=3,
        # limit_test_batches=3,
        # accelerator="auto",
        # logger=False,
        # enable_checkpointing=False,
        fast_dev_run=True
    )

    print("\n--- fit (train + val) ---")
    trainer.fit(lit_model, train_loader, val_loader)

    print("\n--- test ---")
    trainer.test(lit_model, test_loader)

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
