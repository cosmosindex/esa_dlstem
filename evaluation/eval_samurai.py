"""
Unified SAMURAI evaluation script.

SAMURAI = SAM 2.1 + motion-aware (Kalman) memory, zero-shot visual tracker.
Ships a forked `sam2` package under `samurai/sam2/`. To avoid conflicting
with the pip-installed `sam2` used by the other eval scripts, we import
`SamuraiTracker` BEFORE any other code touches `sam2` — its constructor
activates samurai's fork as the process-wide `sam2`.

Usage:
    python eval_samurai.py --config configs/SOT/samurai_ootb.yaml
"""

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# IMPORTANT: import SamuraiTracker first so samurai's sam2 fork is activated
# before any other module pulls in the pip-installed sam2.
from models.samurai import SamuraiTracker  # noqa: E402, isort: skip

import argparse  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402
import lightning as L  # noqa: E402
import yaml  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402

from lightning_modules import (  # noqa: E402
    SAM2DataModule,
    SAM2DataModuleConfig,
    VideoTrackerEvaluationModule,
    SAM2VisualizationCallback,
    SAM2SOTEvalCallback,
)
from transforms import build_eval_transform  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SAMURAI evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    dataset_name = cfg["dataset"]
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)
    model_name = cfg.get("model_name", "large")

    run_name = f"samurai_{model_name}_{prompt_strategy}_{dataset_name.lower()}"
    if prompt_strategy == "every_n":
        run_name = f"samurai_{model_name}_every{prompt_interval}_{dataset_name.lower()}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    img_size = cfg.get("img_size")
    eval_transform = build_eval_transform(tuple(img_size)) if img_size else None

    class_map = cfg["class_map"]
    class_names = {v: k for k, v in class_map.items()}

    dm = SAM2DataModule(
        cfg=SAM2DataModuleConfig(
            datasets={dataset_name: cfg["dataset_root"]},
            class_map=class_map,
            clip_len=cfg.get("clip_len", 32),
            clip_stride=cfg.get("clip_stride", 1),
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 0),
            split=os.environ.get("SOT_SPLIT", cfg.get("split", "test")),
        ),
        eval_transform=eval_transform,
    )

    tracker = SamuraiTracker(
        model_name=model_name,
        ckpt_path=cfg.get("ckpt_path"),
    )

    eval_mode = cfg.get("eval_mode", "sot")
    sot_mode = eval_mode == "sot"

    module = VideoTrackerEvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=sot_mode,
    )

    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "anonymous"),
        name=run_name,
        log_model=False,
    )

    callbacks = [
        SAM2VisualizationCallback(
            class_names=class_names,
            output_dir=experiment_dir,
            iou_thresh=cfg.get("iou_thresh", 0.3),
            max_wandb_images=cfg.get("max_wandb_images", 50),
            score_thresh=cfg.get("score_thresh", 0.5),
            sot_mode=sot_mode,
        ),
    ]

    if sot_mode:
        callbacks.append(
            SAM2SOTEvalCallback(
                class_names=class_names,
                output_dir=experiment_dir,
                score_thresh=cfg.get("score_thresh", 0.5),
            )
        )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    print("=" * 60)
    print(f"SAMURAI Evaluation: {model_name} | {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
