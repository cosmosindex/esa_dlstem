"""
Unified SAM3 evaluation script.

Supports any registered dataset via a YAML config file. Mirrors eval_sam2.py
but uses SAM3Tracker + VideoTrackerEvaluationModule.

Usage:
    python eval_sam3.py --config configs/SOT/sam3_ootb.yaml
"""

# --- repo root on path so top-level modules (transforms, obb_utils, ...) import ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from datetime import datetime

import torch
import lightning as L
import yaml
from lightning.pytorch.loggers import WandbLogger

from models import SAM3Tracker, SAM3TextTracker, SAM31TextTracker
from lightning_modules import (
    SAM2DataModule,
    SAM2DataModuleConfig,
    VideoTrackerEvaluationModule,
    SAM2VisualizationCallback,
    SAM2SOTEvalCallback,
    MOTFormatDumpCallback,
)
from transforms import build_eval_transform


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SAM3 evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")

    # --- Names ---
    dataset_name = cfg["dataset"]
    prompt_strategy = cfg.get("prompt_strategy", "first_frame")
    prompt_interval = cfg.get("prompt_interval", 10)

    # The benchmark slug used in the run dir name. Defaults to the
    # dataset_name lowercased, but configs that share a dataset class
    # (e.g. VISO with `categories: [plane, ship, train]`) can override
    # via `run_name_suffix` so compute_hota can tell the variants apart.
    bench_slug = cfg.get("run_name_suffix", dataset_name.lower())
    # ``model_version`` controls which SAM3 release is loaded. Default
    # "sam3" preserves prior behaviour; "sam3.1" routes to the multiplex
    # tracker. The model-version prefix lands in the run-dir name so
    # compute_hota.py picks up base-vs-multiplex as distinct trackers.
    model_version = str(cfg.get("model_version", "sam3")).lower()
    if model_version in ("sam3.1", "sam31", "sam3p1"):
        run_prefix = "sam3p1"
    else:
        run_prefix = "sam3"
    run_name = f"{run_prefix}_{prompt_strategy}_{bench_slug}"
    if prompt_strategy == "every_n":
        run_name = f"{run_prefix}_every{prompt_interval}_{bench_slug}"

    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/ziwen/experiments")
    experiment_dir = f"{exp_root}/{run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # --- Transform ---
    img_size = cfg.get("img_size")
    eval_transform = build_eval_transform(tuple(img_size)) if img_size else None

    # --- Data ---
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
            dataset_kwargs=cfg.get("dataset_kwargs", {}),
        ),
        eval_transform=eval_transform,
    )

    # --- Model ---
    tracker_type = cfg.get("tracker_type", "box")
    is_sam31 = run_prefix == "sam3p1"
    if tracker_type == "text":
        # Class names ordered by their integer id, so label assignment is stable
        class_names_ordered = [
            name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])
        ]
        if is_sam31:
            tracker = SAM31TextTracker(
                class_names=class_names_ordered,
                label_to_id=class_map,
                checkpoint_path=cfg.get("sam3_checkpoint_path"),
                max_num_objects=cfg.get("sam31_max_num_objects", 16),
                multiplex_count=cfg.get("sam31_multiplex_count", 16),
                compile=cfg.get("sam31_compile", False),
                # FA3 / real RoPE require flash_attn_interface to be
                # installed; default off so the multiplex predictor falls
                # back to PyTorch SDPA.
                use_fa3=cfg.get("sam31_use_fa3", False),
                use_rope_real=cfg.get("sam31_use_rope_real", False),
                apply_temporal_disambiguation=cfg.get("apply_temporal_disambiguation", True),
            )
        else:
            tracker = SAM3TextTracker(
                class_names=class_names_ordered,
                label_to_id=class_map,
                checkpoint_path=cfg.get("sam3_checkpoint_path"),
                apply_temporal_disambiguation=cfg.get("apply_temporal_disambiguation", True),
            )
    else:
        if is_sam31:
            raise NotImplementedError(
                "SAM 3.1 box-prompt MOT is not wired yet — only the text path "
                "(tracker_type=text) is supported. Use sam3 for box-prompt SOT."
            )
        tracker = SAM3Tracker(
            checkpoint_path=cfg.get("sam3_checkpoint_path"),
            apply_temporal_disambiguation=cfg.get("apply_temporal_disambiguation", True),
        )

    eval_mode = cfg.get("eval_mode", "sot")
    sot_mode = eval_mode == "sot"

    module = VideoTrackerEvaluationModule(
        model=tracker,
        prompt_strategy=prompt_strategy,
        prompt_interval=prompt_interval,
        sot_mode=sot_mode,
    )

    # --- Logger ---
    logger = WandbLogger(
        project=cfg.get("wandb_project", "esa-dlstem"),
        entity=cfg.get("wandb_entity", "chengziwen693"),
        name=run_name,
        log_model=False,
    )

    # --- Callbacks ---

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

    # Dump per-video MOTChallenge text files when requested. Required as
    # the upstream stage of the RAFT static-tracklet filter + HOTA eval
    # pipeline (compute_hota.py reads <run_dir>/mot_format/<seq>.txt).
    if cfg.get("dump_mot_format", False) and not sot_mode:
        callbacks.append(
            MOTFormatDumpCallback(
                output_dir=experiment_dir,
                score_thresh=cfg.get("mot_dump_score_thresh", 0.0),
            )
        )

    # --- Trainer ---
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=experiment_dir,
    )

    print("=" * 60)
    print(f"SAM3 Evaluation: {prompt_strategy} | {dataset_name} | "
          f"{'native' if img_size is None else f'{img_size[0]}x{img_size[1]}'}")
    print(f"Output: {experiment_dir}")
    print("=" * 60)

    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
