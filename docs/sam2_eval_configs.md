# SAM2 Unified Evaluation

## Usage

```bash
python eval_sam2.py --config configs/<config_name>.yaml
```

## Available Configs

| Config | Dataset | Resolution | Eval Mode |
|---|---|---|---|
| `configs/sam2_satsot.yaml` | SatSOT | native (`img_size: null`) | SOT |
| `configs/sam2_ootb.yaml` | OOTB | 640x640 | SOT |
| `configs/sam2_birdsai_mot.yaml` | BIRDSAI_MOT | 640x640 | MOT |
| `configs/sam2_birdsai_sot.yaml` | BIRDSAI | 640x640 | SOT |

## Config Fields

```yaml
# --- Dataset ---
dataset: SatSOT                    # registry key in sam2_datamodule.py
dataset_root: /path/to/dataset     # root directory
class_map:                         # category name → class id (0-indexed)
  car: 0
  plane: 1

# --- Eval mode ---
eval_mode: sot                     # "sot" or "mot"

# --- Image size ---
img_size: null                     # null = native resolution, [H, W] = resize

# --- SAM2 model ---
sam2_model_id: facebook/sam2.1-hiera-large

# --- Prompt strategy ---
prompt_strategy: first_frame       # "first_frame" or "every_n"
prompt_interval: 10                # only used when prompt_strategy == "every_n"

# --- Clip parameters ---
clip_len: 32
clip_stride: 1
batch_size: 1
num_workers: 0

# --- Visualization / metrics ---
iou_thresh: 0.3
score_thresh: 0.5
max_wandb_images: 50

# --- W&B ---
wandb_project: esa-dlstem
wandb_entity: chengziwen693
```

## Key Design Decisions

- **`img_size: null`** — no resize transform is applied; frames are fed at native resolution. Required for SatSOT where targets are already very small (~10 px) and resizing to 640x640 would distort aspect ratios across sequences (each has a different resolution).
- **`img_size: [H, W]`** — applies `build_eval_transform(img_size)` (Albumentations `A.Resize`). Used for datasets like OOTB and BIRDSAI where a fixed input size is acceptable.
- **`eval_mode: sot`** — attaches `SAM2SOTEvalCallback` (Success AUC, Precision@20) and sets `sot_mode=True` on visualization callback.
- **`eval_mode: mot`** — visualization callback only (`sot_mode=False`); MOTA, IDF1, TP/FP/FN are computed internally by `SAM2EvaluationModule.on_test_epoch_end()`.

## Metrics by Eval Mode

### SOT (`eval_mode: sot`)

| Metric | Description |
|---|---|
| Success AUC | Area under success curve (IoU threshold sweep 0→1) |
| Precision@20 | Fraction of frames with center distance ≤ 20 px |
| Mean IoU | Average best IoU across all frames |
| Per-category breakdown | Success AUC and Precision@20 per object category |
| Per-size breakdown | Small (<32x32) vs large |

### MOT (`eval_mode: mot`)

| Metric | Description |
|---|---|
| MOTA | 1 − (FP + FN + ID switches) / GT |
| IDF1 | ID F1 combining tracking precision and recall |
| AP50 | Mean average precision at IoU=0.5 |
| Precision / Recall | Per-frame detection metrics |
| ID switches | Number of identity switches |
| FPS | Inference speed |

## Legacy Scripts

The per-dataset eval scripts below are superseded by `eval_sam2.py` and kept for reference only:

- `eval_sam2_ootb.py` → use `configs/sam2_ootb.yaml`
- `eval_sam2_birdsai.py` → use `configs/sam2_birdsai_mot.yaml` or `configs/sam2_birdsai_sot.yaml`
