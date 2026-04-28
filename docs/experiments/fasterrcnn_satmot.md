# Cross-dataset Faster R-CNN fine-tuning on satellite MOT detection

> **Goal**: fine-tune a single Faster R-CNN detector that performs well on the
> union of LMOD + SAT-MTB + VISO + SDM-Car + AIR-MOT, with the anchor pyramid
> tuned for the small-object-dominated scale distribution observed in the
> satellite-imagery bbox stats.

## Files added by this experiment

| Role | Path |
| --- | --- |
| Config (canonical) | `configs/Detection/fasterrcnn_satmot.yaml` |
| Training script | `training_scripts/train_fasterrcnn_satmot.py` |
| Per-dataset eval | `eval_fasterrcnn_satmot.py` |
| Model wrapper | `models/fasterrcnn.py` (extended `FasterRCNNDetector`) |
| Transforms | `transforms.py` (added `build_satmot_train_transform` / `build_satmot_eval_transform`) |
| Datamodule | `lightning_modules/datamodule.py` (added `per_dataset_kwargs`) |
| LMOD bbox stats tool | `tools/analyze_lmod_bbox.py` |
| LMOD bbox stats report | `docs/bbox_stats/bbox_stats_report_lmod.md` |

## Datasets in the union

| Name | Root | Native classes | Frames (train) |
| --- | --- | :---: | :---: |
| LMOD | `/data/ESA_DLSTEM_2025/data/trafic/LMOD` | car, plane, ship, train | 6,754 |
| SAT-MTB (mot) | `/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB` | airplane, car, ship, train | 28,902 |
| VISO | `/data/ESA_DLSTEM_2025/data/trafic/VISO` | car, plane, ship, train | (per VISO loader) |
| SDM-Car | `/data/ESA_DLSTEM_2025/data/trafic/SDM-Car` | car | (per SDM loader) |
| AIR-MOT | `/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100` | airplane, ship | (per AIRMOT loader) |
| **Combined train ds** | | | **69,095** |
| **Combined val ds**   | | | **12,376** |

(`val` and `test` come from each dataset's own splits; the DataModule
`ConcatDataset`s them.)

## Class taxonomy (4 global classes, FasterRCNN labels 1..4, 0=background)

| global id | name | aliases used in raw data |
| :---: | --- | --- |
| 1 | plane | `plane` (LMOD/VISO), `airplane` (SAT-MTB/AIR-MOT) |
| 2 | car   | `car` |
| 3 | ship  | `ship` |
| 4 | train | `train` |

`num_classes = 5` (4 foreground + 1 background).

The class map (passed to every dataset's `_map_label`) explicitly aliases both
spellings to the same id, so no per-dataset shim is needed:

```yaml
class_map:
  plane:    1
  airplane: 1
  car:      2
  ship:     3
  train:    4
```

## Decisions

The five-way training and evaluation strategy was chosen up-front; the
following decisions are documented for the record.

| # | Decision | Rationale |
| :---: | --- | --- |
| 1 | 4-class union `{plane, car, ship, train}` | Aligns native categories across all 5 datasets; AIR-MOT/SAT-MTB's `airplane` aliased to `plane`. |
| 2 | `ConcatDataset` over all 5 train splits (no per-dataset weighting) | Simplest first pass; switch to a `WeightedRandomSampler` later if per-dataset val AP shows LMOD's 460k cars dominating. |
| 3 | Pooled val mAP during training; per-dataset val/test mAP **post-hoc** via `eval_fasterrcnn_satmot.py` | Avoids invasive changes to `ObjectDetectionModule`. The post-hoc script runs the saved checkpoint 6 times (5 datasets + pooled). |
| 4 | Input resolution `min_size=1024, max_size=1333` + `RandomCrop(1024×1024)` for training | LMOD averages 51 px² per box and is 98.8% small-object; resizing to the torchvision default 800-side further shrinks them. RandomCrop preserves native pixel scale; eval lets the model's internal `GeneralizedRCNN` transform handle the resize. |
| 5 | Multi-scale anchor pyramid + relaxed RPN/box thresholds | See "Model" below. |

## Model (Faster R-CNN ResNet-50 FPN v1)

We use **v1**, not v2: `fasterrcnn_resnet50_fpn_v2` hard-codes its own
`rpn_anchor_generator` and rejects user overrides; v1 forwards `**kwargs` to
`FasterRCNN(...)` so a custom anchor generator goes in cleanly.

### Anchor generator

torchvision's `RPNHead` uses **a single** `num_anchors` value applied to all
FPN levels, so anchor counts must be uniform per location (per FPN level).
We therefore use **2 sizes × 3 ratios = 6 anchors per location**, with the
size pyramid spanning 4 → 512 px:

| FPN level | stride | anchor sizes (px) | covers | dataset evidence |
| --- | :---: | :---: | --- | --- |
| P2 | 4  | 4, 8     | tiny | LMOD car (avg 4.7×4.4), VISO car, SDM-Car |
| P3 | 8  | 16, 32   | small | LMOD plane (avg 36×33), AIR-MOT plane, SAT-MTB ship (avg 20) |
| P4 | 16 | 48, 96   | medium | SAT-MTB airplane (avg 50×50), LMOD train (~25×63) |
| P5 | 32 | 128, 256 | large | SAT-MTB ship up to 200, mid-size SAT-MTB train |
| P6 | 64 | 256, 512 | very large | SAT-MTB train (max 304×203 ≈ 512 along long side) |

Aspect ratios: `(0.5, 1.0, 2.0)` on every level. Captures the elongated
shapes of trains and side-on planes/ships.

### Relaxed RPN / box thresholds

Small-object IoU is fragile (1 px centroid offset can drop IoU 0.1+).
torchvision's defaults starve the head of positive samples on satellite data.

| Threshold | torchvision default | sat-MOT setting | Why |
| --- | :---: | :---: | --- |
| `rpn_fg_iou_thresh` | 0.7 | **0.5** | Anchor-to-GT match still labeled positive at moderate overlap. |
| `rpn_bg_iou_thresh` | 0.3 | **0.2** | Reduce ignored band. |
| `box_fg_iou_thresh` | 0.5 | **0.4** | Same logic at the second stage. |
| `box_bg_iou_thresh` | 0.5 | **0.3** | torchvision asserts `bg ≤ fg`; lowered to remain valid. |
| `rpn_pre_nms_top_n_train` | 2000 | **4000** | More small-object proposals survive NMS. |
| `rpn_post_nms_top_n_train` | 2000 | **2000** | Same as default. |

### Other model settings

- `pretrained: true` (COCO weights for FPN backbone + RPN; the box predictor
  is replaced for `num_classes=5`)
- `trainable_backbone_layers: 3`
- `score_thresh: 0.05`, `nms_thresh: 0.5`, `detections_per_img: 300`
  (bumped from 100; satellite frames have many small instances)
- `min_size: 1024`, `max_size: 1333` (input image scaling done internally)

## Training transforms (RandomCrop 1024 to preserve native scale)

`build_satmot_train_transform(crop_size=1024)` chain:

1. `A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, fill=0)`
2. `A.RandomCrop(height=1024, width=1024)`
3. `A.HorizontalFlip(p=0.5)`
4. `A.VerticalFlip(p=0.2)` — most satellite frames are top-down, so vertical
   flip is geometrically valid (unlike ground-camera datasets).
5. `A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3)`

`bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1)`. The
`min_visibility=0.1` is looser than the default `0.3`: small-object boxes
are easy to clip near a crop edge and we don't want to drop too many.

`build_satmot_eval_transform()` is a no-op spatial transform: the model's
internal `GeneralizedRCNNTransform` handles `min_size=1024, max_size=1333`
resizing (preserves aspect ratio).

## Lightning training settings

| Knob | Value | Notes |
| --- | :---: | --- |
| `batch_size` | 4 | 1024² + 6 anchors × 5 levels is heavy; bump to 8 if GPU headroom allows. |
| `num_workers` | 0 | Per-project rule (fork+CUDA deadlock); `dm.train_dataloader` keeps fork off. |
| `max_epochs` | 30 | First-pass; reduce to 10–15 for a faster sanity run. |
| `lr` | `5e-4` | Same as `train_fasterrcnn_birdsai_mot.py`. |
| `weight_decay` | `1e-4` | AdamW. |
| `lr_scheduler` | `cosine` | Linear warmup → cosine decay. |
| `warmup_epochs` | 3 | Lightning module's built-in warmup wrapper. |
| `precision` | `16-mixed` | Fits memory on RTX 5000 Ada (32 GiB). |
| `monitor_metric` | `val/mAP` | ModelCheckpoint + EarlyStopping both watch this. |
| `monitor_mode` | `max` | mAP — higher is better. |
| `patience` | 8 | EarlyStopping. |

## Logging / outputs

- **W&B** project `esa-dlstem`, entity `chengziwen693`, run name `fasterrcnn_satmot`.
- **Local experiment dir**: `/work/ziwen/experiments/fasterrcnn_satmot_<TS>/`
  - `checkpoints/best-epoch=<N>-val_mAP=<X>.ckpt`
  - `pr_curve.json` / `pr_curve.png` (test split, written by
    `ObjectDetectionModule._save_pr_curves` to `default_root_dir`)
  - Visualization callback writes annotated frames + sends up to 50 to W&B
    (`visualization_max_wandb_images: 50`, `score_thresh: 0.5`).

## Per-dataset post-hoc evaluation

`eval_fasterrcnn_satmot.py` runs the saved checkpoint **6 times**:

| Pass | datasets | output dir |
| --- | --- | --- |
| 1 | LMOD          | `<eval_root>/LMOD/` |
| 2 | SAT-MTB       | `<eval_root>/SAT-MTB/` |
| 3 | VISO          | `<eval_root>/VISO/` |
| 4 | SDM-Car       | `<eval_root>/SDM-Car/` |
| 5 | AIR-MOT       | `<eval_root>/AIR-MOT/` |
| 6 | **POOLED** all 5 | `<eval_root>/POOLED/` |

Each writes its own `pr_curve.json` / `pr_curve.png`. A single `summary.csv`
aggregates per-dataset `val_mAP / test_mAP / test_Precision / test_Recall /
test_F1 / test_AP_overall / test_fps`.

`<eval_root>` defaults to `<ckpt_dir>/../eval_<timestamp>/`.

## Smoke-test results (random init, before training)

| Check | Result |
| --- | --- |
| Model construct + train forward (random data) | loss = 2.41 ✓ |
| Model construct + eval forward | returns 100 detections ✓ |
| `num_anchors_per_location()` | `[6, 6, 6, 6, 6]` ✓ uniform |
| DataModule `setup('fit')` | train ds = 69,095 frames, val ds = 12,376 ✓ |
| Real-batch train forward | loss = 2.42 ✓ |
| Class label alias (SAT-MTB ship → label 3, SDM-Car car → label 2) | ✓ |
| Per-dataset kwargs routing (`task: mot` only to SAT-MTB) | ✓ |

## How to launch

### Train

```bash
micromamba run -n esa_dlstem python training_scripts/train_fasterrcnn_satmot.py \
  --config configs/Detection/fasterrcnn_satmot.yaml
```

### Per-dataset eval after training

```bash
micromamba run -n esa_dlstem python eval_fasterrcnn_satmot.py \
  --config configs/Detection/fasterrcnn_satmot.yaml \
  --ckpt   /work/ziwen/experiments/fasterrcnn_satmot_<TS>/checkpoints/best-*.ckpt
```

## Open follow-ups (not done, may want to revisit)

1. **Dataset weighting**: if LMOD's 460k cars dominate the loss and other
   datasets' val AP suffers, switch from `ConcatDataset` to a
   `WeightedRandomSampler` that gives each dataset roughly equal frame share.
2. **Per-dataset val AP during training** (currently only post-hoc): would
   require subclassing `ObjectDetectionModule` to dispatch metrics on
   `target["dataset"]`.
3. **Train-AP per class**: torchmetrics MAP gives per-class AP at val/test
   only; if class imbalance hurts (e.g. SAT-MTB train has 14× fewer boxes
   than airplane), a focal-loss head or per-class loss weighting could help.
4. **OBB**: SAT-MTB ships up to 33k px² with elongated aspect ratios are HBB
   here; for tight rotated boxes we'd need a separate OBB-aware head (out of
   scope for this experiment).
5. **`fasterrcnn_resnet50_fpn_v2` support**: currently can't accept custom
   anchors; would need post-construction surgery on `model.rpn` if v2's
   stronger box head is wanted.
