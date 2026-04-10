# Spaceborne Video Datasets (2020–2025)

A curated list of publicly available spaceborne video datasets for computer vision tasks, including object detection, tracking, segmentation, video super-resolution, and infrared small target detection.

> **Legend:**
> - ✅ Already collected
> - 🆕 Newly added
> - 🔄 Updated entry

---

## Dataset Table

| Dataset | Satellite / Sensor | Modality | #Videos / Seq | Downstream Task | Venue / Year | Link |
|---|---|---|---|---|---|---|
| ✅ **SatSOT** | Jilin-1 | VNIR | 105 seq | SOT | TGRS 2022 | [IEEE](https://ieeexplore.ieee.org/document/9672083/) |
| ✅ **AIR-MOT** | Jilin-1 | VNIR | 10 | MOT | TGRS 2022 | [IEEE](https://ieeexplore.ieee.org/document/9715124) · [GitHub](https://github.com/HeQibin/TGraM) |
| ✅ **SAT-MTB** | Jilin-1 | VNIR | 249 | Det · MOT · Seg | TGRS 2023 | [Zenodo](https://zenodo.org/records/15253996) · [IEEE](https://ieeexplore.ieee.org/document/10130311)|
| ✅ **VISO (SatVideoDT)** | Jilin-1 | VNIR | 100 | Det · MOT | TGRS 2022 | [GitHub](https://github.com/QingyongHu/VISO) · [Project](https://satvideodt.github.io/) |
| 🔄 **IRSatVideo-LEO** | Landsat 8/9 (semi-simulated) | Infrared (TIR) | 200 | IR MIRST Det · SOT · MOT · Seg | **TGRS 2025** | [arXiv](https://arxiv.org/abs/2409.12448) · [GitHub](https://github.com/XinyiYing/RFR) |
| ✅ **SV248S** | Jilin-1 | VNIR | 248 | SOT | — | [GitHub](https://github.com/xdai-dlgvv/SV248S) |
| ✅ **OOTB** | Multiple (JL-1, SkySat, ISS) | VNIR | 110 | SOT (oriented bbox) | ISPRS 2024 | [GitHub](https://github.com/YZCU/OOTB) |
| ✅ **LMOD** | Jilin-1 | VNIR | 8 | Large-scale MOD | TGRS 2025 | [GitHub](https://github.com/RS-Devotee/LMOD) · [IEEE](https://ieeexplore.ieee.org/document/11142571) |
| 🆕 **SAT-MTB-SOS** | Jilin-1 | VNIR | 113 | Single-object Seg (VOS) | CVPR 2024 W | [IEEE](https://ieeexplore.ieee.org/document/10604055/) |
| ✅ **SDM-Car** | Luojia 3-01 | VNIR | 99 | Small/dim vehicle Det · MOT | GRSL 2024 | [GitHub](https://github.com/TanedaM/SDM-Car) · [IEEE](https://ieeexplore.ieee.org/document/10746500/) |

---

## Task Abbreviations

| Abbreviation | Full Name |
|---|---|
| SOT | Single Object Tracking |
| MOT | Multiple Object Tracking |
| Det | Object Detection |
| MOD | Moving Object Detection |
| Seg | Segmentation |
| VOS | Video Object Segmentation |
| VSR | Video Super-Resolution |
| IR MIRST Det | Multi-frame Infrared Small Target Detection |

---

## Sensor / Modality Abbreviations

| Abbreviation | Description |
|---|---|
| VNIR | Visible Near-Infrared (RGB optical) |
| MIR | Mid-wave Infrared |
| TIR | Thermal Infrared |
| MWIR | Mid-Wave Infrared |
| SWIR | Short-Wave Infrared |

---

## Notes

- **IRSatVideo-LEO**: Semi-simulated dataset. Backgrounds from Landsat 8/9; satellite motion, target appearance, trajectory and intensity are synthesized. Formally published in TGRS 2025 (previously arXiv 2024).
  - **200 sequences**, 1024×1024 TIR frames, 91,021 total frames. 16 geographic regions (e.g. EastAfrica, NorthAmericaEast, WestEurope, …).
  - **Annotations**: Pascal VOC XML bounding boxes (xyxy) + binary segmentation masks (uint8, 0/255). Objects named `target0`, `target1`, … — numeric suffix = track ID, single category `"target"`.
  - **Official split**: train (160) / test (40). Test further divided by difficulty: easy (17) / middle (13) / hard (10).
  - **Our split**: train 160 (73,699 frames) / val 13 (5,267 frames) / test 27 (12,055 frames). Val carved from 30% of official test, stratified by region, `seed=42`.
  - **Tasks**: Det, SOT, MOT, Seg. Detection mode carries track_ids for ByteTrack eval; video mode for SAM2/tracking; `load_mask()` for segmentation.
- **AIR-MOT-100**: Multi-object tracking dataset from Jilin-1 satellite video (extended version of original 10-sequence AIR-MOT).
  - **100 sequences**, 1920×1080 JPEG frames, ~22,669 total frames. 2 classes: airplane (class 1, 30 seqs) and car (class 2, 39 seqs). **31 sequences have empty annotations** and are excluded → 69 usable sequences, 19,940 frames.
  - **Annotations**: MOT CSV format — `frame_id, track_id, x, y, w, h, conf, class, visibility`. x/y = top-left. Class mapping: 1=airplane, 2=car.
  - **Black padding bars** (among the 69 usable sequences): 45 clean, 17 bottom-only (400px), 6 right-only (1440px), 1 both. (Full 100-seq stats: 59 clean, 29 bottom, 10 right, 2 both.) All annotations within valid content area. Does not significantly affect training — black bars are just padding that gets resized.

    | Black bar status | Count | Effective content area |
    |---|---:|---|
    | No black bars | 45 | 1920×1080 (full) |
    | Bottom only (400px) | 17 | 1920×680 |
    | Right only (1440px) | 6 | 480×1080 |
    | Bottom + right | 1 | 480×680 |
    | **Total** | **69** | |
  - **No official split**. Our split: 80/10/10 stratified by dominant class, `seed=42` → train 55 (15,559 frames) / val 7 (2,282 frames) / test 7 (2,099 frames).
  - **Implementation**: `AIRMOTDataset(root, split)`. Registered as `"AIR-MOT"` in both DataModule registries. Seq 100 has special image naming (`000001_8.jpg`), handled automatically.
- **SatSOT**: SOT dataset from Jilin-1 satellite video.
  - **105 sequences**, variable resolution (e.g. 335×499), 27,664 total frames. 4 categories: car (65), plane (9), ship (5), train (26).
  - **Annotations**: `groundtruth.txt` per sequence — one `x,y,w,h` line per frame (top-left + size). Frames with `none` = target absent/occluded.
  - **No official split**. Our split: 80/10/10 stratified by category, `seed=42` → train 83 (22,718 frames) / val 11 (2,478 frames) / test 11 (2,468 frames).
  - **JSON metadata**: `SatSOT.json` with per-sequence attributes (BC, DEF, ARC, ROT, etc.) and `gt_rect` in xywh.
- **NUDT-MIRSDT-HiNo**: Extension of NUDT-MIRSDT with higher noise levels. Currently preprint (arXiv 2506.12766, v4 Jan 2026), submitted to IEEE TPAMI — not yet formally accepted. Semi-synthetic sequences; **not real satellite video**.
- **SAT-MTB**: Multi-task satellite video benchmark from Jilin-1 satellite. Published in IEEE TGRS 2023.
  - **240 sequences** across 4 coarse-grained categories: airplane (62), car (92), ship (70), train (16). Variable resolution (512×512 to 3000×1500). 12 fine-grained subcategories (e.g. `narrow_bodied_aircraft`, `yacht`, `speed_boat`).
  - **Annotation availability varies by category**:
    - airplane / ship: HBB (XML) + OBB (XML) + MOT (CSV) + Seg (COCO JSON) — all tasks.
    - car: **MOT only** — no detection or segmentation annotations. This is why car is absent from the det/seg data splits.
    - train 01-07: all tasks. train 08-10: det/seg only (no MOT). train 11-16: MOT only (no det/seg). This explains why only a subset of train sequences appears in each split sheet.
  - **Annotation formats**:
    - `det/HBB/*.xml`: Pascal VOC-like with `<name>` (coarse), `<subname>` (fine), `<objectID>` (track ID), `<bndbox>` (xmin/ymin/xmax/ymax).
    - `det/OBB/*.xml`: Same structure but with `<robndbox>` (4 corner points x0-x3, y0-y3) instead of `<bndbox>`.
    - `mot/<seq>` or `mot/<seq>.txt`: CSV with columns `frame_id, object_id, x, y, w, h, conf, class, subclass, 0, 0`. x/y = top-left. Class mapping: 0=car, 1=airplane, 2=ship, 3=train.
    - `seg/*.json`: COCO-like per-frame JSON with polygon segmentation + bbox in `[xmin, ymin, xmax, ymax]` format. `supercategory` = coarse class, `name` = fine-grained.
  - **Official split**: train / test from `data_split.xlsx` (4 sheets: `det_HBB`, `det_OBB`, `seg`, `mot`). Each sheet has different sequence coverage due to annotation availability.
  - **Our split**: Val carved from 30% of official test, stratified by category, `seed=42`.
    - det_hbb: train 84 (18,932 frames) / val 18 (4,153 frames) / test 40 (9,868 frames) — 142 videos, 3 categories.
    - det_obb: train 61 (12,573 frames) / val 14 (3,104 frames) / test 31 (7,101 frames) — 106 videos, 3 categories.
    - mot: train 149 (28,902 frames) / val 27 (5,538 frames) / test 61 (13,950 frames) — 237 videos, 4 categories.
    - seg: same as det_hbb (142 videos, 3 categories).
  - **Implementation**: `SATMTBDataset(root, split, task="det_hbb"|"det_obb"|"mot"|"seg")`. Registered as `"SAT-MTB"` in both DataModule registries. `load_masks()` for instance segmentation (task=seg only).
- **SAT-MTB-SOS**: Subset of SAT-MTB, re-annotated with pixel-level masks for VOS. No standalone public download link found; dataset may be requested from authors or obtained via the parent SAT-MTB on [Zenodo](https://zenodo.org/records/15253996).
- **VISO / SatVideoDT**: Satellite video detection & tracking dataset from Jilin-1. Non-commercial use only. Download via [Google Drive](https://github.com/QingyongHu/VISO) or Baidu Netdisk (code: VISO).
  - **47 sequences** across 4 categories: car (38), plane (6), ship (2), train (1). Variable resolution (247×286 to 1454×750). 16,204 total frames.
  - **4 annotation formats provided** (same underlying data):
    - `mot/`: MOT-format `gt.txt` per sequence — used by our implementation.
    - `sot/`: Per-track text files (`trackID_startFrame_endFrame.txt`).
    - `coco/`: COCO JSON with train/val/test image directories.
    - `voc/`: Pascal VOC XML with `ImageSets/Main/{train,val,test}.txt`.
  - **Annotation format inconsistency** in MOT `gt.txt`:
    - Car / Train: comma-delimited `frame,obj_id,x,y,w,h,conf,cls,r1,r2` (xywh).
    - Plane / Ship: space-delimited `frame obj_id x1 y1 x2 y2 r1 r2 r3 r4` (xyxy).
    - Our parser auto-detects delimiter and coordinate format.
  - **Official split**: COCO/VOC provide frame-level train/val/test splits. Ship and train categories have **no val split** in the original data (too few sequences). We mapped frame-level splits to sequence-level by majority vote:
    - Car: seqs 001–024 train, 025–028 val, 029–038 test.
    - Plane: seqs 039–042 train, 043 val, 044 test.
    - Ship: 045 train, 047 test (no val).
    - Train: 046 train only (single sequence).
  - **Our split**: train 30 (10,902 frames) / val 5 (1,741 frames) / test 12 (3,561 frames).
  - **Implementation**: `VISODataset(root, split)`. Registered as `"VISO"` in both DataModule registries. Supports detection mode (per-frame) and video mode (clip-based for SAM2/tracking).
- **SV248S**: Single-object tracking dataset from Jilin-1 satellite video (0.92 m resolution, 25 FPS re-encoded via VFI from original 10 FPS). Non-commercial use only.
  - **248 sequences** from 6 videos, 4 classes: car (202), car-large (37), plane (6), ship (3). Variable crop sizes per sequence (e.g. 326×314). 156,621 total frames.
  - **Annotations**:
    - `.rect`: per-frame `x,y,w,h` (top-left + size), comma-separated floats, one line per frame.
    - `.poly`: per-frame tight polygon `x1,y1,x2,y2,...`, comma-separated floats. Provides mask-level annotation for small targets.
    - `.state`: per-frame integer flag — 0=NOR (visible), 1=INV (invisible/disappeared), 2=OCC (occluded). Rects are populated even for non-zero states.
    - `.abs`: JSON metadata with source video info, init_rect, init_poly, class_name, difficulty level.
    - `.attr`: 10 sequence attributes as CSV integers (STO, LTO, DS, IV, BCH, SM, ND, CO, BCL, IPR).
  - **Frames**: 1-indexed TIFF files (`000001.tiff`, ...) in per-sequence directories.
  - **Class mapping**: `car` → vehicle, `car-large` → large-vehicle, `plane` → airplane, `ship` → ship. Note: class names differ from the survey paper.
  - **No official split**. Our split: 80/10/10 stratified by category, `seed=42` → train 197 (124,403 frames) / val 26 (16,111 frames) / test 25 (16,107 frames).
  - **Implementation**: `SV248SDataset(root, split)`. Registered as `"SV248S"` in both DataModule registries. `load_mask()` for polygon-based segmentation masks.
  - **Design decisions**:
    - State=1 (invisible/disappeared) → empty annotations returned; State=2 (occluded) → bbox kept since the annotation is still spatially meaningful.
    - `track_ids=[1]` for all frames (SOT — single object per sequence).
    - Video IDs use `"01/000000"` format (`video_dir/seq_id`) to keep sequences globally unique across the 6 source videos.
    - `load_mask(video, frame_id)` renders the `.poly` polygon into a binary mask `(H, W)` uint8 (0/255) on demand — not loaded by default in detection/video mode.

- **SDM-Car**: Small and dim moving vehicle detection/tracking dataset from Luojia-3-01 satellite (0.75 m resolution). Published in IEEE GRSL 2024.
  - **99 sequences** (AVI video files), single class: car (vehicle). 1920×1080 resolution. 16,423 total frames across all splits.
  - **Annotations**: Per-video `-gt.csv` files (headerless CSV). 10 columns: `frame_id, target_id, x, y, w, h, -1, -1, -1, -1`. Coordinates are top-left xywh, absolute pixels. Frames are 0-indexed. Last 4 columns are reserved (always -1).
  - **Some bboxes slightly exceed image bounds** (e.g. x+w up to 2161 in 1920-wide frames). Clipped by `BaseVideoDataset` automatically.
  - **Official split**: train (64) / validation (15) / test (20).
  - **Our split**: train 64 (10,483 frames) / val 15 (2,409 frames) / test 20 (3,531 frames).
  - **Implementation**: `SDMCarDataset(root, split)`. Registered as `"SDM-Car"` in both DataModule registries. Reads frames directly from AVI via OpenCV `VideoCapture`. Single category `"car"` for all objects.

---

## Citation Hints

If you use this list, consider citing the original dataset papers. Key references:

```
SatSOT:      Zhao et al., IEEE TGRS 2022. DOI: 10.1109/TGRS.2022.3140809
SAT-MTB:     Li et al., IEEE TGRS 2023. DOI: 10.1109/TGRS.2023.3278075
VISO:        Yin et al., IEEE TGRS 2022. DOI: 10.1109/TGRS.2021.3130436
OOTB:        Chen et al., ISPRS 2024. DOI: 10.1016/j.isprsjprs.2024.03.013
IRSatVideo-LEO: Ying et al., IEEE TGRS 2025. arXiv: 2409.12448
SDM-Car:     Zhang et al., IEEE GRSL 2024. DOI: 10.1109/LGRS.2024.3493249
LMOD:        IEEE TGRS 2025. DOI: 10.1109/TGRS.2025.11142571
```
