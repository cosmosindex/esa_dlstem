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
| ✅ **SAT-MTB** | Jilin-1 | VNIR | 249 | Det · SOT · MOT · Seg | TGRS 2023 | [Zenodo](https://zenodo.org/records/15253996) |
| ✅ **VISO (SatVideoDT)** | Jilin-1 | VNIR | 100 | Det · MOT | TGRS 2022 | [GitHub](https://github.com/QingyongHu/VISO) · [Project](https://satvideodt.github.io/) |
| 🔄 **IRSatVideo-LEO** | Landsat 8/9 (semi-simulated) | Infrared (TIR) | 200 | IR MIRST Det · SOT · MOT · Seg | **TGRS 2025** | [arXiv](https://arxiv.org/abs/2409.12448) · [GitHub](https://github.com/XinyiYing/RFR) |
| ✅ **SV248S** | Jilin-1 | VNIR | 248 | SOT | — | [GitHub](https://github.com/xdai-dlgvv/SV248S) |
| ✅ **OOTB** | Multiple (JL-1, SkySat, ISS) | VNIR | 110 | SOT (oriented bbox) | ISPRS 2024 | [GitHub](https://github.com/YZCU/OOTB) |
| ✅ **LMOD** | Jilin-1 | VNIR | 8 | Large-scale MOD | TGRS 2025 | [GitHub](https://github.com/RS-Devotee/LMOD) · [IEEE](https://ieeexplore.ieee.org/document/11142571) |
| 🆕 **SAT-MTB-SOS** | Jilin-1 | VNIR | 113 | Single-object Seg (VOS) | CVPR 2024 W | [IEEE](https://ieeexplore.ieee.org/document/10604055/) |
| 🆕 **SDM-Car** | Luojia 3-01 | VNIR | 99 | Small/dim vehicle Det · MOT | GRSL 2024 | [GitHub](https://github.com/TanedaM/SDM-Car) · [IEEE](https://ieeexplore.ieee.org/document/10746500/) |

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
- **SatSOT**: SOT dataset from Jilin-1 satellite video.
  - **105 sequences**, variable resolution (e.g. 335×499), 27,664 total frames. 4 categories: car (65), plane (9), ship (5), train (26).
  - **Annotations**: `groundtruth.txt` per sequence — one `x,y,w,h` line per frame (top-left + size). Frames with `none` = target absent/occluded.
  - **No official split**. Our split: 80/10/10 stratified by category, `seed=42` → train 83 (22,718 frames) / val 11 (2,478 frames) / test 11 (2,468 frames).
  - **JSON metadata**: `SatSOT.json` with per-sequence attributes (BC, DEF, ARC, ROT, etc.) and `gt_rect` in xywh.
- **NUDT-MIRSDT-HiNo**: Extension of NUDT-MIRSDT with higher noise levels. Currently preprint (arXiv 2506.12766, v4 Jan 2026), submitted to IEEE TPAMI — not yet formally accepted. Semi-synthetic sequences; **not real satellite video**.
- **SAT-MTB-SOS**: Subset of SAT-MTB, re-annotated with pixel-level masks for VOS. No standalone public download link found; dataset may be requested from authors or obtained via the parent SAT-MTB on [Zenodo](https://zenodo.org/records/15253996).
- **VISO / SatVideoDT**: Non-commercial use only. Download via [Google Drive](https://github.com/QingyongHu/VISO) or Baidu Netdisk (code: VISO).
- **SV248S**: Non-commercial use only.

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
