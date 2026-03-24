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
| ❓ **AIR-MOT** | Jilin-1 | VNIR | 10 | MOT | TGRS 2022 | [IEEE](https://ieeexplore.ieee.org/document/9715124) · [GitHub](https://github.com/HeQibin/TGraM) |
| ❓ **Jilin-189** | Jilin-1 | VNIR | 102 videos (647 clips) | VSR | TGRS 2022 | [GitHub](https://github.com/XY-boy/MSTT-STVSR) |
| ✅ **SAT-MTB** | Jilin-1 | VNIR | 249 | Det · SOT · MOT · Seg | TGRS 2023 | [Zenodo](https://zenodo.org/records/15253996) |
| ❓ **MWIRSTD** | MWIR cooled imager | MIR | 14 | IR small target Det | arXiv 2024 | [arXiv](https://arxiv.org/abs/2406.08063) |
| ✅ **VISO (SatVideoDT)** | Jilin-1 | VNIR | 100 | Det · MOT | TGRS 2022 | [GitHub](https://github.com/QingyongHu/VISO) · [Project](https://satvideodt.github.io/) |
| 🔄 **IRSatVideo-LEO** | Landsat 8/9 (semi-simulated) | Infrared (TIR) | 200 | IR MIRST Det | **TGRS 2025** | [arXiv](https://arxiv.org/abs/2409.12448) · [GitHub](https://github.com/XinyiYing/RFR) |
| ✅ **SV248S** | Jilin-1 | VNIR | 248 | SOT | — | [GitHub](https://github.com/xdai-dlgvv/SV248S) |
| ✅ **Cross-Videos OT** | Jilin-1 / Luojia-03 | VNIR | 26 | Cross-satellite SOT | Kaggle 2023 | [Kaggle](https://www.kaggle.com/datasets/lizilong1993/cross-videos-object-tracking) |
| ✅ **OOTB** | Multiple (JL-1, SkySat, ISS) | VNIR | 110 | SOT (oriented bbox) | ISPRS 2024 | [GitHub](https://github.com/YZCU/OOTB) |
| ✅ **LMOD** | Jilin-1 | VNIR | — | Large-scale MOD | TGRS 2025 | [GitHub](https://github.com/RS-Devotee/LMOD) · [IEEE](https://ieeexplore.ieee.org/document/11142571) |
| 🆕 **SAT-MTB-SOS** | Jilin-1 | VNIR | 113 | Single-object Seg (VOS) | CVPR 2024 W | [IEEE](https://ieeexplore.ieee.org/document/10604055/) |
| 🆕 **SAT-MTB-VSR** | Jilin-1 | VNIR | 431 clips | Video Super-Resolution | JSTARS 2023 | [GitHub](https://github.com/Alioth2000/RASVSR) |
| 🆕 **SDM-Car** | Luojia 3-01 | VNIR | 99 | Small/dim vehicle Det · MOT | GRSL 2024 | [GitHub](https://github.com/TanedaM/SDM-Car) · [IEEE](https://ieeexplore.ieee.org/document/10746500/) |
| 🆕 **NUDT-MIRSDT-HiNo** | Semi-simulated (aerial IR bg) | Infrared (MWIR/SWIR) | 120 seq | IR MIRST Det (high noise) | Preprint 2025 (→TPAMI) | [arXiv](https://arxiv.org/abs/2506.12766) |

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
