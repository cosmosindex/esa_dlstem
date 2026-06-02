# Use-Case Datasets

Overview of the two non-spaceborne **use-case** datasets evaluated in this work:
an aerial **forest-fire detection** benchmark and an aerial **wild-animal
detection & tracking** benchmark. Style mirrors the spaceborne video datasets
table (`Formatting Instructions For NeurIPS 2026/tables/spaceborne_video_datasets_table.tex`).

| Dataset | Platform / Sensor | Modality | #Cls | #Seq | #Frames | #Labels | Small% | Task | Venue / Year |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| *Forest Fire Detection* | | | | | | | | | |
| RGBT-3M | UAV (RGB + Thermal IR) | RGB-T | 3 | 8 | 11,220 | 30,777 | 44.1 | Det | Remote Sensing 2025 |
| *Wild Animal Detection & Tracking* | | | | | | | | | |
| BIRDSAI | Fixed-wing UAV (LWIR) | TIR | 2 | 48 | 123,991 | 166,221 | 49.9 | Det+Trk | WACV 2020 |

**Column definitions**

- **#Cls** — number of annotated object categories.
- **#Seq / #Frames / #Labels** — video sequences, annotated frames, and bounding-box annotations on disk.
- **Small%** — fraction of boxes with area below 32×32 = 1024 px² (the project-wide small-object threshold).
- **Task** — Det (detection only) or Det+Trk (detection and tracking).

**Notes / provenance**

- **RGBT-3M** — *A UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion Model
  for Enhanced Forest Fire Detection*, Remote Sensing 2025
  ([MDPI 2072-4292/17/15/2593](https://www.mdpi.com/2072-4292/17/15/2593)).
  Time-synchronized UAV RGB + thermal-infrared capture; 3 classes
  {smoke, fire, person}. #Frames / #Labels / Small% from
  [`bbox_stats_report_fire.md`](bbox_stats/bbox_stats_report_fire.md)
  (11,220 frames over 8 source videos; 30,777 boxes; 44.1% < 32×32 px).
  Organized as 8 videos but used frame-wise for detection → Task = Det.
- **BIRDSAI** — *BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal
  Infrared Videos*, WACV 2020. Real subset only (32 TrainReal + 16 TestReal =
  48 sequences, 123,991 frames). #Labels / Small% from
  [`bbox_stats_report_wild_animals.md`](bbox_stats/bbox_stats_report_wild_animals.md)
  (166,221 boxes; 49.9% < 32×32 px). 2 top-level classes {animal, human}; the
  animal class carries 8 species sub-labels.
- **Dropped columns** vs. the spaceborne table: **FPS** (not stated in either
  source paper) and **#Attr** (neither dataset defines sequence-level attributes).
