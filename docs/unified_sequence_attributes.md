# Unified Sequence Attributes Across SatSOT, SV248S, and OOTB

This document consolidates the sequence attribute definitions from three spaceborne video single object tracking (SOT) benchmarks — **SatSOT**, **SV248S**, and **OOTB** — and proposes a unified attribute taxonomy for cross-dataset evaluation.

---

## 1. Original Attribute Definitions (Per Dataset)

### 1.1 SatSOT — 11 Sequence Attributes

SatSOT defines 11 attributes to characterize tracking challenges in each sequence:

| Attribute | Full Name | Definition |
|---|---|---|
| **BC** | Background Clutter | The background has similar appearance as the target |
| **IV** | Illumination Variation | The illumination of the target region changes significantly |
| **LQ** | Low Quality | The image is low quality and the target is difficult to be distinguished |
| **ROT** | Rotation | The target rotates in the video |
| **POC** | Partial Occlusion | The target is partially occluded in the video |
| **FOC** | Full Occlusion | The target is fully occluded in the video |
| **TO** | Tiny Object | At least one ground truth bounding box has less than 25 pixels |
| **SOB** | Similar Object | There are objects of similar shape or same type around the target |
| **BJT** | Background Jitter | Background jitter caused by the shaking of satellite camera |
| **ARC** | Aspect Ratio Change | The ratio of the bounding-box aspect ratio of the first and the current frame is outside the range [0.5, 2] |
| **DEF** | Deformation | Non-rigid object deformation |

---

### 1.2 SV248S — Basic Properties, Frame State Flags, and Sequence Attributes

SV248S distinguishes three categories: Basic Properties, Frame State Flags, and Sequence Attributes.

| Category | Abbr. | Full Name | Definition | Dependence | Auto |
|----------|-------|-----------|------------|------------|------|
| **Basic Properties** | CP | Center Point | Represents the center point of the object bounding box. | BBox | Y |
| | MV | Movement Velocity | Calculates the average velocity considering both speed value and direction in adjacent five frames (unit: pps). | CP | Y |
| | OS | Object Size | The largest value selected from the height and width of the object bounding box. | BBox | Y |
| **Frame State Flags** | INV | Invisible | The object disappeared without any occluder or is too similar to its surroundings. | M | N |
| | NOR | Normal Visible | The object is visible and found easily. | M | N |
| | OCC | Occlusion | The object is in the shadow of the building or behind something. | M | N |
| **Sequence Attributes** | BCH | Background Change | The background of the tracked object has noticeable changes in color or texture. | M | N |
| | BCL | Background Cluster | There are at least 10 frames that contain the INV flag. | INV | Y |
| | CO | Continuous Occlusion | STO or LTO occur twice or more times in a sequence. | STO, LTO | Y |
| | DS | Dense Similarity | One or more similar objects exist around the tracked object within 2.5× OS range. | OS | N |
| | IPR | In-Plane Rotation | The object has an in-plane rotation at an angle ≥ 30°. | MV | Y |
| | IV | Illumination Variation | The object has noticeable changes in brightness or color. | M | N |
| | STO | Short-Term Occlusion | The sequence has ≤ 50 consecutive frames with OCC flags. | OCC | Y |
| | LTO | Long-Term Occlusion | The sequence has > 50 consecutive frames with OCC flags. | OCC | Y |
| | ND | Natural Disturbance | The object's appearance is influenced by smog, sandy weather, or blocked by clouds. | M | N |
| | SM | Slow Motion | The moving speed of the tracked object is < 2.2 pps. | MV | Y |

**Notes:**
- **Abbr.**: abbreviation
- **M**: manually inspected
- **Semi-Auto**: manually inspected with machine visualization
- **Auto**: automatically estimated by machine calculation (Y = Yes, N = No)
- **NOR**: normal visible state; **INV**: invisible state; **OCC**: occluded state
- **pps**: pixels per second

#### Dependence Column — What Each Attribute/Flag Depends On

| Symbol | Meaning | Examples |
|--------|---------|----------|
| **BBox** | Depends on the bounding box | CP and OS are computed from the bbox |
| **CP** | Depends on Center Point | MV is calculated from CP across adjacent frames |
| **M** | Manually inspected (no automatic dependency) | BCH, IV, ND, DS, NOR, INV, OCC |
| **MV** | Depends on Movement Velocity | SM (speed threshold) and IPR (rotation angle) |
| **OS** | Depends on Object Size | DS uses "2.5× OS" as the search range |
| **INV** | Depends on the Invisible frame flag | BCL requires accumulated INV flags (≥ 10 frames) |
| **OCC** | Depends on the Occlusion frame flag | STO (≤ 50 consecutive OCC frames) and LTO (> 50 consecutive OCC frames) |
| **STO, LTO** | Depends on other derived attributes | CO requires STO or LTO to occur 2+ times |

#### Auto Column — Whether the Attribute Can Be Auto-Derived

| Symbol | Meaning |
|--------|---------|
| **Y** (Yes) | Can be automatically computed from existing bbox, CP, or frame flags |
| **N** (No) | Requires manual annotation; cannot be auto-derived |
| **Semi-Auto** | Manually inspected with machine visualization assistance (mentioned in the paper's footnote but not explicitly used as a value in Table 5) |

#### Practical Implications

- **Auto = Y attributes** (BCL, CO, IPR, STO, LTO, SM): Can be regenerated by scripts from `.occ` flags or trajectory data — useful for cross-dataset annotation consistency checks.
- **Auto = N attributes** (BCH, DS, IV, ND): Require manual labeling — harder to port across datasets when unifying evaluation protocols.

---

### 1.3 OOTB — 12 Fine-Grained Sequence Attributes

Each sequence in OOTB is labeled with 12 fine-grained attributes (from the OOTB paper, Chen et al., ISPRS 2024):

| Attribute | Description |
|-----------|-------------|
| **DEF** | Deformation – non-rigid deformation of an object. |
| **IPR** | In-Plane Rotation – the object rotates in the image plane. |
| **PO** | Partial Occlusion – the object appears partially occluded in satellite video. |
| **FO** | Full Occlusion – the object appears fully occluded in satellite video. |
| **IV** | Illumination Variation – the illumination around the object is significantly changed. |
| **MB** | Motion Blur – the object region is blurred due to the motion of the object or satellite platform. |
| **BC** | Background Clutters – the background near the object has a similar texture or color as the object. |
| **OON** | Out-of-Normal – the aspect ratio of the bounding box is outside the range [0.3, 3] in a video. |
| **SA** | Similar Appearance – there are objects with similar appearance near the tracked object. |
| **LT** | Less Textures – the texture information of the target is less leading to extreme difficulty to discriminate. |
| **IM** | Isotropic Motion – there are objects with similar moving in magnitude and direction near the tracked object. |
| **AM** | Anisotropic Motion – there are objects with similar magnitude of motion but in opposite directions near the tracked object. |

---

## 2. Unified Attribute Table (Merged Across All Three Datasets)

The merging principle is: **preserve every attribute** — semantically equivalent attributes across datasets are merged into a unified category, and attributes unique to a single dataset are retained as standalone entries.

| # | Unified Abbr. | Full Name | Definition | SatSOT | SV248S | OOTB |
|---|---|---|---|---|---|---|
| 1 | **BC** | Background Clutter | Background has similar appearance (texture/color) to the target | BC | — | BC |
| 2 | **BCH** | Background Change | Background has noticeable changes in color or texture along the sequence | — | BCH | — |
| 3 | **IV** | Illumination Variation | Illumination around the target changes significantly | IV | IV | IV |
| 4 | **LQ** | Low Quality | Image quality is low; target is hard to distinguish | LQ | — | — |
| 5 | **LT** | Less Textures | Target has poor texture information, causing discrimination difficulty | — | — | LT |
| 6 | **MB** | Motion Blur | Target region is blurred due to object or platform motion | — | — | MB |
| 7 | **ND** | Natural Disturbance | Target appearance affected by smog, sand, or clouds | — | ND | — |
| 8 | **ROT** | Rotation (in-plane) | Target rotates in the image plane (≥30° for SV248S) | ROT | IPR | IPR |
| 9 | **POC** | Partial Occlusion | Target is partially occluded | POC | STO¹ | PO |
| 10 | **FOC** | Full Occlusion | Target is fully occluded | FOC | LTO¹ | FO |
| 11 | **CO** | Continuous Occlusion | Occlusion events occur twice or more in a sequence | — | CO | — |
| 12 | **INV** | Invisible | Target disappears without occluder (too similar to surroundings) — BCL in SV248S | — | BCL² | — |
| 13 | **TO** | Tiny Object | At least one GT bbox has fewer than 25 pixels | TO | — | — |
| 14 | **SOB** | Similar Object | Nearby objects of similar shape/type/appearance | SOB | DS | SA |
| 15 | **IM** | Isotropic Motion | Nearby objects move with similar magnitude and direction | — | — | IM |
| 16 | **AM** | Anisotropic Motion | Nearby objects move with similar magnitude but opposite direction | — | — | AM |
| 17 | **SM** | Slow Motion | Target moves slowly (<2.2 pps in SV248S) | — | SM | — |
| 18 | **BJT** | Background Jitter | Background jitter from satellite camera shaking | BJT | — | — |
| 19 | **ARC** | Aspect Ratio Change | Aspect ratio of bbox changes beyond a threshold (SatSOT: [0.5, 2]; OOTB OON: [0.3, 3]) | ARC | — | OON |
| 20 | **DEF** | Deformation | Non-rigid object deformation | DEF | — | DEF |

¹ SV248S 的 STO/LTO 是基于连续 OCC 帧数 (≤50 vs >50) 的时长定义，与 SatSOT/OOTB 基于“部分/完全”的空间遮挡定义**不完全等价**。详见下方注意事项。
² SV248S 的 BCL 是基于 ≥10 帧 INV 累计得出的，语义上接近“target 与背景无法区分”。

Total: **20 unified attributes** (after separating BC/BCH and POC/FOC/CO as independent categories).

---

## 3. Key Merging Decisions & Caveats

These points should be explicitly stated in the paper to ensure transparency.

### 3.1 Rotation 的合并

SatSOT 的 `ROT` 只说 “rotates in the video”，没规定平面内 / 外；SV248S 和 OOTB 明确是 in-plane。统一使用 **ROT (in-plane)** 并在论文里注明 SatSOT 的 ROT 被视作 in-plane 处理。如果想更保守，可以单独保留 `ROT`（泛指）和 `IPR`（平面内）两个类别。

### 3.2 Occlusion 的合并 ⚠️ 最需要注意

三个数据集的遮挡定义维度不同：

- **SatSOT / OOTB**：**空间维度**（部分 vs 完全）→ POC/PO, FOC/FO
- **SV248S**：**时间维度**（短时 vs 长时连续 OCC 帧）→ STO, LTO

严格来说 POC ≠ STO，FOC ≠ LTO。建议两种处理方式：

1. 按上表做 “近似映射”，并在论文里明确声明这是一个 approximation；
2. 保留四个独立标签（POC、FOC、STO、LTO），让模型在各自数据集上按原标签评估，只在 “overall occlusion” 这个大类下聚合。

### 3.3 Similar Object 合并

SatSOT 的 `SOB`、SV248S 的 `DS`（2.5× OS 范围内）、OOTB 的 `SA` 语义高度一致，合并为 **SOB** 合理。DS 的距离阈值是定量的，其他两个是定性的 —— 合并时要注明这一点。

### 3.4 ARC vs OON

SatSOT 的 ARC 阈值 [0.5, 2]，OOTB 的 OON 阈值 [0.3, 3]，本质都是 aspect ratio 的显著变化。合并为 **ARC**，但**阈值差异会导致 per-dataset 的标注率不同**，这是数据集特性不是 bug。

### 3.5 SV248S 的 BCL

BCL 在 SV248S 里是 “sequence 含 ≥10 帧 INV（invisible）” 的统计，语义上是 “target 长时间与背景难以区分”。它既不完全等同于 BC（背景杂乱），也不等同于 FOC（完全遮挡）。建议**单独保留为 INV**（或 BCL），不强行合并。

---

## 4. Practical Evaluation Strategy

做 attribute-based evaluation 时，推荐**两级策略**：

- **Tier 1 (Overlap subset, 用于 cross-dataset 对比)**：只使用三个数据集都有或能安全映射的 attribute → IV, ROT, POC, FOC, SOB, DEF 等约 6–8 个。这部分可以做跨数据集的 attribute-wise 比较。
- **Tier 2 (Dataset-specific attributes)**：每个数据集独有的 attribute 在自己的 table 里单独报告（如 SatSOT 的 TO/BJT，SV248S 的 SM/ND/BCH/CO，OOTB 的 MB/LT/IM/AM）。这样既不丢信息，又不会做不公平的跨数据集比较。

论文里可以放一个类似第 2 节的 mapping table 作为 appendix，再在正文里放一个精简的 Tier 1 vs Tier 2 示意图。这对 reviewer 来说是很加分的 transparency。
