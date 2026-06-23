# BIRDSAI 跟踪任务对比 — 6 跟踪器 × 3 检测器

> **范式**: TBD(tracking-by-detection)——把前面 3 个检测器的缓存检测喂给在线跟踪器
> **数据集**: BIRDSAI test · **GT = `annotations_sam3`**(与检测表同一套 GT,可直接对照)
> **协议**: 同一批 **11/16 视频**(sweep 覆盖集,18 个 run 完全相同)· 12,236 帧 · IoU 0.5 · fine 5-class
> **跟踪器**: SORT / OC-SORT / ByteTrack / BoT-SORT / BoT-SORT+ReID / TrackTrack
> **指标**: MOTA = 1−(FP+FN+IDsw)/GT · IDsw = ID 切换数 · F1/P/R = 检测层

---

## 表 1 · MOTA（行 = 跟踪器,列 = 检测器主干）

| 跟踪器 | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | −3.579 | **+0.103** | −1.311 |
| OC-SORT | −0.888 | +0.243 | **+0.263** |
| ByteTrack | −0.299 | **+0.190** | +0.030 |
| BoT-SORT | −0.933 | **+0.186** | −0.001 |
| BoT-SORT+ReID | −1.055 | **+0.221** | +0.154 |
| TrackTrack | −0.510 | +0.256 | **+0.299** |

> 加粗 = 该跟踪器在 3 个主干里的最优。**最佳组合: DINOv3 + TrackTrack = +0.299**。

---

## 表 2 · F1（检测层,IoU 0.5）

| 跟踪器 | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 0.232 | **0.488** | 0.364 |
| OC-SORT | 0.391 | 0.496 | **0.557** |
| ByteTrack | **0.463** | 0.433 | 0.391 |
| BoT-SORT | 0.389 | **0.476** | 0.462 |
| BoT-SORT+ReID | 0.378 | 0.505 | **0.543** |
| TrackTrack | 0.436 | 0.500 | **0.557** |

---

## 表 3 · IDsw（ID 切换数,越低越好）

| 跟踪器 | FasterRCNN | YOLO11l | DINOv3 |
|---|---:|---:|---:|
| SORT | 7142 | **1554** | 8283 |
| OC-SORT | 4997 | **870** | 2188 |
| ByteTrack | 4797 | **2823** | 6745 |
| BoT-SORT | 7151 | **4057** | 9265 |
| BoT-SORT+ReID | 5856 | **2451** | 5120 |
| TrackTrack | 2703 | **580** | **749** |

> TrackTrack 的 IDsw 远低于其它跟踪器(YOLO 580 / DINOv3 749),关联质量最强。

---

## 表 4 · 每类 F1 — FasterRCNN 主干

| 跟踪器 | human | elephant | giraffe | lion | unknown |
|---|---:|---:|---:|---:|---:|
| SORT | 0.201 | 0.431 | 0.064 | 0.000 | 0.012 |
| OC-SORT | 0.183 | 0.611 | 0.051 | 0.000 | 0.002 |
| ByteTrack | 0.122 | 0.671 | 0.046 | 0.000 | 0.001 |
| BoT-SORT | 0.174 | 0.613 | 0.053 | 0.000 | 0.002 |
| BoT-SORT+ReID | 0.206 | 0.597 | 0.052 | 0.000 | 0.002 |
| TrackTrack | 0.229 | 0.648 | 0.047 | 0.000 | 0.001 |

> 跟踪后每类仍是 elephant 独大,giraffe/lion/unknown 依旧 ≈0——**瓶颈在检测,不在关联**(换跟踪器救不回小目标)。

---

## 结论（PPT 讲点）

1. **跟踪器换不动小目标**。无论哪个跟踪器,长尾(giraffe/lion/unknown)仍 ≈0;TBD 范式的天花板由前端检测决定,关联算法只能在已检到的目标上做文章。
2. **检测器的工作点决定 MOTA 正负**。FasterRCNN 高召回低精度 → FP 多 → MOTA 几乎全负;YOLO/DINOv3 高精度 → FP 少 → MOTA 普遍为正。**MOTA 对误检极敏感**。
3. **TrackTrack / OC-SORT 关联最强**。两者 IDsw 最低(TrackTrack: YOLO 580 / DINOv3 749),MOTA 与 F1 也领先;**最佳组合 = DINOv3 + TrackTrack(MOTA +0.299, F1 0.557)**。
4. **ReID 帮助有限**。热成像上 ReID(MOT17 权重,域不匹配)对 BoT-SORT 仅小幅改善 IDsw,不及无外观的 TrackTrack/OC-SORT 的运动建模。

---

*生成脚本: `evaluation/_birdsai_tracking_compare.py`（离线解析 mot_format 轨迹,按 `class = track_id // 1e6` 还原类别,对 annotations_sam3 重打分,无 GPU/重跟踪）。*
*CAVEAT: sweep 仅覆盖 11/16 测试视频(缺 5 个纯 human 视频);18 个 run 同一视频集,互相可比;与检测表(16 视频)对照时注意此差异。IDF1 在本流水线 == 检测 F1,故未单列。*
