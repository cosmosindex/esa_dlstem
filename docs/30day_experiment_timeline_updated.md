# 30-Day Experiment Timeline
# Spaceborne Video Benchmark — NeurIPS 2026 D&B Track

> **Deadline**: May 7, 2026 (Day 30)
> **GPU**: 2× NVIDIA RTX 5000 Ada Generation (32 GB each), CUDA 13.1
> **Strategy**: Zero-shot → Pretrained eval → Fine-tune, per track
> **Track order**: SOT (Days 1–8) → Detection (Days 9–18) → MOT (Days 19–26) → Buffer (Days 27–30)
> **Last updated**: April 14, 2026 — STAR 改为使用官方 checkpoint 直接 eval（不再 fine-tune）；时间线整体前移 2 天，SOT 结束日 Apr 15，Phase 2 始于 Apr 16

---

## Legend

| Symbol | Meaning |
|---|---|
| ✅ Done | Completed |
| 🟡 Zero-shot | Load official pretrained weights, run directly on test set — no training |
| 🔵 Pretrained eval | Load COCO/LaSOT/MOT17 pretrained weights, eval on our test set (OOD generalisation) |
| 🔴 Fine-tune | Fine-tune on our train split, eval on test — upper bound reference |
| 🟢 Analysis | Result compilation, writing, table generation |
| ⚠ Blocked | Cannot run — see note |

---

## Phase 1 — Single Object Tracking (SOT) · Days 1–8

**Datasets**: SatSOT (105 seqs), SV248S (248 seqs, 156K frames), SAT-MTB subset, OOTB (110 seqs, OBB), IRSatVideo-LEO (200 seqs, TIR)
**Primary metrics**: AUC, Precision (P), Normalised Precision (NP)

---

### SOT Model List (11 models — updated April 14)

| Model | Venue | 类别 | 备注 | Eval tier | 评估数据集 |
|---|---|---|---|---|---|
| SiamRPN++ | CVPR 2019 | Siamese 经典 | 全领域公共锚点，引用极高 | ✅ Pretrained | 全部 |
| OSTrack-256 | ECCV 2022 | One-stream transformer 基准 | 范式奠基 | ✅ Pretrained | 全部 |
| ODTrack | AAAI 2024 | One-stream transformer 最新 | token propagation | ✅ Pretrained | 全部 |
| LoRAT | ECCV 2024 | 大 ViT + LoRA | Scaling ViT 方向代表 | ✅ Pretrained | 全部 |
| LoRATv2 | NeurIPS 2025 | Causal temporal + LoRA | 时序建模 SOTA | ⚠ 待定 · 代码未公开 | — |
| DreamTrack | CVPR 2025 | 时序未来预测 | 卫星匀速运动契合 | ⚠ 待定 · 代码未公开 | — |
| DF | JSTARS 2022 | SV-specific CF | 唯一领域专用方法 | ⚠ 待定 · MATLAB 不兼容 Python | — |
| SAM 2 | Meta 2024 | FM zero-shot 基准 | 代际对比起点 | ✅ Zero-shot | 全部 |
| SAM 3 / SAM 3.1 | Meta Nov 2025 | FM zero-shot 最新 | 文本驱动，novelty 极高 | ✅ Zero-shot | 全部 |
| SAMURAI *(可选)* | arXiv 2024 | SAM 2 + 运动感知 | 读完论文后决定 | ✅ Zero-shot | 全部 |
| STAR | TGRS 2025 | SV-specific Transformer | 官方已发布 fine-tune 后的 checkpoint，直接使用。SV248S / OOTB 对 STAR 是 zero-shot（fully held out）；SatSOT test 与 train 同源但为不同 split。作为 **domain-adapted baseline**，论文中单独标注† | 🔵 Fine-tuned checkpoint† | 全部 (SatSOT / SV248S / OOTB) |

> **†** STAR 训练细节（引自论文 § IV-A-2）：150 epoch 预训练于 GOT-10k / TrackingNet / LaSOT / COCO（与其它 pretrained tracker 同源），随后 60 epoch fine-tune 于 SatSOT-train（947条，来自 SatMTB）。我们直接加载作者发布的 fine-tune 后 checkpoint，不做额外训练。因此：
> - **SatSOT-test**: 与训练集同数据集但不同 split。只要 train/test split 清晰分离即可使用，属标准 in-distribution eval。
> - **SV248S / OOTB**: 完全独立数据集，STAR 从未见过 → **zero-shot on those datasets**，仅在卫星视频域 fine-tuned。
>
> 论文中将 STAR 定位为 "domain-adapted baseline"，需明确标注训练数据来源(GOT-10k+TrackingNet+LaSOT+COCO→SatSOT-train)。不参与通用 pretrained eval 的横向排名。
>
> **⚠ 待定说明**：LoRATv2 / DreamTrack 代码一旦公开即可补测；DF 需 MATLAB 环境或联系作者获取 Python 版本，当前实验 pipeline 无法直接运行。

---

### Phase 1 Day-by-Day Schedule (revised April 14)

| Day | Date | GPU-0 | GPU-1 | 状态 |
|---|---|---|---|---|
| 1 | Apr 8 | ✅ SAM 2 — zero-shot, all SOT datasets | ✅ SAM 2 — zero-shot, all SOT datasets | ✅ 完成 |
| 2 | Apr 9 | ✅ SAMURAI — zero-shot, all SOT datasets | ✅ SAM 3 / 3.1 — zero-shot, text-prompted SOT, all SOT datasets | ✅ 完成 |
| 3 | Apr 10 | ✅ ODTrack — pretrained, SatSOT + SV248S + OOTB | ✅ OSTrack-256 — pretrained, SatSOT + SV248S + OOTB | ✅ 完成 |
| 4 | Apr 11 | ✅ SiamRPN++ — pretrained, all SOT datasets | ✅ LoRAT — pretrained, all SOT datasets | ✅ 完成 |
| 5 | Apr 12 | ✅ Rerun / SV248S format + metric sanity check | ✅ IRSatVideo-LEO TIR zero-shot SAM 2 / SAM 3 check | ✅ 完成 |
| 6 | Apr 13 | ✅ OSTrack-256 fine-tune — SatSOT + SV248S train split (multi-GPU) | ✅ OSTrack-256 fine-tune — distributed, same run | ✅ 完成 |
| **7** | **Apr 14 今天** | 🔵 **STAR eval — SatSOT-test + SV248S (下载作者 checkpoint，直接 eval，无训练)** | 🔵 **STAR eval — OOTB (parallel)** | 🔄 进行中 |
| 8 | Apr 15 | 🔵 STAR eval 收尾（若 SV248S 未跑完）+ 🟢 SOT result analysis — AUC/P/NP 表格，FM zero-shot vs pretrained gap，pretrained vs fine-tuned gap，STAR 作为 domain-adapted baseline 单独标注，写 SOT section draft | ← same | **SOT 实验完成** |

---

### Phase 1 风险点

**1. STAR eval 时间估计**
STAR 使用作者发布的 fine-tuned checkpoint，无需训练。Eval on SV248S（156K frames）预计 6–10 小时（作为最大的 SOT 测试集,占 GPU 时间大头）；SatSOT-test + OOTB 合计约 2–3 小时。Day 7–8 双卡并行 (GPU-0 跑 SV248S，GPU-1 跑 SatSOT-test + OOTB) 足以在 Apr 15 前完成全部评估。

**2. DF / LoRATv2 / DreamTrack 三个 blocked 模型**
目前标记为"待定"，不占用 Phase 1 时间：
- DF：如需纳入，需在 Phase 4 buffer（May 4）前联系作者获取 Python 版本，或通过 MATLAB Engine API 集成（额外 0.5 天工作量）
- LoRATv2 / DreamTrack：代码公开后可在 buffer 期补测，不影响主要 deadline

**3. STAR 结果定位**
STAR 在全部三个 SOT 测试集上评估，但训练数据构成不同于其它 tracker，不能与通用 pretrained eval 行横向比较：
- **SatSOT-test**: in-distribution eval（同数据集不同 split）。只要 split 隔离干净即有效。
- **SV248S / OOTB**: 对 STAR 是 zero-shot（从未见过这两个数据集），仅在卫星视频域 fine-tuned。这两个数据集上的结果相对公平，可作为 "domain-adapted but held-out" baseline 参考。
- 论文表格中 STAR 单独一行加 † 注释，明确标注训练数据来源 (GOT-10k+TrackingNet+LaSOT+COCO → SatSOT-train)。Reviewer 若误解，用"† domain-adapted baseline; see § IV-A"指引。

---

## Phase 2 — Object Detection · Days 9–18

**Datasets**: SAT-MTB det_hbb (142 seqs), SAT-MTB det_obb (106 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR)
**Primary metrics**: AP50, AP50:95

| Day | Date | GPU-0 | GPU-1 | Milestone |
|---|---|---|---|---|
| 9 | Apr 16 | 🟡 Grounding DINO — zero-shot, SAT-MTB HBB + VISO | 🟡 YOLO-World — zero-shot, SAT-MTB HBB + VISO | Zero-shot start |
| 10 | Apr 17 | 🟡 Florence-2 — zero-shot, all det datasets incl. SDM-Car | 🔵 Faster R-CNN — pretrained, all VNIR det datasets | |
| 11 | Apr 18 | 🔵 YOLOv8-s/l — pretrained, all VNIR det datasets | 🔵 FCOS — pretrained, all VNIR det datasets | Pretrained start |
| 12 | Apr 19 | 🔵 Deformable DETR — pretrained, all VNIR det datasets | 🔵 DINO-Det (Swin-L) — pretrained, all VNIR det datasets | |
| 13 | Apr 20 | 🔵 RT-DETR — pretrained, all VNIR det datasets | 🔵 LSKNet — pretrained, SAT-MTB + VISO | |
| 14 | Apr 21 | 🔵 Oriented R-CNN — pretrained, SAT-MTB det_obb + OOTB | 🔵 H2RBox-v2 — pretrained, SAT-MTB det_obb + OOTB | |
| 15 | Apr 22 | 🔵 DNANet / ISNet — pretrained, IRSatVideo-LEO only | 🔵 UIU-Net — pretrained, IRSatVideo-LEO only | |
| 16 | Apr 23 | 🟢 Detection SOTA analysis — rank by AP50, select best model for MOT detector | 🔴 DINO-Det fine-tune — on SAT-MTB train split | |
| 17 | Apr 24 | 🔴 YOLOv8 fine-tune — upper bound for fast detector | 🔴 DINO-Det fine-tune — continued + eval test split | Fine-tune start |
| 18 | Apr 25 | 🟢 Detection result analysis + MOT detector prep — compile AP tables, confirm SOTA detector for MOT, generate detection CSV/JSON files for all MOT datasets, write detection section draft | ← same | **Det done · MOT det ready** |

---

## Phase 3 — Multi-Object Tracking (MOT) · Days 19–26

**Datasets**: AIR-MOT (69 seqs), SAT-MTB mot (237 seqs), VISO (47 seqs), SDM-Car (99 seqs), IRSatVideo-LEO (200 seqs, TIR), LMOD (8 seqs — qualitative only)
**Primary metrics**: HOTA, MOTA, IDF1
**Fixed detector**: DINO-Det pretrained (confirmed on Day 18) — all TbD trackers use the same detection input

| Day | Date | GPU-0 | GPU-1 | Milestone |
|---|---|---|---|---|
| 19 | Apr 26 | 🟡 MASA — zero-shot, all MOT datasets | 🟡 SAM 3 (MOT mode) — zero-shot, AIR-MOT + SAT-MTB | Zero-shot start |
| 20 | Apr 27 | 🟡 UNINEXT — zero-shot MOT, all datasets | 🔵 SORT + DeepSORT — pretrained, all MOT datasets (fast, batch together) | Pretrained start |
| 21 | Apr 28 | 🔵 ByteTrack — pretrained, all MOT datasets | 🔵 OC-SORT + BoT-SORT — pretrained, all MOT datasets | |
| 22 | Apr 29 | 🔵 StrongSORT + Deep OC-SORT — pretrained, all MOT datasets | 🔵 MOTIP — pretrained, SAT-MTB + AIR-MOT | |
| 23 | Apr 30 | 🔵 TrackTrack + CenterTrack — pretrained, all MOT datasets | 🔴 TGraM / MO-TAMA fine-tune — on AIR-MOT train split | Fine-tune start |
| 24 | May 1 | 🔴 ByteTrack fine-tune — on SAT-MTB mot train split | 🔴 TGraM fine-tune — continued + eval test split | |
| 25 | May 2 | 🟢 MOT result analysis — HOTA/MOTA/IDF1 tables, TbD vs E2E comparison, LMOD qualitative figs, write MOT section draft | ← same | **MOT done** |
| 26 | May 3 | 🟢 Buffer — MOT rerun 任何异常；Paper draft 整合 Phase 1–3 section | ← same | → Paper assembly ready |

---

## Phase 4 — Buffer + Paper Assembly · Days 27–30

| Day | Date | Task |
|---|---|---|
| 27 | May 4 | Rerun any failed or suspicious results — missing metrics, NaN values, model crashes, TIR zero-shot SAM 3 recheck；如 DF Python 版本已获取，可在此补测 |
| 28 | May 5 | Main result tables + figures — LaTeX tables, per-dataset AP/AUC/HOTA, modality gap figures, FM zero-shot vs pretrained bars |
| 29 | May 6 | Final paper assembly — abstract, intro, benchmark design section, conclusion, references, supplementary appendix |
| 30 | May 7 | Final proofread + submit — 最后一轮 review，确认所有 figure 清晰度和 caption 完整性 |

---

## Key Notes & Risk Points

### 1. SV248S is the biggest bottleneck
156K frames — even on RTX 5000 a single SOT model can take 8–12 hours.
**Action**: On Day 1, first run the full pipeline on SatSOT (27K frames) to verify format and metrics are correct. Only then add SV248S to the queue. Do not run SV248S blind on Day 1.

### 2. Day 18 detector choice determines MOT quality
The MOT evaluation depends entirely on the detection files generated from the chosen detector.
**Action**: Fix DINO-Det (pretrained) as the default MOT detector regardless of fine-tuned results. Write clearly in your paper: *"All TbD trackers are evaluated with pretrained DINO-Det as the shared detector."* This keeps MOT results comparable and reproducible.

### 3. End-to-end MOT models need separate treatment
MOTIP and TGraM / MO-TAMA are end-to-end joint detection + tracking models — they do not use the shared detector.
**Action**: Report them in a separate sub-table labelled "End-to-End (E2E)" in the MOT results section, clearly distinguished from Tracking-by-Detection (TbD) methods. Do not compare E2E AP directly with TbD AP.

### 4. STAR 表格标注规范
STAR 在论文 SOT 结果表中需单独处理：
- 放在表格最后一行，用横线与 pretrained eval 行隔开
- 标注 "† domain-adapted baseline: pretrained GOT-10k+TrackingNet+LaSOT+COCO → fine-tuned SatSOT-train"
- 填写全部三个数据集 (SatSOT / SV248S / OOTB) 的数字
- 明确说明: SV248S / OOTB 对 STAR 是 held-out zero-shot（仅域适应），SatSOT-test 与训练集同源但 split 隔离
- 不参与整体 pretrained eval 排名，仅作为 "with satellite-video adaptation" 参考

### 5. Segmentation track — optional fast path
Segmentation is not in the 30-day plan. If you want minimal seg coverage without extra GPU time:
- SAM 2, SAM 3 are already running on Days 1–2 — grab their segmentation mask outputs at the same time (zero marginal cost).
- Report J&F score on SAT-MTB-SOS only, no fine-tune, one paragraph in the paper.
- Do not start a full seg track — it will break the timeline.

### 6. Daily backup
At the end of every day, `rsync` all result files to `compute01.cosmos-index.com`.
```bash
rsync -avz ./results/ ziwen@compute01.cosmos-index.com:~/benchmark_results/
```
One corrupted local disk should not cost you a week of experiments.

### 7. SAM 3 paper citation status
SAM 3 (arXiv Nov 2025) is currently under double-blind review at ICLR 2026. Cite as:
```
Carion et al., "SAM 3: Segment Anything with Concepts", arXiv:2511.16719, 2025.
```
Check ICLR 2026 acceptance status before submission (~May 2026) and update citation format if accepted.

### 8. TIR vs VNIR — always report separately
IRSatVideo-LEO is a fundamentally different modality. Never mix TIR and VNIR numbers in the same table row.
Use separate columns or sub-tables labelled **VNIR** and **TIR** consistently throughout the paper.

### 9. Compute estimate (revised April 14)

| Track | Models | Est. GPU-hours | Notes |
|---|---|---|---|
| SOT pretrained + zero-shot | 7 (done) | ~55 h | 已完成 |
| SOT STAR eval (pretrained checkpoint, no training) | 1 | ~10 h | Day 7–8，SV248S eval 占大头；无 fine-tune 省 10h |
| Detection | 14 | ~60 h | IR models faster (fewer seqs) |
| MOT | 13 | ~65 h | Depends on det file I/O speed |
| Fine-tune (Det + MOT) | ~5 | ~130 h | SOT fine-tune already counted above |
| **Total remaining** | | **~320 GPU-hours** | ~6.7 days on 2 GPUs running 24/7 — 时间线前移 2 天后 buffer 更宽松 |
