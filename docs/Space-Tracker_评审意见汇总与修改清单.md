# Space-Tracker 评审意见汇总与修改清单
**目标：NeurIPS 2026 → WACV 2027 Datasets & Benchmarks Track 转投**

评审来源：
| 来源 | 评分 | 置信度 | 核心立场 |
|---|---|---|---|
| NeurIPS Reviewer tsgG | 3（Borderline Reject） | 3 | 工程扎实、分析深入，但三个关键问题需大改 |
| NeurIPS Reviewer YStx | 2（Reject） | 3 | 贡献偏增量，缺领域特异性分析与可靠性讨论 |
| NeurIPS Reviewer eKyB | 2（Reject） | 4 | 一致性问题严重，数据集贡献定位不清，有伦理关切 |
| CS Paper 外部评审 | 2（Reject） | 4 | 最详细：模态声明、评测独立性、协议公平性均有实质问题 |

---

## 一、评审公认的优点（转投时应保留并强化）

1. **选题重要且切中痛点**：星载视频小目标跟踪领域数据集碎片化是真实问题，统一格式与评测接口能显著降低社区的工程成本（四位评审均认可）。
2. **亚 8 像素极小目标视角有价值**：Figure 1 用 SR 曲线实证推导 √area < 8px 阈值，比常规 32×32 小目标定义更严格、更贴合领域（tsgG、CS 评审）。
3. **工程整合工作量大且规范**：将 HBB/OBB/polygon、XML/JSON 等异构标注统一为 COCO 风格；18 个属性从不一致的源定义中对齐（tsgG）。
4. **SOT 协议与指标定义详细**：首帧初始化、clip 状态传递、原生分辨率、置信度过滤、SR/NPR/PR/P@5 公式清晰（CS 评审）。
5. **实证发现非平凡**：SORT 在极小车辆上优于 ByteTrack（HOTA 0.335 vs 0.017）、LoRAT 在长遮挡下比 SAM 系更稳健、SAM 3 全集 SR 0.453 但 tiny 子集降至 0.334 等结论对后续工作有启发（tsgG、CS 评审）。
6. **MOT 共享检测缓存的设计思路正确**：用缓存检测隔离关联器行为，作为 association stress test 有潜在价值（CS 评审）。
7. **属性表透明度好**：Table 7 明确区分共享属性与数据集特有属性，并承认部分定义不等价，好于默默混用（CS 评审）。
8. **定性示例（Figure 2）直观**：让极小目标与背景难以区分的视觉难度具体可感（CS 评审）。

---

## 二、必须修改的问题（按优先级排序）

### P0：致命的一致性与事实性错误（转投前无条件修复）

- [ ] **数据集数量矛盾**：摘要说 nine、正文说 eight、结论出现占位符 "XX disparate datasets"（YStx、eKyB、CS 评审）。给出唯一权威数字，并在 Table 1 逐一列出。
- [ ] **属性数量矛盾**：摘要 18 个 vs 结论 12 个 fine-grained attributes（eKyB、CS 评审）。明确定义"什么算一个属性"，说明 18 = 共享属性 + 特有属性 + 遮挡子类型的构成。
- [ ] **"multi-sensory / visible-thermal" 声明无数据支撑**：Table 1 全部标注为 VNIR，没有任何热红外序列、对齐协议或跨模态实验（CS 评审第 1 条，最重的一击）。二选一：删除所有 thermal/multimodal 表述，或补充热红外数据源、序列/帧数、对齐方式与模态特异性结果。若保留 IRSatVideo-LEO 相关内容，需在正文中把模态构成写清楚。
- [ ] **标注实例总数存疑**："over 1 million annotated instances" 与 Table 1 中 MOT 源数据集之和不符（CS 评审第 2 条）。给出清洗后的精确统计。
- [ ] **拼写与排版**：line 16 "avaialbe"、"Space-tracker/Space-Tracker" 大小写不统一、Ref [7][8] 重复引用 Chen et al. ISPRS 2024（tsgG、YStx）。全文校对。

### P1：核心技术缺陷（决定录用与否的实验补充）

- [ ] **MOT 检测器不一致混淆了核心结论**（tsgG 列为最关键缺陷；eKyB、CS 评审同样指出）：
  - Car 用 HiEUM（无监督、RsCarData 训练、NMS 0.1、max det 128），Airplane/Ship/Train 用 Faster R-CNN（有监督、SAT-MTB 训练、NMS 0.5、max det 300），跨列比较不成立。
  - 行动：(a) 在 Car 子集上用同一检测器（建议 Faster R-CNN）跑对照实验；(b) 报告两个检测器的 AP；(c) 若无法补齐，明确在 Table 3 标注"不可跨列比较"并改写 L291-311 的结论表述。
- [ ] **RsCarData 检测器泄漏问题**（CS 评审第 4 条，NeurIPS 评审未发现但 WACV 评审可能发现）：
  - HiEUM 使用作者发布的 RsCarData checkpoint，而 Car 池又包含 RsCarData 本身，属于 in-domain 评测混入统一基准。
  - 行动：按数据集分别报告结果；区分 in-domain 与 cross-domain 检测；考虑 leave-one-dataset-out 或在并集上训练检测器的协议。
- [ ] **属性统一可能被源标注异质性污染**（tsgG、CS 评审第 8 条）：
  - 例如 occlusion 在 SV248S（>50 帧）与 SatSOT（空间覆盖）定义不同却被合并。
  - 行动：提供逐属性的映射表（精确阈值 + 每个数据集的序列/帧覆盖统计）；对定义严格一致的属性子集做敏感性分析，验证 Tab 9-11 的排名结论是否成立；小样本属性组（LTO 9 条、IM 12 条、AM 17 条、DEF 22 条）需报告不确定性或降调因果性表述。
- [ ] **SOT clip 边界效应未分析**（tsgG、CS 评审第 5 条）：
  - 32 帧切片 + 状态传递会改变失败恢复行为，且对不暴露置信度/valid box 状态的方法不公平；全局 τ=0.5 阈值在不同 tracker 家族间校准不可比。
  - 行动：补 T=16/32/64 消融；提供标准的不间断逐帧在线评测作为主结果或对照；统一定义"no valid box"；量化置信度过滤的影响。若 32 帧是沿用先前工作的标准选择，引用出处。
- [ ] **OOTB 几何评测不统一**（CS 评审第 6 条）：
  - mask-based 方法转 OBB 对 OBB GT 评测，HBB tracker 对 enclosing HBB 评测，两类方法解的不是同一个定位任务。
  - 行动：以统一的 HBB-only 指标作为主要可比分数；OBB-aware 分析作为附加；把 OOTB 拆分单独呈现而非只放补充材料。
- [ ] **8 像素阈值验证不够稳健**（CS 评审第 7 条）：
  - Figure 1 缺不确定性区间、逐数据集曲线、bin 计数与对 tracker 选择的敏感性。
  - 行动：补 bootstrap 置信区间、per-dataset 曲线、明确 bin 定义与样本数、6/8/10px 阈值敏感性分析。

### P2：基线与相关工作补充

- [ ] **补领域特异性基线**（YStx、CS 评审第 9 条、tsgG）：
  - 卫星视频专用 MOT：MCTracker（2024）、Cui et al. 卫星视频小目标 MOT（2024）、Feng et al. 端到端图跟踪（2025）。
  - Tiny object tracking：Zhu et al. Tiny Object Tracking 数据集与 MKDNet 基线。
  - SAM 系卫星适配：SatSAM2（2025），与 SAM 2/SAMURAI 的长遮挡结论直接相关。
  - 轻量级 tracker（如 SiamFC）用于星上部署的速度-精度权衡（tsgG）。
  - MOTIP、MASA 等端到端方法：至少在 10-20 条序列上给出零样本定量结果 + 失败分析放附录，而非仅口头说明（tsgG）。
- [ ] **收窄结论表述**：如"SORT is the right inductive bias for satellite MOT"应限定为"在两个固定检测器配置下、有限关联模块集合中最优"，除非补齐卫星专用与端到端方法对比（CS 评审）。
- [ ] **补充相关工作讨论**（YStx 列出 7 篇，CS 评审列出 7 篇，有重叠）：
  - SmallTrack (TGRS 2023)、Tiny Object Tracking (TNNLS 2023)、TSFMO (ACCV 2022)、UAVMOT (CVPR 2022)、Extended FPN (TMM 2021)、ESOD (TIP 2024)、UIU-Net (TIP 2022)。
  - Visible-Thermal Tiny Object Detection benchmark（Ying et al. 2024，与模态声明修订直接相关）。
  - 讨论重点：星载场景与常规小目标场景的本质区别（亚 8 像素、弱纹理、非静止背景、低帧率运动模糊、长时遮挡），并基于此给出解决方向性的分析而非只报告差的性能（YStx 的核心诉求）。

### P3：数据集论文的规范性要求（DB Track 评审会重点看）

- [ ] **贡献定位与新颖性论证**（tsgG、YStx、eKyB 三位都提）：
  - 在 Introduction 中明确列出 2-3 条"只有跨数据集统一分析才能得到的洞见"（如 tiny + isotropic motion 模式、SORT/ByteTrack 在不同尺度上的反转），把这个论证放到中心位置。
  - 明确说明：哪些是新标注、哪些只是格式转换、哪些做了人工修正、做了多少质量控制（eKyB 的直接提问）。
- [ ] **标注可靠性与质量控制**（YStx）：
  - 亚 8 像素目标上微小的标注偏差或格式转换误差会显著影响 IoU 指标与排名。补充：质量控制流程、多标注者一致性、时序验证、HBB/OBB 统一转换的误差影响分析。
- [ ] **去重与数据独立性**（CS 评审第 3 条）：
  - 多个源数据集可能来自相关卫星平台，存在重复/近重复序列风险。补充：视频/帧/地理区域/采集日期/轨迹级别的重叠检测，以及防止源视频跨 train/val/test 泄漏的显式规则。
- [ ] **许可与伦理治理**（eKyB 标记了 ethics review；CS 评审同样标记）：
  - 提供逐数据集的治理表：许可证、获取方式、帧 vs 标注的再分发权限、地理/时间来源、已知使用限制。
  - 增加具体的 dual-use 声明：可能的滥用场景、为何采用 evaluation server 或 annotation-only 发布、敏感影像的保护措施。
  - 说明卫星影像的安全/隐私/军民两用含义。（你们已做过大量 license 审计与作者联络，把这些工作显式写进论文即可转化为加分项。）
- [ ] **可复现性发布清单**（eKyB、CS 评审）：
  - 版本化的源序列 manifest、数据校验和、精确 split 文件、转换脚本、检测器 checkpoint、配置文件、逐序列 provenance 元数据、data card 文档、能复现所有表格的说明。
- [ ] **补充 Limitations 章节**（YStx 说缺失，tsgG 给了具体建议）：
  - 标注噪声、数据集偏差、泛化限制、结论对基线配置的依赖；无夜间/多云样本、地理多样性有限（仅 Jilin-1/SkySat/ISS）、无动态卫星平台运动场景、（若修订后）无真正的多光谱/热红外数据。

---

## 三、建议的修改路线图

**第一阶段（1 周内，纯写作）**：P0 全部项 + 贡献定位改写 + Limitations 章节。这些不需要跑实验，但决定了论文的第一印象，NeurIPS 三位评审对占位符和数字矛盾的反应非常负面。

**第二阶段（2-4 周，补实验）**：
1. MOT 同检测器对照实验 + 检测器 AP（tsgG 明说这是 critical for rating）
2. RsCarData 按数据集拆分报告 + in/cross-domain 区分
3. Clip 长度消融 T=16/32/64 + 不间断评测对照
4. 8px 阈值敏感性 + 置信区间
5. OOTB 统一 HBB 评测
6. 属性映射表 + 一致属性子集敏感性分析

**第三阶段（并行进行）**：
1. 补 2-3 个卫星专用/tiny-object 基线（MCTracker、MKDNet、SatSAM2 择优）
2. 端到端方法零样本附录实验
3. 治理表、manifest、data card、dual-use 声明

**WACV 转投注意**：CS 评审发现的问题（模态声明、检测器泄漏、OOTB 几何、去重）在 NeurIPS 评审中只被部分发现，WACV 评审同样可能命中，建议全部处理而非只回应 NeurIPS 意见。WACV DB Track 对治理文档和可复现性的要求与 NeurIPS DB Track 类似，P3 部分不可省略。
