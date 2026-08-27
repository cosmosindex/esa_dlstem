# Space-Tracker 评审意见汇总与修改清单
**目标：NeurIPS 2026 → WACV 2027 Datasets & Benchmarks Track 转投**

评审来源：
| 来源 | 评分 | 置信度 | 核心立场 |
|---|---|---|---|
| NeurIPS Reviewer tsgG | 3（Borderline Reject） | 3 | 工程扎实、分析深入，但三个关键问题需大改 |
| NeurIPS Reviewer YStx | 2（Reject） | 3 | 贡献偏增量，缺领域特异性分析与可靠性讨论 |
| NeurIPS Reviewer eKyB | 2（Reject） | 4 | 一致性问题严重，数据集贡献定位不清，有伦理关切 |
| CS Paper 外部评审 | 2（Reject） | 4 | 最详细：模态声明、评测独立性、协议公平性均有实质问题 |
| Ethics Reviewer oFw4 | —（黄灯：Moderate） | — | 许可证 / 再分发实质已合规，但论文里没写；broader impact statement 根本不存在却在 checklist 中被声称存在 |
| Ethics Reviewer Fs91 | —（绿灯：Minor/negligible） | — | 伦理无问题；正文是技术意见，整体评价正面，六条中四条与已有清单重合 |

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

## 三、伦理评审意见（Ethics Review，2 位）

两位 ethics reviewer 的定位差别很大：oFw4 是真正的伦理评审（许可证 / 隐私 / 军民两用），Fs91 虽走 ethics 通道但正文几乎全是技术意见，且伦理上判定为绿灯。

### 3.1 Ethics Reviewer oFw4 — Moderate（黄灯，"可能可以解决"）

被 flag 的三项：data privacy、copyright、consent。具体两问：(1) 记录源数据集的许可证与再分发条款；(2) 说明卫星影像的安全 / 隐私 / 军民两用含义。

**（1）版权与再分发：实质已合规，问题在"没写进论文"。**
评审亲自核查了仓库，结论对我们有利：

- 九个源数据集条款互不兼容：SAT-MTB 是 CC BY 4.0；VISO 是 CC BY-NC-SA（ShareAlike 条款会"传染"任何含它的合并发布）；SV248S 与 OOTB 是 CC BY-NC；SatSOT、AIR-MOT、SDM-Car **无任何许可证声明**，按版权法默认为 all rights reserved。因此重新打包并再分发影像不合法。
- **但仓库没有这样做**：发布物只是一份轻量 JSON manifest（序列名、文件路径、属性元数据）+ 评测代码，用户各自到原始来源按其原始许可证下载——与 LaSOT / GOT-10k 相同的获取模式。评审确认 manifest 中**不含影像、不含拷贝的标注**。
- 评审还注意到并认可 `docs/dataset_license_audit.md` 这份内部审计，认为它正确地得出了"evaluation-only 是唯一可行方案"的结论。

→ 原文："So the substance of the concern is handled." 这一条**不是要补工作，而是要把已经做完的工作写进论文**。

**（2）三项必须修的文档问题：**

- **论文里完全没有这套发布模式与许可证分析**，读者不翻仓库就无从知晓。→ 加一个 licensing 小节或 datasheet。
- **审计文档自相矛盾**：`dataset_license_audit.md` 摘要表把 SatSOT 与 SDM-Car 标为"CC BY-NC 4.0 / 已确认"，逐数据集详情里却写的是"Not stated，待确认"；Action Items 里也仍挂着未完成的确认邮件。（**已复核属实**。该文件保持原样不动，矛盾在论文侧统一口径解决，见 3.3。）
- **CC BY 4.0 + Croissant 的覆盖范围要写明**：只覆盖作者自己的 manifest、标注与代码，**不覆盖第三方底层影像**。

**（3）隐私与知情同意：评审认为这个 flag 站不住脚。** 商业卫星与 ISS 视频，GSD 约 0.75–1 m，人在该尺度不可分辨，被跟踪目标只有几个像素宽；无人类受试者、无个人数据、无众包标注。→ 无需额外工作，但值得在 datasheet 里用一句话固化这个论证。

**（4）军民两用：评审发现了一个四位正式评审都没抓到的硬伤。**
checklist 里写"dual-use risks are discussed in the broader impact statement"、并声称已承认监控滥用风险，但**投稿中根本不存在 broader impact statement**，结论部分只谈正面应用。评审直言 "The checklist answer is simply inaccurate as written."
底层风险他判定为温和：序列、标注、tracker 全部已公开，论文主结论恰恰是"现有 tracker 在这些目标上表现很差"，没有释放任何提升他人能力的新东西。
→ 修法就是**真的把 checklist 声称存在的那段 broader impact 写出来**，写了他就认为问题关闭。

**oFw4 建议的缓解措施（三条，全部是写作 / 文档层面）：**
1. 论文中加 licensing / datasheet 小节，并**完成尚未闭环的许可证确认**；
2. 补 broader impact statement；
3. 明确 CC BY 4.0 的覆盖范围。

### 3.2 Ethics Reviewer Fs91 — Minor/negligible（绿灯）

伦理上判绿灯，正文给的是技术意见，且**对论文整体评价明显正面**：认可统一多源、清洗合并、百万级实例、18 属性评测体系，以及 7 SOT + 7 MOT 的基准结论（尤其"大基础模型在极小目标上的局限"这一发现），称之为 "a highly valuable dataset paper"，只是"部分技术细节与文字一致性需要打磨"。

六条具体意见：

| # | 意见 | 与现有清单的关系 |
|---|---|---|
| 1 | 数据集数量不一致：摘要 line 14 "nine" vs 引言 line 35 与 Sec 3 line 103 "eight" | **已在 P0**；至此四位评审独立命中，务必修 |
| 2 | SOT 评测不公平：OBB 数据集上 HBB tracker 对"GT OBB 的外接 HBB"评测，人为放大目标面积、稀释旋转误差，偏袒 HBB tracker；请说明为何不直接对原生 OBB 评测 | 与 P1「OOTB 几何评测不统一」同源，但**要求方向相反**，见下方注 |
| 3 | MOT 检测器异质：car 用现成检测器，airplane/ship/train 用自训练检测器，上游差异与下游跟踪指标强耦合，可能扭曲排名；请澄清影响或统一检测器架构 | **已在 P1**；至此四位评审独立命中，升级为最高优先级实验 |
| 4 | 属性定义含糊：18 个统一属性缺少量化定义，"motion blur""non-stationary background"等在各源数据集中标准主观；请给出严格的数学 / 边界判据 | 与 P1「属性统一被源标注异质性污染」同源，但要求更硬：**不只是映射表，要可计算的阈值** |
| 5 | 结论 line 323 写 "12 fine-grained attributes"，与摘要 line 16 / 引言 line 534 / 附录的 18 矛盾 | **已在 P0** |
| 6 | 8 像素小目标阈值的普适性：请讨论该经验阈值对未来不同 GSD 的超高分辨率卫星视频是否仍适用 | **新增**，见 3.3 的 P1 补充 |

> **注（第 2 条的方向冲突）**：CS 外部评审要求"以统一的 HBB-only 指标作为主要可比分数"，Fs91 则问"为什么不直接对原生 OBB 评测"。两者的共同底线是：**同一张表里的所有方法必须解同一个定位任务**。建议的统一答复是：主表用 HBB-only 指标（所有方法口径一致、都能参与），附加一张 OBB-aware 表只比较能输出旋转框的方法，并在正文中显式说明"外接 HBB 会放大面积、稀释旋转误差"这一偏置及其量级——给出外接 HBB 与原生 OBB 的平均面积比即可量化。

### 3.3 伦理评审带来的新增待办

**P0（纯写作即可完成，但性质是"checklist 与论文不符"，优先级等同事实性错误）**

- [ ] **补 Broader Impact / Societal Impact 章节**（oFw4 唯一的硬伤指控）。checklist 已声称它存在，现在必须让它真的存在。内容至少覆盖：
  - 正面用途：灾害响应、海事安全、交通与基础设施监测、环境监测；
  - 滥用场景：从轨道对船舶 / 飞机 / 车辆做持续监控；
  - 为何风险有限：所有序列、标注、tracker 均已公开，本文不发布影像、也不发布新的检测 / 跟踪能力，主结论恰恰是现有方法在这些目标上失效；
  - 已采取的缓解：manifest-only 发布、逐数据集许可证遵从、影像中不含可分辨的人物。
- [ ] **统一 SatSOT / SDM-Car 的许可证口径**（已复核 `docs/dataset_license_audit.md` 确有矛盾：摘要表标"CC BY-NC 4.0 / 已确认"，详情节写"Not stated"）。**该审计文件保持原样，不做改动**；在论文的 licensing 小节 / datasheet 中**以详情节的保守口径为准**——即 SatSOT 与 SDM-Car 按"未声明许可证 = all rights reserved"处理，因此只做 evaluation-only 引用。评审读的是审计文件本身，所以论文里最好用一句话说明该保守口径，避免读者比对时再次撞上这个矛盾。
- [ ] **明确 CC BY 4.0 与 Croissant 的覆盖范围**：在 README、Croissant 元数据与论文中同时写明"本许可仅覆盖本文作者产出的 manifest、标注与代码，不覆盖第三方底层影像；影像须按各自原始许可证从原始来源获取"。

**P1（新增实验 / 分析）**

- [ ] **8 px 阈值的 GSD 普适性讨论**（Fs91 #6）：现有 P1 只要求 bootstrap 区间与 6/8/10 px 敏感性，这里再补一段把阈值**从像素改写为物理量**的讨论——8 px @ 0.75–1 m GSD ≈ 6–8 m 地面尺度，正好落在小汽车（约 4.5 m）与中型船 / 飞机之间；说明真正不变的是"目标地面尺寸 / GSD"这一比值，未来更高分辨率视频下像素阈值会等比例放大。可直接用现有逐数据集 bbox 统计（`docs/bbox_stats/`）画一张像素 vs 米的对照。
- [ ] **属性的可计算判据**（Fs91 #4，强化现有 P1 项）：对至少 motion blur、non-stationary background、fast motion、occlusion 这四个最主观的属性给出可复算的定义（如 MB 用帧间梯度能量比或 GT 框位移 / 框边长；非静止背景用配准后的背景残差；FM 用 |Δcenter| / √area；OCC 用连续被遮挡帧数 + 可见面积比），并**在合并后的数据上重算一遍**，报告与源数据集原始标注的一致率。一致率低的属性要么重标、要么在表中降级为 dataset-specific。

**P3（治理文档，与原 P3「许可与伦理治理」合并执行）**

- [ ] **论文中新增 Licensing & Release Model 小节 / datasheet**：明确写出 (a) 九个源数据集逐一的许可证与获取方式；(b) 本发布是 manifest + 评测代码的 evaluation-only 模式，与 LaSOT / GOT-10k 一致；(c) manifest 不含影像、不含拷贝标注；(d) 隐私论证（GSD 0.75–1 m，人不可分辨，无个人数据、无众包标注）。oFw4 明确说这些 "fits comfortably in a camera-ready revision"。
- [ ] **完成尚未闭环的许可证确认邮件**：SatSOT、AIR-MOT、SDM-Car、SAT-MTB-SOS、IRSatVideo-LEO、LMOD。即使拿不到回复，也要在 datasheet 中如实写"已联系、未获答复，因此按 all rights reserved 处理，仅做 evaluation-only 引用"。

---

## 四、建议的修改路线图

**第一阶段（1 周内，纯写作）**：P0 全部项 + 贡献定位改写 + Limitations 章节 + **Broader Impact 章节** + **Licensing & Release Model 小节 / datasheet** + 统一 SatSOT / SDM-Car 的许可证口径（论文侧，不动审计文件）+ 明确 CC BY 4.0 覆盖范围。这些不需要跑实验，但决定了论文的第一印象：NeurIPS 三位评审对占位符和数字矛盾的反应非常负面，而 oFw4 的黄灯**只需要这一阶段的写作就能转绿**（他明说 licensing 与 broader impact 都 fits comfortably in a camera-ready revision）。

**第二阶段（2-4 周，补实验）**：
1. MOT 同检测器对照实验 + 检测器 AP（tsgG 明说这是 critical for rating）
2. RsCarData 按数据集拆分报告 + in/cross-domain 区分
3. Clip 长度消融 T=16/32/64 + 不间断评测对照
4. 8px 阈值敏感性 + 置信区间
5. OOTB 统一 HBB 评测
6. 属性映射表 + 一致属性子集敏感性分析 + **可计算的属性判据与源标注一致率**（Fs91 #4）
7. **8px 阈值的 GSD 普适性分析**（像素 ↔ 地面米制换算，Fs91 #6）

**第三阶段（并行进行）**：
1. 补 2-3 个卫星专用/tiny-object 基线（MCTracker、MKDNet、SatSAM2 择优）
2. 端到端方法零样本附录实验
3. 治理表、manifest、data card、dual-use 声明
4. **完成尚未闭环的许可证确认邮件**（SatSOT、AIR-MOT、SDM-Car、SAT-MTB-SOS、IRSatVideo-LEO、LMOD）；拿不到回复也要在 datasheet 中如实记录已联系但未获答复

**伦理评审的净结论**：两位 ethics reviewer 都**没有**要求任何新实验，也没有质疑数据获取的合法性——oFw4 甚至主动为我们的 evaluation-only 发布模式背书。唯一真正的失分点是**checklist 声称存在 broader impact statement 而论文里没有**，这属于必须无条件修掉的事实性问题。Fs91 的六条里四条与正式评审重合，等于把 P0 的数字矛盾与 P1 的检测器异质性再确认了一遍。

**WACV 转投注意**：CS 评审发现的问题（模态声明、检测器泄漏、OOTB 几何、去重）在 NeurIPS 评审中只被部分发现，WACV 评审同样可能命中，建议全部处理而非只回应 NeurIPS 意见。WACV DB Track 对治理文档和可复现性的要求与 NeurIPS DB Track 类似，P3 部分不可省略。
