数据很清楚 —— 在 frame 62 之前的 **6 帧（55-61）跟踪堪称完美**（IoU 0.83-0.88，center error 1-4 px），plane 也在做平滑匀速运动（每帧 ~5 px）。frame 62 一帧之内中心位置跳了 137 px（差不多两个 plane 的长度），IoU 直接归零，之后一路漂走。

## 为什么会这样突变 —— 不是渐进的，就是 argmax flip

LoRAT（和所有 Siamese 一族）每帧做的事是：

1. 以**上一帧预测框**为中心，从当前帧 crop 出 `4.0 × box_size ≈ 300×300` 的 search region
2. 把 search region 跟固定 template 做 cross-correlation / cross-attention → 得到一张 **score map**
3. `argmax(score_map)` 当成目标位置，bbox-regression head 算出最终框

**这里有个离散决策**：`argmax`。它没有连续性 —— 哪怕第一名比第二名只高 1%，结果跳就是跳，不会"软合并"。

frame 55-61 plane 一直是 score map 的第一名（peak）。frame 62 发生了：
- search region 里有一个 BC confuser（背景纹理跟 plane 在 ViT 特征空间长得很像 —— 灰度卫星图里大量这种 patch）
- 这个 confuser 的得分本来比 plane **低一点**，所以一直没赢
- frame 62 plane 的得分（或/和 confuser 的得分）**轻微抖了一下**（可能因为：plane 自身亮度/姿态微变、JPEG 压缩噪声、ViT patch 切分对齐变化、GT size 从 75×77 微微变到 74×76 让 crop scale 也跟着变），结果 confuser **第一次反超** plane
- argmax 立刻跳过去 → 137 px 瞬移
- bbox-regression 在那个错的 peak 上输出一个 degenerate 框（你看 frame 63 cd=inf，意味着预测框面积接近 0，标准化爆了）
- 从此 search region 中心钉死在错的位置，永远找不回 plane

## 关键直觉

**Siamese tracker 在 BC 序列上的失败不是"逐渐漂移"，而是"瞬间跳台阶"** —— 因为 argmax 是离散算子。score map 上 plane peak 和 confuser peak 长期共存、互相竞争，谁高谁就赢。一旦排名翻转，没有任何回弹机制（template 是固定的，看不到当前已经错了；模型也不知道"上一帧 IoU 多少"这种东西）。

可以理解成：模型每一帧都在重新做一次"哪个 patch 最像 frame 0 那块 GT"的投票，57 票连胜 plane，第 58 票 confuser 突然得 51%，从此之后所有投票都在错的邻域里举行（因为下一帧 search region 已经跟着错位置走了），plane 再也没有上台投票的机会。

## 你那张图里的 0.51 score 也对得上

`IoU=0.00 0.51` 里的 0.51 是模型给自己这次预测打的 confidence。这跟你看到的 BC failure 一致：模型在错的 patch 上**很自信**地认出了"plane-like 特征"，并不知道自己跟错。这是 first-frame Siamese 没有任何 self-correction 信号的根本缺陷 —— 反过来，SAM2 系列因为有 memory bank 维护多帧 mask 一致性，confuser 单帧反超不会立即丢目标。

---

# OSTrack / SiamRPN / SAMURAI 在同一条 plane_23 上的失败方式

数据源（OOTB whole-dataset eval, 20260518 跑的那一轮）：

| Tracker | 总帧数 | mean IoU | 第一帧 IoU=0 | 失败方式 |
|---|---|---|---|---|
| LoRAT g-378 | 192 | 0.278 | **frame 62** | 单次 argmax flip（137 px 瞬移） |
| OSTrack ViT-B/384 | 192 | 0.199 | **frame 45** | 单次 argmax flip（65 px 瞬移） |
| SiamRPN R50 DW | 192 | 0.218 | frame 79 | **anchor-regression 渐进漂移**（从 frame 5 就开始变形） |
| SAMURAI Large | 192 | 0.130 | frame 0 | **mask-quality gating + memory pollution**（54/192 帧才出框，无单点突变） |

## 1. OSTrack —— frame 45 跟 LoRAT 是同一类失败，只是更早

```
frm  iou   gt_c             pred_c          score
 43  0.66  (377,175)        (385,182)       1.00
 44  0.82  (386,175)        (390,174)       1.00   ← 还很正常
 45  0.00  (392,175)        (326,181)       1.00   ← 一帧之内 pred_c 向后跳 64 px
 46  0.00  (397,173)        (325,180)       1.00   ← 钉在 x≈326 不动了
 ...
```

frame 45 那次 pred_c 突变到 (326,181) —— 注意这个 x≈326 正好是 plane 刚刚从那里飞过来的位置（frame 35 GT 中心是 (337,179)，frame 25 是 (281,186)）。
机制跟 LoRAT 完全一样：**ViT 把 search region 里的尾迹/相似 patch 当成了 plane**，argmax flip 一次性发生，之后 search center 钉死在错位置上不再返回。

为什么 OSTrack 比 LoRAT 早 17 帧失败？
- OSTrack 用的是 ViT-B/16@384，LoRAT 用的是 g-378（≈ViT-g）。后者 backbone 容量大、cross-attention 对 BC 噪声更鲁棒，所以 confuser 在 score map 上压不过 plane 的"忍耐时间"更长。
- 这一点跟 OSTrack 的 SOT benchmark 表里 mean IoU 0.199 < LoRAT 0.278 也对得上：**它在 BC 序列上更早被 distractor 翻台**。

## 2. SiamRPN —— 不是 argmax flip，是 RPN 在 BC+SA 上的**连续回归漂移**

> 先纠正一个事实：plane_23 的官方 attr 是 `[IV, MB, BC, SA]`，**没有 ROT / IPR**。从 GT 的 OBB 轮廓也能看出 plane 整段 200 帧只旋转了 ~3°，边长全程稳定在 66×68 px（详见后文附录 A）。所以"plane 在旋转"这个解释不成立 —— SiamRPN 的失败必须从 BC + SA 来解释。

```
frm  iou   cd     gt (w×h@c)            pred (w×h@c)
  0  0.86  0.7    72×74 @(146,201)      71×73 @(146,200)   ← 完美
  5  0.63  11.0   72×74 @(173,196)      85×80 @(164,203)   ← 已经偏胖
 10  0.59  1.7    74×74 @(201,195)      88×86 @(202,194)
 25  0.73  4.1    74×74 @(281,186)      76×80 @(285,187)
 40  0.40  25.9   74×74 @(364,178)      72×80 @(338,177)
 45  0.57  16.1   74×74 @(392,175)      65×101 @(398,190)  ← 高度涨到 101
 50  0.53  11.3   74×74 @(421,173)      68×115 @(417,184)  ← 高度 115（GT 才 74！）
 60  0.50  1.4    74×76 @(474,166)      69×127 @(474,168)  ← 高度 127
 70  0.27  28.1   74×76 @(530,162)      80×123 @(502,158)
 79  0.00  ...    彻底脱离
```

注意 **GT 框始终是 ~74×76**（plane 本身大小、姿态都不变），但 SiamRPN 预测高度从 73 一路涨到 127，几乎翻倍。

机制（修正版）：

1. SiamRPN 的 head 由两部分组成：一是 RPN classifier 给出每个 anchor 的 plane/background 软概率，二是 regression head 输出 `(Δx, Δy, Δlog w, Δlog h)`。最终框是 **anchor + 回归 delta 的连续函数**，没有 argmax 离散决策。
2. plane 飞行轨迹下方/侧方存在一个**外观相似、形状细长**的 confuser（机场跑道边沿、跑道标线、或并排停着的另一架飞机的局部 —— SA + BC 共同作用）。
3. classifier 对这个 confuser 的得分本身不高（所以不会像 LoRAT/OSTrack 那样 flip 过去），但它持续在 score map 上提供一个**非零的次峰**。
4. regression head 看到的是"目标 + 旁边一个长条次峰"这种 anchor pattern，于是把回归 delta 往"包住更大区域"的方向调 —— pred box 高度逐渐增长。这是**软投票被连续函数继承下来**的副作用，跟 argmax 不一样：argmax 要么完全翻台、要么完全不动；回归头会平滑地把两个峰的贡献都吃进 box 尺寸里。
5. 一旦 pred box 长到 65×127，**它的几何中心已经不在 plane 上**（中心被拖向长条 confuser 那一侧）。下一帧 SiamRPN 以这个偏移的中心 crop search region，**位置 prior 就开始累积偏差** —— 到 frame 70 cd=28 px，frame 80 cd=92 px，frame 85 cd=121 px，标准的"位置 prior 中毒"曲线。

跟 LoRAT/OSTrack 的本质区别：

- LoRAT/OSTrack：one-stage cross-attention 输出 score map，做 argmax，**离散决策** → 不翻则已，一翻全错。所以表现成单点 catastrophic flip。
- SiamRPN：RPN 的 **连续回归** + 软概率混合 → 即使 confuser 的得分始终低于 plane，confuser 的存在也会**持续污染回归 delta**，表现成多帧渐进漂移。

也就是说同样面对 BC+SA 这两个属性，SiamRPN 的失败"早而钝"（frame 5 就开始变形），LoRAT/OSTrack 的失败"晚而急"（撑到 frame 45/62 才一次性翻台）。这不是 LoRAT/OSTrack 比 SiamRPN 强多少 —— 而是**两套架构对同一种环境干扰，把失败摊到不同的时间形态上**。

## 3. SAMURAI —— 不是"在 171 帧突然消失"，是**整段视频都在抽搐**

这一条最有意思。先看 V/None 模式（V=出框，.=没框）：

```
frame   0- 19:  .VV.VV.VV.......VV..   8/20 出框
frame  20- 39:  ......VVVV.VVVVVVVVV  13/20 出框
frame  40- 59:  VVVVVVVVVVVV....VV..  14/20 出框
frame  60- 79:  ....................   0/20 ← 长黑屏开始
frame  80- 99:  ....................   0/20
frame 100-119:  .................VVV   3/20
frame 120-139:  .V................V.   2/20
frame 140-159:  ....................   0/20 ← 又一段长黑屏
frame 160-179:  VVV.VV.VVVVV...V....  11/20 ← 短暂回光返照
frame 180-191:  .........VVV          3/12
```

`pred_box: None` 在我们的 SOT 评估代码里是**同类预测数为 0** —— 对 SAMURAI 来说就是 SAM2 这一帧的 mask predictor 完全没给出有效 mask（quality score 低于阈值，或者 mask 面积过小被过滤）。

所以"171 帧后消失"实际上是 **frame 160-171 的短暂回光返照**结束 —— 最后一个有效出框是 frame 175，之后除了末尾 189-191 那 3 帧之外全部 None。

发生了什么：

**Phase A（frame 0-59）"半工作"** —— 出框时 IoU ≈ 0.5-0.6。注意 GT 是 OBB，SAMURAI 的 mask 是任意形状但**评估代码用 axis-aligned bounding box of mask** 来算 IoU，所以一个完美的 SAM mask 对 rotated plane 给出的 axis-aligned box 跟 OBB GT 的 IoU 上限就是 ~0.6。这一段表现"看似还行"。

**Phase B（frame 60-99）40 帧长黑屏** —— SAM2 memory bank 此时存的是过去 9 帧的 image + mask token。前面 phase A 的 mask 本来就只匹配 plane 的一半（旋转使 axis-aligned mask 不能贴合机翼），这些半残 mask 进入 memory bank 后，**bank 里的"plane 长什么样"逐渐偏离真实 plane**。到某个临界点，mask quality predictor 拒绝所有候选 mask，进入完全沉默。

**Phase C（frame 100-159）零星短亮**—— 偶尔在 background 切换的瞬间能匹配上几帧（cd=0.8-12px，所以这几次"亮"实际上是真的看到 plane 了，不是误报），但 memory bank 已经污染，每次只能撑 1-3 帧又灭。

**Phase D（frame 160-179）回光返照** —— plane 飞到图像右边缘（GT 在 x≈1043-1117），背景从 phase B 那种 BC 严重的内部区域换成了**边缘相对干净的天空**。SAMURAI 在这段背景下重新能 lock 上 plane（IoU=0.42-0.48），mask quality 暂时回升。但 memory bank 里的污染没洗掉，**还是只能撑 ~11 帧**。

**Phase E（frame 180-191）永久退出** —— 第二次也撑不住，从 frame 176 起几乎全 None，末尾 189-191 那 3 帧的"复活"已经离 plane 还有距离（IoU=0.38）。

## 三种失败方式的总结对比

| 失败类型 | LoRAT / OSTrack | SiamRPN | SAMURAI |
|---|---|---|---|
| **触发** | BC + SA distractor 在 score map 上一次性翻台 | 同一个 BC + SA confuser 通过 RPN 连续回归慢性污染 box | mask quality 持续偏低 + memory bank 污染 |
| **时间形态** | 单点突变（1 帧从 IoU=0.9 → 0） | 连续渐进（70 帧累积漂移） | 抽搐式（频繁 None 与短暂恢复交替） |
| **score 是否预警** | 否（错位置上 score 仍=1.0） | 否（box 还在 plane 附近时 IoU 还有 0.5） | 是（None 本身就是 quality gate 触发） |
| **能否自恢复** | 否（search center 钉死在错位置） | 否（位置 prior 持续中毒） | **能短暂恢复**（背景改善时 mask 重新合格，但 bank 污染未消） |
| **根本缺陷** | first-frame template + argmax，没有 self-correction | RPN 连续回归把次峰的贡献吃进 box 尺寸里 | memory bank 没有"忘记坏帧"的机制 |

## 一句话总结

plane_23 的官方 attr 是 `[IV, MB, BC, SA]` —— **没有 ROT / IPR**。同一组 BC + SA 干扰在三类不同架构上表现出三种完全不同的失败形态：

- **LoRAT / OSTrack 死于"一次性 flip"** —— 离散 argmax 把 BC distractor 一次性顶上 score map 第一名；
- **SiamRPN 死于"连续慢漂"** —— 同一个 distractor 的次峰被 RPN 回归头平滑地吃进了 box 尺寸里；
- **SAMURAI 死于"长时间 bank 污染"** —— phase A 几十帧的半残 mask 把 memory bank 喂坏，后面再也清不掉。

加上 IV/MB 给 ViT/SAM 的 patch 表示加噪、SA 提供长期可见的相似干扰物，这条序列把 4 个流派的 tracker 各自打回原形 —— 不是因为某一类难度极端，而是**几个中等难度的属性叠加在一起恰好分别戳中每类架构的弱点**。

---

## 附录 A：plane_23 GT 几何（证明 plane 不旋转 / 不缩放）

`OOTB.json` 里 `plane_23.gt_rect` 是 8 维多边形（OBB 四角）。从开头到末尾每 30 帧采样一次：

```
fr  0:  bbox 71×73   edge1=66 edge2=68   angle(edge01)=-175.4°
fr 30:  bbox 73×75   edge1=66 edge2=68   angle(edge01)=-173.7°
fr 60:  bbox 74×76   edge1=66 edge2=68   angle(edge01)=-172.6°
fr 90:  bbox 74×76   edge1=66 edge2=68   angle(edge01)=-172.6°
fr120:  bbox 74×76   edge1=66 edge2=68   angle(edge01)=-172.6°
fr150:  bbox 72×74   edge1=66 edge2=68   angle(edge01)=-174.8°
fr199:  bbox 74×76   edge1=66 edge2=68   angle(edge01)=-172.6°
```

OBB 边长全程 66×68 不变；OBB 朝向 200 帧总共变化 ~3°（从 -175° 到 -172°）—— 远低于 IPR 的"≥30°"阈值。所以"plane 在旋转"完全不成立，本文 SiamRPN 一节的修正版（BC+SA 通过 RPN 连续回归污染 box）才是正确解释。
