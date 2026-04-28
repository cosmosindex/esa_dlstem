# Re-ID 在 MOT 里到底是干什么的

## 一句话

**Re-ID（Re-Identification）= 给每个 detection 算一个"长相向量"，让 tracker 能用"长得像不像"来判断两个框是不是同一个物体。**

不是让模型"知道"被追踪物体长什么样（那是分类/SOT 模板）；而是给 tracker 在做"这一帧的 detection 跟上一帧的哪条 track 是一对儿"这个匹配题时，多塞一条"外观相似度"作为线索。

## 它解决的问题：association

Tracking-by-detection 的流程是：

```
每帧:  detector → N 个框 (x,y,w,h,score)
        ↓
      tracker:  把这 N 个框分给 M 条已有 track  ← association
        ↓
      输出: 每个框带个 track_id
```

Association 是一个二分图匹配：行 = M 条 active track，列 = N 个 detection，每个 cell 是 cost。把 cost 矩阵丢给匈牙利算法 / lapjv 求解，得到匹配。

**关键问题：cost 怎么算？**

最朴素的办法（SORT、ByteTrack、OC-SORT 都这么干）：

- track 用 Kalman 预测它在当前帧的位置 → 得到一个预测框。
- cost = `1 − IoU(预测框, detection框)`。
- IoU 越大 → cost 越小 → 越可能是一对。

这套**只用运动**的方案在三种情形会崩：

1. **遮挡 / 漏检**：某条 track 连续几帧没有 detection 命中。Kalman 预测会越漂越偏，等物体重新出现时，预测框跟真实框已经没 IoU 了 → 匹配不上 → 物体被分配新 ID（**ID switch**）。
2. **快速运动 / 低帧率**：相邻帧位移大，IoU 直接掉到 0 → 匹配不上。
3. **拥挤场景里物体交叉**：两个人擦肩而过，纯靠位置 IoU 会把 ID 互换。

## Re-ID 怎么补救

**思路**：除了"它现在应该在哪儿"（运动），再加一条"它长什么样"（外观）。

具体步骤：

1. **特征提取**：拿一个预训练的 CNN（比如 FastReID 的 SBS-ResNet50），输入是 detection 的 crop（把框抠出来 resize 到 128×384），输出一个 D 维向量（FastReID 是 2048-D），L2 归一化。
   - 训练目标：让"同一个人不同时刻的 crop"在特征空间里距离近，"不同人的 crop"距离远。这就是 Re-ID 任务的本意——cross-camera person re-identification。

2. **track 维护一个滑动平均 embedding**：每次 track 跟某个 detection 匹配上，就用 EMA 把这个 detection 的特征融进 track 的 "appearance memory"：

   ```
   track.feat = α · track.feat + (1−α) · det.feat   (α≈0.95)
   ```

   这样一条 track 的 feat 是它最近被观测时的"平均长相"。

3. **关联时算 cosine 距离**：

   ```
   cos_dist[i, j] = 1 − cos(track_i.feat, det_j.feat)
   ```

   长得像 → cos 接近 1 → cos_dist 接近 0。

4. **融合到 cost 矩阵**：TrackTrack 的具体公式：

   ```
   cost = 0.5 · iou_dist + 0.5 · cos_dist
        + 0.1 · conf_dist + 0.05 · angle_dist
   ```

   IoU 和外观各占一半。运动暂时丢了（IoU=0）的时候，外观还在；外观无信息（cos≈0.5）的时候，IoU 还能救。

## 它带来什么、代价是什么

**好处**：

- **降 ID switch**：物体被遮挡 N 帧后重新出现，运动预测已经不准了，但外观特征记得"它就长这样"，能把它跟同一条 track 接回去。
- **拥挤场景更稳**：两个人交叉时，IoU 会指向错误的人，外观特征能纠正。

**代价**：

1. **额外的前向计算**：每帧每个 detection 都要过一遍 Re-ID 网络。这就是为什么把所有 detection 的 embedding 预先缓存到 `.npz`——把这部分推理一次性做掉，eval tracker 时直接读特征，避免反复算。
2. **Re-ID 模型要跟数据域匹配**。这是这边的关键问题：FastReID 的 `mot17_sbs_S50.pth` 是在 MOT17 行人数据上训的，crop 大小 128×384（行人长宽比）。卫星视频里：
   - 车 ~10×10 px，被强行拉到 128×384 后已经面目全非；
   - 类内差异极小（一辆灰色小轿车跟另一辆灰色小轿车从空中看几乎一样）；
   - 行人特有的外观线索（衣服颜色、姿态、配件）在车上完全没有。

   结果就是这 2048-D 特征里**有效信息很少**，cos 距离接近随机噪声 → cost 矩阵里 cos 那一半基本不起作用 → 实际等价于"权重砍半的 IoU + 一些扰动"。

## 这就是为什么 TrackTrack 这次效果不一边倒

如果在 MOT17 上跑，TrackTrack 全方位碾压 SORT/ByteTrack（HOTA 高 5-10 个点），靠的就是 Re-ID + 复杂关联策略。但在卫星车数据上，Re-ID 那一半几乎是哑的，剩下的只是 IoU 关联策略上的差异，所以：

- **rscardata**（小数据 + 简单运动）：Re-ID 哑了也无所谓，TrackTrack 复杂的关联策略带来一点优势 → HOTA 0.358 vs SORT 0.402，接近。
- **sdmcar**（车多但分布均匀）：类似情况 → HOTA 0.281 vs SORT 0.252，略胜。
- **satmtb**（场景多样、目标尺度跨度大）：阈值没适配 + Re-ID 也帮不上忙 → 直接崩到 0.098。

## 想让 Re-ID 真正有用怎么办

要么：

- **在 RsCarData/SAT-MTB 上微调 FastReID**：把数据按 track_id 切成 query/gallery，跑一遍 cross-id triplet loss。一两天的事。
- 或者**换个对小尺寸目标更合适的 backbone**（不强求 128×384，可以训一个 64×64 的小模型）。

不是必须的——目前 CSV 里的对比已经能讲一个故事："在卫星目标场景下，appearance-based MOT 的 ReID 部分会被域差异稀释，运动+IoU 仍然是更鲁棒的关联信号"。

## 当前 sweep 中各 tracker 的 Re-ID 状态

| Tracker     | 用 Re-ID? | 实际配置                                                                                                                                                                                |
| ----------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SORT        | 否        | 纯 Kalman + IoU 关联，从设计上就没有外观特征。                                                                                                                                          |
| OC-SORT     | 否        | OCM/OCR 都基于运动观测，没有 Re-ID。                                                                                                                                                    |
| ByteTrack   | 否        | 只用 score 分高/低两阶段 + IoU 匹配。                                                                                                                                                   |
| BoT-SORT    | 否（关闭状态） | 上游 BoT-SORT 默认 `--with-reid=False`，是 motion+IoU 版本（即所谓 BoT-SORT-IoU）；要带 Re-ID 必须传 `--with-reid` 并加载 FastReID 检查点。`models/trackers/botsort.py` 包装时没有传 reid 相关参数，跑的是 IoU-only 变体。 |
| **TrackTrack** | **是**    | `iterative_assignment` cost = `0.5·iou_dist + 0.5·cos_dist + 0.1·conf_dist + 0.05·angle_dist`，cos 来自 FastReID 2048-D 特征（pre-cached `.npz`）。                                       |

也就是说，5 个 tracker 中**只有 TrackTrack 启用了 Re-ID**。BoT-SORT 在论文里跟 ByteTrack 的主要差距正是 Re-ID + GMC，而这边两者都关了，所以它的 HOTA 反而比 ByteTrack 还差（关联完全靠 IoU，但 BoT-SORT 的 NMS/threshold 默认值没适配卫星小目标）。

---

## 后续实验：补一组 BoT-SORT-ReID

为了让对比更公平，又跑了一组 **BoT-SORT-ReID**（复用现有的 FastReID feats cache），跟原版 BoT-SORT-IoU 形成 ablation。实现细节：

- `models/trackers/botsort_reid.py` — `with_reid=True`，把 `self.tracker.encoder` 替换成一个查表 stub（`_CachedFeatureEncoder`），按 box 精确匹配从 `.npz` 缓存里取出对应行的 2048-D 特征。整个 FastReID forward 一次都不会被调用。
- `eval_botsort_reid.py` + `configs/MOT/tracker/botsort_reid_*.yaml` — 跟 TrackTrack 路径并列。

### 数值对比（来自 `hota_summary.csv`）

| Dataset   | BoT-SORT (IoU-only) HOTA | BoT-SORT-ReID HOTA | Δ      |
| --------- | ------------------------ | ------------------ | ------ |
| rscardata | 0.042                    | **0.233**          | +0.191 |
| satmtb    | 0.051                    | **0.291**          | +0.240 |
| sdmcar    | 0.040                    | **0.225**          | +0.185 |

ID switch 也大幅下降：

| Dataset   | BoT-SORT IDsw | BoT-SORT-ReID IDsw |
| --------- | ------------- | ------------------ |
| rscardata | 70633         | 45591              |
| satmtb    | 72128         | 24647              |
| sdmcar    | 58448         | 15720              |

### 但这不能直接归功于 Re-ID 特征本身

差距大并不代表 appearance 起了作用。两点关键观察：

1. **跟 SORT/OC-SORT 比，BoT-SORT-ReID 没有 appearance 加成应有的优势。**
   - rscardata：SORT 0.402 > BoT-SORT-ReID 0.233。如果 ReID 是有效信号，appearance-aware 不应该输给纯 IoU 的 SORT。
   - satmtb：OC-SORT 0.278 ≈ BoT-SORT-ReID 0.291，AssA 0.279 ≈ 0.307 —— 持平，没有"看脸认人"那种数量级的提升。

2. **原版 BoT-SORT 崩得彻底是关联策略问题，不是缺 ReID。**
   上游 `BoTSORT.update` 在 `with_reid=False` 路径里走 `dists = matching.fuse_score(ious_dists, detections)` —— 把 IoU dist **乘以**检测置信度。HiEUM 那种 0.3-0.5 的低分被 fuse_score 推向高代价，导致几乎一帧一新 ID（IDsw 7w+）。
   打开 ReID 后走的是另一条分支：`dists = np.minimum(ious_dists, emb_dists)`，绕过了 fuse_score 这个对小目标致命的死路。所以分数飙升的"功劳"主要在**走了不同的代码路径**，跟 appearance 特征本身关系不大。

### 验证

跟 TrackTrack 的对比也能佐证这一点。TrackTrack 的代价是 `0.5·iou + 0.5·cos`，cos 是真的在被使用的：

- rscardata：TrackTrack HOTA=0.358（接近 SORT 0.402，比 BoT-SORT-ReID 0.233 高得多）
- sdmcar：TrackTrack 0.281 > SORT 0.252 > BoT-SORT-ReID 0.225
- satmtb：TrackTrack 因为 `min_box_area=25 + det_thr=0.3` 把绝大多数小目标 dets 滤掉了（n_dets=16747 vs 其他 10 万+）— 是阈值问题，不是 ReID 问题

如果 FastReID 的 MOT17-行人特征对卫星车真的有信息量，TrackTrack 应该全面碾压所有 motion-only 的 tracker。实际上 TrackTrack 跟 SORT/OC-SORT 是混在同一个性能区间的——再次印证 appearance 那部分是哑信号。

### 结论

5 个 tracker 都跑过之后的 takeaway：

- **在卫星目标 + MOT17 ReID 权重的组合下，appearance 特征几乎不贡献任何信号**。
- BoT-SORT-IoU vs BoT-SORT-ReID 的 19-24 个 HOTA 点的差距，主要来自上游代码里 `with_reid=True` 走了一条对低分小目标更宽松的关联分支，**不是** appearance 真的在帮忙做识别。
- 想要真正回答"appearance-aware MOT 对卫星车有没有用"，需要在 RsCarData / SAT-MTB 上 fine-tune 一个 ReID backbone（按 track_id 切 query/gallery，cross-id triplet loss 训一两天就行）。
- 当前这组实验已经能讲一个完整的故事："motion+IoU 在卫星目标场景下仍然是更鲁棒的关联信号；MOT17-pretrained ReID 在域差异下被稀释；用 ReID 模块要小心不是真的在用 ReID，而是不小心走对了一条代码路径。"
