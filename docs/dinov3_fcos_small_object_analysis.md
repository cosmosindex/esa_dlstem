# Why DINOv3 + FCOS collapses on small objects

Analysis of the frozen-ViT / dense-FCOS detector used in the fire, BIRDSAI and
SAT-MTB detection experiments. The headline symptom is the leftmost bin of
`docs/use_case_results/figures/fire_size_trend.pdf`: objects below 18 px get
**Recall 0.047 at Precision 0.864**.

The collapse is not a failure of the ViT features. It is three stacked
mechanisms, two of which are pure geometry in the FCOS head, and one of which
is a genuine information bottleneck in the ViT patch embedding. The reported
`0.047` is additionally inflated by the `SCORE_THR = 0.5` plotting convention:
the same checkpoint reaches `0.386` at a 0.05 threshold.

All numbers below are measured on the dumped per-frame predictions, not
estimated.


## 1. The architecture as trained

A **frozen DINOv3 ViT-B/16 backbone** plus a **single-scale, anchor-free FCOS
head trained from scratch**. No FPN, no anchors, no multi-level assignment.
Total 90.6 M parameters, of which 86.6 M are the frozen backbone.

```
input (B, 3, H, W)
      | frozen ViT-B/16, torch.no_grad(), eval() mode
tokens (B, N, 768)                       drop CLS + register tokens, keep the
      |                                  trailing (H/16)*(W/16) patch tokens
feat (B, 768, H/16, W/16)                stride-16 token map
      | optional F.interpolate(x2, bilinear, align_corners=False)
feat (B, 768, Hp, Wp)                    only when fcos_feat_stride is set
      |
input_proj: Conv2d(768 -> 256, k=1)
      |
      +-- cls_tower: 4x [Conv2d(256,256,k=3,p=1) -> GroupNorm(32,256) -> ReLU]
      |        \-- cls_logits:  Conv2d(256 -> num_classes, k=3)
      |
      \-- reg_tower: 4x [Conv2d(256,256,k=3,p=1) -> GroupNorm(32,256) -> ReLU]
               +-- bbox_pred:   Conv2d(256 -> 4, k=3) -> ReLU(scale * x)
               \-- centerness:  Conv2d(256 -> 1, k=3)
```

Source: `DINOv3FCOSHead`, `models/dinov3.py:105-156`.

The two towers share a design but not weights. `centerness` branches off the
regression tower. `bbox_pred` emits `(l, t, r, b)` distances **in stride
units**, passed through ReLU to stay non-negative and multiplied by
`self.stride` at decode time. `scale` is a single scalar `nn.Parameter` --
vanilla FCOS has one per FPN level, and there is only one level here. The
classification bias is initialised to `-log((1 - 0.01) / 0.01)`, the standard
focal-loss prior.

This head replaced a DETR-style head (`head_type: detr`, still in the code)
which no-object-collapsed on this data and never produced a result.

### Losses (`_fcos_loss`, `models/dinov3.py:597-642`)

```
loss = loss_cls + loss_reg + loss_ctr          # equal weight, no coefficients
```

| term | definition |
| --- | --- |
| `loss_cls` | sigmoid focal loss over all locations, divided by `num_pos` |
| `loss_reg` | GIoU loss on positives only, **weighted by the centerness target** |
| `loss_ctr` | BCE-with-logits on positives only |

### Inference (`_fcos_postprocess`)

```python
scores = cls_flat[b] * ctr_flat[b][:, None]    # models/dinov3.py:655
```

Decode ltrb per location, clamp to the image, class-wise `batched_nms`.

### Per-experiment stride

| config | `img_size` | `fcos_feat_stride` | effective grid |
| --- | --- | --- | --- |
| `dinov3_fire.yaml` | 640 | *absent* -> 16 | 40x40 |
| `dinov3_birdsai.yaml` | 640 | 8 | 80x80 (upsampled) |
| `dinov3_satmtb.yaml` | 1024 | 8 | 128x128 (upsampled) |


## 2. The sampling grid

Feature location `i` corresponds to the input-image patch spanning pixels
`[i*16, (i+1)*16)`. Its geometric centre is `i*16 + 8`, hence

```python
sx = (torch.arange(Wp) + 0.5) * self.stride    # models/dinov3.py:546-548
```

giving `8, 24, 40, ..., 632` for a 40x40 grid on a 640 input. Two reasons the
half-pixel offset matters:

* **No systematic regression bias.** FCOS regresses `l = x - x1`, `t = y - y1`,
  `r = x2 - x`, `b = y2 - y` from the location to the four box edges. Anchoring
  at `i*stride` (the patch's top-left corner) while the feature describes the
  whole patch would understate every `l`/`t` and overstate every `r`/`b` by
  half a stride, and would move the peak of the centerness target off the patch
  centre.
* **Symmetric coverage.** `(i+0.5)*16` leaves an 8 px margin on both sides of a
  640 px axis; `i*16` leaves 0 on the left and 16 on the right.

This is the FCOS paper's definition, written there as
`(floor(s/2) + x*s, floor(s/2) + y*s)`.

The `align_corners=False` in the upsample branch uses the same
half-pixel-centre convention, so after upsampling the stride-8 locations
`4, 12, 20, ...` still sit at the centre of their 8x8 regions. `align_corners=True`
would offset the two conventions by half a cell.


## 3. Mechanism 1 -- boxes that contain no grid point

A location is positive iff it lies strictly inside the GT box **and** inside a
`center_radius * stride` window around the box centre:

```python
inside    = ltrb.min(dim=-1).values > 0                    # models/dinov3.py:573
in_center = (xs > cx - rad) & (xs < cx + rad) & ...        # rad = 1.5 * stride
valid     = inside & in_center
```

Ambiguity is then resolved by minimum area (`areas[~valid] = 1e9; areas.min(-1)`).
Note the asymmetry: this guarantees each *location* is claimed by at most one
GT, but **nothing guarantees each GT receives at least one location**. There is
no fallback assignment.

For a box of side `L` on a grid of spacing `s`, the probability of containing a
grid point is `min(1, L/s)` per axis. For `L < s` most boxes contain none.

Measured over all 9227 GT boxes of the fire test split (stride 16, 640^2 space,
bins are quantiles of `sqrt(area)`):

| bin (px) | zero-positive GT | mean #positives |
| --- | --- | --- |
| 0-18 | **47.6 %** | 0.59 |
| 18-30 | 7.1 % | 1.95 |
| 30-42 | 0.2 % | 4.44 |
| 42-63 | 0.0 % | 6.87 |
| 63-208 | 0.0 % | 8.69 |
| >208 | 0.0 % | 8.74 |

Nearly half of all sub-18 px objects contribute **exactly zero** to all three
loss terms. They are never seen in training, and at inference no location was
ever taught to regress them.

The `mean #positives` saturates near 9 because `center_radius = 1.5` caps the
centre window at `3s x 3s`, i.e. at most 3x3 grid points regardless of how
large the box is.


## 4. Mechanism 2 -- centerness is a hard ceiling on the score

Since `score = cls.sigmoid() * ctr.sigmoid()`, we have `score <= ctr`
regardless of how confident the classifier is. And the centerness target

```
ctr = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
```

decays as the grid point moves off the box centre. The single grid point that a
small box happens to contain is almost never near its centre.

Computing, for every GT box, the best centerness achievable over its positive
locations gives a **recall ceiling at `SCORE_THR = 0.5`**:

| bin (px) | median best ctr | recall ceiling | measured DINOv3 R |
| --- | --- | --- | --- |
| 0-18 | 0.131 | 14.7 % | 4.7 % |
| 18-30 | 0.441 | 40.1 % | 23.7 % |
| 30-42 | 0.593 | 76.5 % | 63.6 % |
| 42-63 | 0.708 | 96.9 % | 84.8 % |
| 63-208 | 0.862 | 100 % | 90.8 % |
| >208 | 0.953 | 100 % | 97.2 % |

The measured recall curve is bounded by this purely geometric ceiling across
every bin. Much of the shape of `fire_size_trend.pdf` is drawn by this
multiplicative term.

A third-order effect compounds it: `loss_reg` is also weighted by the
centerness target, so the one off-centre positive a small box owns has its
regression supervision scaled down by the same factor.


## 5. How much is threshold artefact, how much is real?

Recall per bin as a function of the score threshold (fire test split, IoU 0.5,
class-aware greedy matching):

| bin (px) | DINOv3 @0.5 | @0.2 | @0.05 | FasterRCNN @0.05 | YOLO11l @0.05 |
| --- | --- | --- | --- | --- | --- |
| 0-18 | 0.047 | 0.239 | **0.386** | 0.880 | 0.801 |
| 18-30 | 0.237 | 0.579 | 0.748 | 0.885 | 0.889 |
| 30-42 | 0.636 | 0.892 | 0.917 | 0.952 | 0.939 |
| 42-63 | 0.848 | 0.940 | 0.961 | 0.953 | 0.948 |
| 63-208 | 0.908 | 0.958 | 0.977 | 0.956 | 0.951 |
| >208 | 0.972 | 0.994 | 0.999 | 0.997 | 0.988 |

Three readings:

* Dropping the threshold from 0.5 to 0.05 multiplies the smallest bin's recall
  by 8x. **Most of the `0.047` is centerness suppression plus the 0.5 plotting
  convention, not blindness.**
* `0.386` is still far below FasterRCNN's `0.880`. And
  `0.386 / 0.524 = 0.74` -- roughly three quarters of the boxes that *do* own a
  positive location get detected, while the zero-positive ones are lost almost
  entirely. Mechanism 1 converts near-directly into the recall gap.
* Above 42 px, DINOv3 @0.05 (`0.961`) **exceeds** FasterRCNN (`0.953`). The ViT
  features are not weak; they are unreadable at small scale.

The high precision in the smallest bin (0.864) says the same thing from the
other side: a small box only produces a detection when a grid point lands near
its centre, and in that case localisation is accurate. These are misses, not
false alarms -- evidence that signal is present but not extractable.


## 6. Mechanism 3 -- patchify is irreversible, and upsampling does not undo it

SAT-MTB uses `fcos_feat_stride: 8`, so mechanism 1 is largely neutralised. It
still collapses. Same analysis, bins in native px, grid in the 1024^2 input
space:

| bin (native px) | median size @1024^2 | zero-positive | median best ctr | recall ceiling | measured R |
| --- | --- | --- | --- | --- | --- |
| 0-20 | 11.6 | 13.6 % | 0.492 | **49.2 %** | **0.037** |
| 20-26 | 22.9 | 0.0 % | 0.731 | 94.5 % | 0.312 |
| 26-42 | 35.0 | 0.0 % | 0.785 | 99.7 % | 0.548 |
| 42-51 | 49.7 | 0.0 % | 0.863 | 100 % | 0.681 |
| 51-67 | 45.1 | 0.0 % | 0.871 | 100 % | 0.612 |
| 67-343 | 80.6 | 0.0 % | 0.908 | 100 % | 0.769 |

The geometric ceiling in the smallest bin is 49.2 % but only 3.7 % is realised
-- a 0.075 utilisation, against 0.32 for fire. Geometry has stopped being the
binding constraint; the features are.

The reason is that the stride-8 map is a **bilinear interpolation of the
stride-16 token map**, not a higher-resolution feature. Two adjacent stride-8
locations are read off the same tokens. It adds sampling positions, not
sub-patch information.

ViT-B/16 collapses each 16x16 pixel block into one token in its very first
layer, before any non-linearity. SAT-MTB's smallest bin has a median size of
11.6 px in the 1024^2 input -- the whole object fits inside a single patch.

Strictly, the patch embedding is `Conv2d(3, 768, k=16, s=16)`: `16*16*3 = 768`
inputs to 768 outputs, exactly rank-preserving in principle. Information is not
mathematically destroyed at that layer. But the 12 transformer blocks that
follow were trained for a self-supervised semantic objective with no pressure
to preserve sub-patch high-frequency detail, and `freeze_backbone: true` means
the backbone never reorganises its representation for the task. An object
covering 3.5 % of a patch's pixels is a small perturbation competing with the
background after LayerNorm.

The accurate statement is: **the information is not discarded, it is not
organised into a form the head can read, and the backbone is not allowed to
reorganise it.**

By contrast, FasterRCNN reaches stride 4 at P2 and YOLO stride 8 at P3; both
reach 16x downsampling gradually across four stages, each preceded by
convolutions that encode sub-pixel structure into channels, and both fine-tune
the whole backbone.


## 7. Worked example -- a 3 px object at stride 16

Monte-Carlo over 4x10^5 random placements of an `L x L` box on the grid,
replicating `inside & in_center`:

| L (px) | P(>=1 positive) | P(best ctr >= 0.5) | mean #positives |
| --- | --- | --- | --- |
| **3** | **3.5 %** | **0.7 %** | **0.03** |
| 4 | 6.3 % | 1.3 % | 0.06 |
| 8 | 25.1 % | 5.2 % | 0.25 |
| 12 | 56.2 % | 11.6 % | 0.56 |
| 16 | 100 % | 20.6 % | 1.00 |
| 24 | 100 % | 46.5 % | 2.25 |
| 32 | 100 % | 76.7 % | 4.00 |

For 100 objects of 3 px: ~96.5 produce no supervision at all. Of the 3.5 % that
do, the grid point sits near an edge, so only 0.7 % can ever clear a 0.5 score.
**Even at threshold 0, the recall ceiling is 3.5 %.**

Localisation compounds it. Two 3x3 boxes offset by 1 px along one axis already
sit at IoU 0.5, so the centre error budget is under ~1 px, i.e. `0.06` in
stride units, to be read out of a token covering 256 pixels of which the object
occupies 9.

Refining the grid only helps if the stride is *real*, not interpolated:

| true stride | P(>=1 positive), L=3 | P(ctr >= 0.5) |
| --- | --- | --- |
| 16 | 3.5 % | 0.7 % |
| 8 | 14.1 % | 2.9 % |
| 4 | 56.2 % | 11.7 % |
| 2 | 100 % | 46.5 % |
| 1 | 100 % | 100 % |

For reference, on a stride-16 grid a box needs `L >= 15.5 px` for 90 % of
instances to own a positive location, and `L >= 25.0 px` for half of them to be
able to clear a 0.5 score.


## 8. What follows

**Head-side fixes** -- fallback assignment (give every GT its nearest grid
point, ATSS-style) and decoupling `centerness` from the score -- are close to
free and would recover a large part of the 10-18 px regime, where fire has
52.4 % of boxes owning a positive but only 4.7 % recall at 0.5.

**They do nothing for 3 px.** Zero-positive is decided by sampling geometry, not
by the loss. The ceiling stays at 3.5 % however the head is written.

For objects far below the patch size the only real levers are:

1. **Real resolution.** Upsample-and-tile the input (3 px -> 16 px needs 5.3x,
   -> 25 px needs 8.3x; ViT attention is quadratic in tokens, so tiling is
   mandatory).
2. **A backbone with an early high-resolution stage.** The DINOv3 ConvNeXt
   variants are already in `HF_MODEL_DIMS` and handled by `_extract_tokens`,
   but only `last_hidden_state` (stride 32) is consumed today; taking
   intermediate stages and building an FPN is the point of a CNN trunk.
3. **Unfreeze** at least the last blocks, so patch tokens can reorganise.

Reporting note: the size-trend figures threshold at `SCORE_THR = 0.5`, which
systematically penalises a detector whose score is multiplied by centerness.
This is not unfair, but it conflates "cannot detect" with "scores low". Any
write-up of the small-object gap should either state the threshold explicitly
or add the `@0.05` curve.


## Reproducing

The per-frame dumps are produced by `evaluation/eval_fire_detect_dump.py` and
`evaluation/eval_satmtb_detect_dump.py`, writing
`$EXPERIMENT_ROOT/{fire,satmtb}_detect_dump/*_predictions.json` with GT and each
model's boxes in a common pixel space. The geometry tables above need only the
GT boxes:

```python
def probe(L, stride, center_radius=1.5, n=400_000, rng=None):
    """P(>=1 positive), P(best centerness >= 0.5) for a randomly placed LxL box."""
    x1 = rng.uniform(0, stride, n); y1 = rng.uniform(0, stride, n)   # grid is periodic
    x2, y2 = x1 + L, y1 + L
    cx, cy = x1 + L / 2, y1 + L / 2
    rad = center_radius * stride
    ks = np.arange(-1, int(np.ceil(L / stride)) + 2)
    npos = np.zeros(n, int); best = np.zeros(n)
    for a in ks:
        for b in ks:
            px, py = (a + 0.5) * stride, (b + 0.5) * stride
            l, r, t, bo = px - x1, x2 - px, py - y1, y2 - py
            v = (l > 0) & (r > 0) & (t > 0) & (bo > 0) \
                & (abs(px - cx) < rad) & (abs(py - cy) < rad)
            npos += v
            lr = np.minimum(l, r) / np.maximum(np.maximum(l, r), 1e-9)
            tb = np.minimum(t, bo) / np.maximum(np.maximum(t, bo), 1e-9)
            best = np.where(v, np.maximum(best, np.sqrt(np.clip(lr * tb, 0, 1))), best)
    return (npos >= 1).mean(), (best >= 0.5).mean(), npos.mean()
```

Substituting real GT boxes for the uniform placement reproduces the per-bin
tables in sections 3, 4 and 6.
