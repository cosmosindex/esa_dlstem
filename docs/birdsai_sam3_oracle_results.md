# BIRDSAI — SAM3 oracle probes vs trained detectors

**2026-06-18.** Wild-animal thermal MOT. Question: on BIRDSAI's tiny (13–35 px)
thermal objects, is the failure a *detection* problem or a *tracking* problem?

All numbers below are on the **16 BIRDSAI test videos**, scored at **fine 5-class,
IoU 0.5, class-aware greedy matching** — identical to `eval_birdsai_detect_track.py`,
so the SAM3 rows are directly comparable to the three trained detectors.

## Methods

- **DINOv3 / FasterRCNN / YOLO11l** — trained fine-5-class detectors + per-class
  OC-SORT (best ckpt, 2026-06-17 retrain). Fair MOT.
- **SAM3 Exp1 `gt_init`** — each GT track initialised with its GT box at its
  first-appearance frame, then SAM3 (SAM2-style mask memory) propagates it.
  **Oracle / upper bound** (test GT used as init). = tracker ceiling.
- **SAM3 Exp2 `exemplar_detect`** — at each 32-frame clip's first frame, every GT
  box of a class is looped as a single visual exemplar through SAM3's find head;
  union+dedup of detected boxes seeds the SOT propagation. **Oracle** (exemplar
  boxes are test GT). = exemplar-self-bootstrapped ceiling.

Script: `evaluation/eval_birdsai_sam3_oracle.py --mode {gt_init,exemplar_detect}`.

## Results — F1 (per fine class)

| class (size) | DINOv3 | FRCNN | YOLO | SAM3 Exp2 (exemplar) | SAM3 Exp1 (GT-init) |
|---|---|---|---|---|---|
| human 35 px (fast-moving) | 0.028 | 0.071 | 0.076 | 0.414 | **0.640** |
| elephant (big) | 0.596 | 0.526 | 0.472 | 0.587 | **0.700** |
| giraffe 18 px | 0.000 | 0.015 | 0.028 | 0.616 | **0.894** |
| lion 14 px | 0.000 | 0.000 | 0.000 | 0.553 | **0.783** |
| unknown ~17 px | 0.000 | 0.013 | 0.000 | 0.216 | **0.540** |
| **OVERALL** | 0.425 | 0.364 | 0.334 | **0.504** | **0.681** |

OVERALL MOTA: DINOv3/FRCNN/YOLO ≈ 0.0–0.25; **Exp1 +0.322**, **Exp2 −0.168**
(exemplar detection floods FP — precision 0.447; unknown worst at MOTA −1.59).

## Conclusions

1. **Detection is the entire bottleneck, not tracking.** The three trained
   detectors score ≈0 on every small species (giraffe / lion / unknown). The
   *same* objects, once given a GT box, are tracked by SAM3 at **0.54–0.89**.
2. **Even imperfect exemplar bootstrapping beats trained detectors.** Exp2 OVERALL
   F1 **0.504 > best detector 0.425**: a few visual exemplars + SAM3 mask
   propagation recover more small objects than fully-trained specialist
   detectors — at the cost of precision (FP flood → negative MOTA).
3. **The tracking ceiling itself depends on motion, not just size.** `human`
   (35 px but fast) stays at 0.640 even under GT-init — SAM2 mask memory drifts
   off fast small humans, unlike the slow giraffe (0.894) / lion (0.783).

## Caveats

- Exp1 and Exp2 are **oracle upper-bound rows** (test GT reaches the model as
  init / exemplar) — methodologically the same class as the MOT Exp2 GT-box
  oracle, **not fair MOT rows**. They bound what is achievable, they do not rank
  SAM3 against the detectors as a deployable system.
- SAM3 has no cross-image exemplar API in this build; train-set exemplars are
  impossible, so any exemplar/init box is necessarily from a test frame.
- Diagnostic probe scripts: `evaluation/_sam3_exemplar_smoke.py`,
  `_sam3_sot_oracle_smoke.py`, `_birdsai_coarse_compare.py`,
  `_birdsai_score_at_detection.py`.
