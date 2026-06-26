# SAM3 + RAFT static-tracklet filter — experimental setup

Source of truth for the writeup. Mirrors the actual values used at run
time. Cross-references: `eval_sam3.py`, `tools/raft_filter_tracklets.py`,
`compute_hota.py`, `run_sam3_raft_mot.sh`,
`configs/MOT/sam3_{viso_no_car,rscardata,airmot,satmtb,sdmcar}.yaml`.

---

## 1. Datasets (test split only)

All datasets are satellite (Jilin-1 / Luojia / VISO-source) videos.
Split assignment is the official mapping where one exists; otherwise a
`seed=42` shuffle (see per-dataset class docstrings).

| Dataset       | # test seqs | Resolution         | Classes evaluated                  | Source root |
|---------------|------------:|--------------------|------------------------------------|-------------|
| viso_no_car   |  2          | varies (per-seq)   | plane, ship, train                 | `/data/ESA_DLSTEM_2025/data/trafic/VISO`         |
| rscardata     |  7          | 1024 × 1024        | car                                | `/data/ESA_DLSTEM_2025/data/trafic/RsCarData`    |
| airmot        |  7          | varies             | airplane, ship                     | `/data/ESA_DLSTEM_2025/data/trafic/AIR-MOT-100`  |
| satmtb        | 61          | varies             | airplane, car, ship, train (coarse)| `/data/ESA_DLSTEM_2025/data/trafic/SAT-MTB`      |
| sdmcar        | 20          | varies             | car                                | `/data/ESA_DLSTEM_2025/data/trafic/SDM-Car`      |
| viso_combined |  9          | mixed              | plane, ship, train, car            | viso_no_car ∪ rscardata, pooled via TrackEval     |

Notes:
- `viso_no_car` is `VISODataset(categories=["plane","ship","train"])`
  on the official COCO/VOC test split — i.e. plane/044 + ship/047.
- `rscardata` test split = HiEUM's 7 sequences in `test1024/` (paper
  protocol `dataNum=[2,3,5,6,8,9,10]`).
- `viso_combined` is constructed at HOTA time via
  `compute_hota.py --combine-viso`, which materialises both halves
  into a single TrackEval benchmark (sequence names are disjoint, no
  collision possible).
- `satmtb` is restricted to coarse classes (4) here. The on-disk fine
  category labels are not used for SAM3 evaluation.

## 2. SAM 3 / SAM 3.1 inference

Wrappers: `models/sam3.py::SAM3TextTracker` (SAM 3 base) and
`models/sam3.py::SAM31TextTracker` (SAM 3.1 multiplex).
Driver: `eval_sam3.py`. The wrapper is selected at runtime by
`cfg.model_version` (`sam3` / `sam3.1`); the rest of the
hyperparameter table is shared between the two.

| Hyperparameter                | Value                          |
|-------------------------------|--------------------------------|
| SAM 3 weights                 | `facebook/sam3` (HF, gated; `sam3_checkpoint_path: null` → auto-download) |
| SAM 3.1 weights               | `facebook/sam3.1` (`sam3.1_multiplex.pt`, HF, gated)             |
| SAM 3 builder                 | `sam3.model_builder.build_sam3_video_model`                       |
| SAM 3.1 builder               | `sam3.model_builder.build_sam3_predictor(version="sam3.1")` (multiplex predictor; bf16 autocast, tf32) |
| Prompt mode                   | text (`tracker_type=text`, `prompt_strategy=text`)                |
| Text prompt                   | dataset's dominant category string per video (`set_text_prompt(clip.category)`) |
| Apply temporal disambiguation | true (SAM 3); always-on for SAM 3.1                              |
| SAM 3.1 multiplex bucket size | 16 (`sam31_multiplex_count: 16`)                                  |
| SAM 3.1 max objects           | 16 (`sam31_max_num_objects: 16`)                                  |
| `torch.compile` (SAM 3.1)     | off (`sam31_compile: false`) — avoid first-run compile latency   |
| Input resolution              | native (no resize, `img_size: null`)                              |
| Clip length                   | 32 frames                                                         |
| Clip stride                   | 1                                                                 |
| Batch size                    | 1                                                                 |
| `num_workers`                 | 0 (fork+CUDA deadlock)                                            |
| Cross-clip ID stitching       | IoU-greedy match between clip-1 frame-T boxes and clip-2 frame-0 boxes; matched → reuse global id, else allocate new (see `_stitch_text_track_ids` in `lightning_modules/video_tracker_module.py`) — **same logic for both versions** |
| Per-frame outputs kept        | all (`mot_dump_score_thresh: 0.0`)                                |
| GT box leakage                | none (open-vocabulary, prompt-text only)                          |
| Run-dir prefix                | `sam3_text_<slug>_<TS>` (SAM 3) / `sam3p1_text_<slug>_<TS>` (SAM 3.1) |

The MOT dump (per-video MOTChallenge text files) is produced by
`MOTFormatDumpCallback` and contains every emission SAM3 makes — no
score gating is applied at this stage so the RAFT filter sees the full
track set.

## 3. RAFT optical-flow filter

Tool: `tools/raft_filter_tracklets.py`. RAFT repo:
`github.com/princeton-vl/RAFT`, vendored at `RAFT/`.

| Hyperparameter           | Value                                                    |
|--------------------------|----------------------------------------------------------|
| Variant                  | RAFT (large), `small=False`, no mixed precision          |
| Pretrained weights       | `raft-things.pth` (Sintel-style, pretrained on FlyingThings3D) at `/work/anon/checkpoints/raft/` |
| Refinement iterations    | 20 (`iters=20`, `test_mode=True`)                        |
| Frame pairing            | consecutive: `F_t = flow(I_{t-1} → I_t)`                 |
| First-frame fallback     | `F_0 = flow(I_0 → I_1)` (so each tracklet gets ≥1 sample)|
| Padding                  | `InputPadder` (round to multiple of 8)                    |
| Per-frame, per-box score | **median** of `‖F_t‖` over pixels strictly inside the box |
| Per-tracklet aggregator  | **80th percentile** (`p80`) over all per-frame medians   |
| Threshold τ              | 0.5 pixels per frame (default; sweepable)                |
| Decision rule            | keep tracklet iff `p80 > τ`                              |
| Coordinate space         | original-image pixels                                    |
| Cache                    | `<run_dir>/raft_track_motion.json` (per-tracklet per-frame medians) so re-running with new τ skips RAFT |

Why median-over-pixels: the box edges spill onto background; mean is
biased toward the static background, median is robust to ~50%
mislabelled pixels.

Why 80th percentile across the lifetime: max is sensitive to single
frame spikes (e.g. shadow-edge flow noise); 80th percentile captures
"the tracklet moves on most frames" while still tolerating a fraction
of low-motion frames.

## 4. HOTA evaluation

Tool: `compute_hota.py`. Library: TrackEval 1.3.0
(`MotChallenge2DBox` benchmark).

| Setting                  | Value                                                    |
|--------------------------|----------------------------------------------------------|
| Metrics                  | HOTA, DetA, AssA, LocA (HOTA package); MOTA, MOTP, IDsw, MT, ML, CLR_TP/FP (CLEAR); IDF1 (Identity) |
| `CLASSES_TO_EVAL`        | `["pedestrian"]` — TrackEval 's single-foreground-class slot. Every retained track is mapped to this class, so multi-class datasets are evaluated *class-agnostically* (HOTA does not separate per category here). |
| `DO_PREPROC`             | False (no MotChallenge-specific filtering of small / static GT) |
| `BENCHMARK`              | one of {viso_no_car, rscardata, airmot, satmtb, sdmcar, viso_combined} |
| `SPLIT_TO_EVAL`          | `test`                                                  |
| Sequence map             | written per-benchmark to `_hota_workspace/seqmaps/<bench>-test.txt` |
| GT format                | MOTChallenge `frame,id,x,y,w,h,1,1,1.0` (conf=1, cls=1, vis=1) — written by `_write_gt` from each dataset's `_load_annotations`. |
| Per-(frame,id) dedup     | yes (TrackEval rejects duplicates; SDM-Car ships some)  |
| Negative GT track ids    | dropped (SDM-Car's `tid=-1` "no track" sentinel)        |
| Frame-id offset          | per-seq `1 - min(frame_ids)` so timesteps are 1-indexed (SDM-Car ships 0-indexed) |

Each (dataset, tracker) pair is evaluated in its own
`Evaluator.evaluate()` call so a TrackEval edge case on one sequence
doesn't kill the rest of the sweep
(`USE_PARALLEL=False`, `BREAK_ON_ERROR=False`, `RETURN_ON_ERROR=True`).

## 5. Reported rows

For every dataset in `{viso_no_car, rscardata, airmot, satmtb,
sdmcar}` the CSV gets two rows:
- `sam3_text` — raw SAM3 (reads `mot_format/<seq>.txt`)
- `sam3_text_raft` — after the RAFT filter (reads
  `mot_format_filtered/<seq>.txt`)

When `--combine-viso` is enabled (default in the driver), an
additional benchmark `viso_combined` is materialised that pools the
sequences of viso_no_car and rscardata into a single TrackEval bench;
both rows (`sam3_text`, `sam3_text_raft`) are reported there too.

CSV columns: `dataset, tracker, HOTA, DetA, AssA, LocA, MOTA, MOTP,
IDF1, IDsw, MT, ML, n_dets`.

## 6. Compute / environment

| Item                  | Value                                          |
|-----------------------|------------------------------------------------|
| Hardware              | 2× NVIDIA RTX 5000 Ada (sm_89), 32 GiB each   |
| CUDA allocator        | `PYTORCH_ALLOC_CONF=expandable_segments:True`  |
| Python env            | `micromamba run -n esa_dlstem`                 |
| TrackEval version     | 1.3.0                                          |
| Float32 matmul        | `torch.set_float32_matmul_precision("high")`   |

## 7. Output layout

```
/data/ESA_DLSTEM_2025/experiments/MOT/sam3_raft_filtered_<TS>/
  sam3_text_<bench>_<TS>/
    mot_format/                <seq>.txt        # raw SAM3
    mot_format_filtered/       <seq>.txt        # after RAFT
    raft_track_motion.json                     # per-tracklet motion cache
    raft_filter_summary_p80_tau0.5.json        # per-tracklet keep/drop decisions
    per_image_metrics.json                     # per-frame TP/FP/FN
    test_metrics.json                          # run-level summary
    visualizations/<seq>_frameNNNN.jpg
  _hota_workspace/                             # TrackEval scratch (kept for inspection)
  hota_summary.csv                             # the table you cite
```

## 8. Reproducibility one-liner

```bash
# SAM 3 baseline (output: .../sam3_raft_filtered_<TS>/)
bash run_sam3_raft_mot.sh

# SAM 3.1 multiplex (output: .../sam3p1_raft_filtered_<TS>/)
bash run_sam3p1_raft_mot.sh

# Subsetting / re-thresholding (works the same way for both drivers):
DATASETS="viso_no_car rscardata" bash run_sam3_raft_mot.sh
TAU=0.3 SKIP_SAM3=1 bash run_sam3p1_raft_mot.sh   # cache hit, only re-filters
```
