# Static objects are missing from the MOT ground truth

Outside AIR-MOT, the Space-tracker MOT datasets annotate only *moving*
objects. Static ones — parked aircraft, moored ships, stabled trains — are
visible in frame but carry no label, so every detector that finds them is
charged a false positive it has no way to avoid.

This directory quantifies the gap and scopes what, if anything, is worth
re-annotating.

## Evidence 1 — track displacement (29,135 GT tracks, all 5 MOT datasets)

Net displacement of each GT track's centre from first to last frame,
normalised by the object's own `sqrt(area)`. A GT that covers static objects
has tracks near zero; a movers-only GT does not.

| dataset | category | tracks | static (<0.5) | net p50 | net p90 |
|---|---|---:|---:|---:|---:|
| airmot | airplane | 216 | **81.5%** | 0.15 | 1.17 |
| airmot | ship | 993 | **26.8%** | 1.11 | 2.91 |
| satmtb | airplane | 158 | 9.5% | 3.01 | 14.30 |
| satmtb | ship | 275 | 2.2% | 9.25 | 27.26 |
| satmtb | train | 28 | 21.4% | 1.26 | 9.50 |
| satmtb | car | 9,143 | 4.9% | 23.14 | 70.23 |
| viso | airplane / ship / train | 14 | 0.0% | 6.2–7.8 | — |
| sdmcar | car | 13,707 | 3.0% | 24.47 | 62.91 |
| rscardata | car | 4,601 | 1.6% | 24.89 | 82.85 |

AIR-MOT is the only dataset whose GT is full of stationary objects. Everywhere
else the GT is movers-only.

## Evidence 2 — SAT-MTB detection XML vs MOT CSV (1,261 detection tracks)

SAT-MTB ships per-frame detection XML *and* MOT CSV for airplane / ship /
train, so the two can be diffed directly. (`car` has no detection XML — its
MOT CSV is the only source and cannot be cross-checked.) Matching is per
category at IoU >= 0.5; a detection track counts as "in MOT" when the MOT GT
covers >= 50% of its frames.

| category | det tracks | in MOT | missing | missing % | net disp (in MOT) | net disp (missing) | size (in MOT) | size (missing) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| airplane | 859 | 142 | 717 | **83.5%** | 3.03 | **0.04** | 45.8 | 45.4 |
| ship | 378 | 257 | 121 | **32.0%** | 9.55 | **0.30** | 11.8 | 24.6 |
| train | 24 | 20 | 4 | 16.7% | 1.15 | **0.00** | 146.3 | 94.7 |

The missing tracks do not move (0.00–0.30 of their own size, i.e. annotation
jitter) while the retained ones travel 1–10 body lengths. They are **not**
small — missing aircraft are the same size as retained ones, and missing ships
are twice as large. The MOT GT is filtered purely on motion.

### What that costs a model

A detector that finds every object in a SAT-MTB airplane sequence scores at
most **16.5% precision** against the MOT GT; 83.5% of its correct detections
are counted as false positives. The ceiling is 68% on ship and 83% on train.

## How far do the SAT-MTB detection annotations actually reach?

The recovery plan above is only as good as the detection-XML coverage, so it
was audited per sequence.

| dir category | MOT sequences | with `det/HBB` | coverable |
|---|---:|---:|---:|
| airplane | 62 | 62 | **100%** |
| ship | 70 | 70 | **100%** |
| train | 13 | 7 | 54% |
| car | 92 | 0 | 0% |
| **total** | **237** | **139** | **59%** |

Two things this table hides:

- The 59% figure is dominated by `car`, which is out of scope by design.
  Against the sequences the plan actually targets — the 145 non-car ones —
  coverage is **139/145 = 95.9%**.
- Where coverage exists it is complete: every one of the 139 sequences has one
  XML per frame, no sparse or keyframe-only annotation.

The gap is entirely in `train`: `01–07` have both det and MOT; `08–10` have det
but no MOT (they are not in the MOT manifest at all); `11–16` have MOT but no
det and therefore **cannot be completed from existing labels**. That gap is
small in practice — those six sequences carry 1–2 GT tracks each, and train's
MOT GT was never filtered on motion the way airplane's was (5 of its 13
sequences already contain static tracks). Only 3 of the 23 train detection
tracks are missing from MOT.

`det/HBB`'s `objectID` is a genuine cross-frame track id, not a per-frame
index: across 1,261 detection tracks the median is 283 observations and exactly
one track (0.1%) is single-frame. Recovered static objects therefore come with
usable track ids, no re-association needed.

**Sequences whose detection tracks are entirely absent from MOT: 1 of 139**
(`satmtb/ship/39`, 18 det tracks, 8 of them static). The problem is not that
whole videos were skipped — it is that movers were kept and non-movers dropped,
consistently, within otherwise-annotated videos.

### Bonus

`train/08–10` have detection annotations but no MOT GT, which is why they are
absent from the manifest. Since `objectID` is a real track id, they could be
promoted to three additional MOT sequences at zero annotation cost.

## Scope: what is worth fixing

| dataset | categories | status |
|---|---|---|
| AIR-MOT | airplane, ship | already annotates static objects — nothing to do |
| SAT-MTB | airplane, ship | **717 + 121 static tracks recoverable from `det/HBB` XML at zero human cost; det coverage is 62/62 and 70/70 sequences** |
| SAT-MTB | train | marginal: only 4 missing tracks, det covers 7/13 MOT sequences, and its MOT GT already contains static objects |
| SAT-MTB, SDM-Car, RsCarData | car | out of scope, see below |
| VISO | airplane, ship, train | COCO detection annotations have identical density to MOT (1.19 / 1.61 / 1.00 boxes per image) — they are movers-only too, so no free labels. Only 9 sequences. |

### Why `car` is deliberately excluded

Median car size is 4.9–6.5 px across SAT-MTB / RsCarData / SDM-Car. A
stationary car at that scale is not separable from road texture in a single
frame — which is precisely why this line of work (VISO, HiEUM, DSFNet) is
posed as *moving object* detection in the first place. Annotating static cars
would change the research question, invalidate comparison against every
moving-object baseline, and cost enormously (SDM-Car already averages 48
moving cars per frame).

## Restricted to the small bucket

The 412-sequence small bucket has a very different composition from the corpus
as a whole, so the plan has to be re-scoped against it.

| source | small seqs | GT protocol | action |
|---|---:|---|---|
| SDM-Car + RsCarData + SAT-MTB `car` | **268** | movers-only | none — out of scope by design |
| AIR-MOT | 45 | **all-object already** | none |
| SAT-MTB `airplane` (25) + `ship` (63) | 88 | movers-only | **recoverable, 100% det coverage** |
| SAT-MTB `train` | 7 | movers-only | only 1 of 7 has det |
| VISO | 4 | movers-only | no det available |

Detection coverage inside the small bucket is 89/187 SAT-MTB sequences (47.6%)
— but again the shortfall is `car` (0/92 by construction). Among non-car small
sequences it is 89/95 = **93.7%**.

The gap is just as severe inside the small bucket as outside it:

| bucket | category | det tracks | missing from MOT | missing % |
|---|---|---:|---:|---:|
| small | airplane | 392 | 332 | **84.7%** |
| small | ship | 337 | 97 | **28.8%** |
| small | train | 5 | 1 | 20.0% |
| large | airplane | 467 | 385 | 82.4% |
| large | ship | 41 | 24 | 58.5% |
| large | train | 19 | 3 | 15.8% |

Being small does **not** make a sequence's annotation more complete. Static
aircraft are dropped at the same rate in both buckets.

### The protocol-consistency problem this exposes

The small bucket already mixes two annotation protocols: AIR-MOT's 45
sequences label static objects, the other 367 do not. Completing the 88
SAT-MTB airplane / ship sequences would move them into AIR-MOT's protocol and
leave the 268 car sequences in the other one — so a single headline score over
"the small bucket" would average two incompatible definitions of ground truth
either way. The split therefore has to be reported per protocol (or per class),
not as one number. This is a pre-existing defect of the benchmark, not
something completion introduces.

## Interaction with the size split

This bites hardest exactly where [`../size_split`](../size_split) puts its
large bucket: 35 of the 44 large-bucket MOT sequences are airplane — the
category missing 83.5% of its static annotations. Left unfixed, "models do
worse on the large bucket" reads as a size effect when it is really an
incomplete-GT effect.

## Regenerating

```bash
python tools/analyze_static_objects.py --out-dir docs/static_annotation
```

Outputs `track_displacement.csv` (per GT track) and `satmtb_det_vs_mot.csv`
(per detection-XML track, with `in_mot` and `net_disp_ratio`).
