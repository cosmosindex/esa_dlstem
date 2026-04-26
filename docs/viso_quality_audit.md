# VISO Dataset Quality Audit

> Empirical audit of the VISO / SatVideoDT release (Yin et al., TGRS 2022)
> as distributed at `/data/ESA_DLSTEM_2025/data/trafic/VISO/`.
>
> **Bottom line:** the release has substantive annotation bugs, undocumented
> formats, and paper–release inconsistencies. The upstream GitHub repo has
> been effectively unmaintained since 2021: 7 issues are open and not a single
> one has a maintainer reply. Use VISO for qualitative reference only; do not
> report raw MOT / SOT numbers on the test split without filtering.

Date of audit: 2026-04-24
Data source: `/data/ESA_DLSTEM_2025/data/trafic/VISO/` (extracted from official RARs)
Upstream: https://github.com/QingyongHu/VISO

---

## 1. What VISO actually ships

Full enumeration of every non-image file (from both the extracted directory tree
and the four distributed RAR archives, verified via `unrar lb`):

| Path | Schema | Contains attributes? |
|---|---|---|
| `mot/*/*/gt/gt.txt` | MOT16: `frame,id,x,y,w,h,conf=1,cls,-1,-1,-1` — last 3 cols always `-1` | No |
| `mot/car/*/gt/gt_centroid.txt` | same as above but centroid instead of xy | No |
| `sot/*/*/gt/<id>_<start>_<end>.txt` | plain `x,y,w,h` per frame | No |
| `coco/*/Annotations/instances_*.json` | keys = `{bbox, segmentation, area, iscrowd=0, ignore=0, image_id, category_id, id}`; image keys = `{id, file_name, width, height}`; `info` field is `None` | No |
| `coco/*/Annotations/<split>/*.xml` | same as VOC XML | No |
| `voc/*/Annotations/*.xml` | full tag set = `{annotation, folder, filename, path, source/{database,annotation,image,flickrid}, size/{width,height,depth}, segmented, object/{name, pose=Unspecified, truncated=0, difficult=0, bndbox}}` — `truncated` and `difficult` are always `0` | No |
| `voc/*/ImageSets/Main/*.txt` | train/val/test image-id lists | No |

Swept every unique XML tag across 2000 VOC files and every annotation-key union
across the 4 COCO JSONs. There is **no** `attribute`, `tag`, `challenge`,
`occlusion`, `visibility`, `color_change`, `LR`, `SOB`, etc. field anywhere in
the release.

### 1.1 There are no attribute annotations in VISO

Claims that VISO provides per-sequence or per-frame attributes (e.g. "color
change", "Low Resolution", "Similar Object") are not supported by the
distribution. Those attribute names belong to **other** satellite SOT datasets:

- `SOB` (Similar Object) — SatSOT (`datasets/satsot.py:40`)
- `SA` (Similar Appearance) — OOTB (`datasets/ootb.py:40`)
- `DS` (Dense Similarity) — SV248S (`datasets/sv248s.py:53`)
- `LR` / `Color Change` — not defined in any of the SOT datasets used in
  this repo

---

## 2. Upstream issue tracker: 7 open, 0 maintainer replies

The repo has **not received a single maintainer response** since it opened in
Dec 2021. All 7 issues are still open.

| # | Author | Date | Subject |
|---|---|---|---|
| #1 | camilo-aguilar | 2021-12 | Google Drive link needs a download code |
| #2 | UncleSan | 2022-03 | Baseline-method source code not provided |
| #3 | ccrutchf | 2022-06 | Request for video descriptions / documentation |
| #4 | Mr-IronMan | 2022-06 | **"Labeling issues with SOT"** — attached two screenshots of bad labels, asked for a fix |
| #5 | wht-bupt | 2022-09 | **Kalman Filter MOTA = 73.6 in paper Table IX, = 5.6 on README** — which is correct? |
| #6 | zhaoxingle | 2022-10 | **SOT test seqs 24-27: first frames have no object, mid-sequence annotations become all-zero, some sequences are entirely empty** |
| #7 | YuTingShi123 | 2024-03 | **What do the columns of `gt.txt` mean?** (no documentation anywhere) |

Four of these (`#4`, `#5`, `#6`, `#7`) are direct complaints about data
quality, paper–release inconsistency, or missing documentation.

---

## 3. Empirical verification of Issue #6 (and worse)

I independently swept `sot/car/{024,025,026,027}/gt/*.txt` (robust parsing,
tolerant of both comma- and whitespace-separated rows):

| Sequence | Track files | First row = `[0,0,0,0]` (SOT init broken) | Tracks with any all-zero row | Total all-zero rows |
|---|---:|---:|---:|---:|
| `car/024` | 212 | 2 | 28 | 816 |
| `car/025` | 113 | 3 | 51 | 732 |
| `car/026` | 157 | 6 | 20 | 881 |
| `car/027` | 123 | 4 | 20 | 2063 |
| **Total** | **605** | **15** | **119** | **4,492** |

Implications:
- **15 tracks have an all-zero first frame.** In the standard SOT protocol the
  first-frame box is the tracker's init; `[0,0,0,0]` makes those tracks
  unusable.
- **119 tracks contain at least one `[0,0,0,0]` row**. VISO has no visibility /
  out-of-view flag, so these zeros are ambiguous: could be occlusion, could be
  annotation gap, **cannot be disambiguated from the release alone**.

### 3.1 Beyond Issue #6: malformed rows

Some track files are not merely sparse — they are **corrupt at the byte level**.
Example: `sot/car/026/gt/114_17_112.txt` (filename claims track 114, frames
17–112 = 96 frames). First rows:

```
,27,488,7,5
,26,488,7,5
,25,488,7,4
,24,487,7,5
...
,10,484,9,5      <- exact same line repeated ~20 times in a row
,10,484,9,5
,10,484,9,5
...
0,0,0,0          <- 14 consecutive all-zero rows
0,0,0,0
...
,9,484,8,6       <- recovery, still missing x
```

Two distinct bugs:

1. **First column missing.** Lines start with `,27,488,7,5` not `27,488,7,5`;
   the `x` coordinate is blank, so the file is neither the documented
   `x,y,w,h` format nor any other standard. Likely a broken export on the
   author side that went out unreviewed.
2. **Dozens of identical consecutive rows** — a car in Jilin-1 satellite video
   cannot stay pixel-locked for ~40 frames at 10 fps. Strong signal of a
   label-propagation bug or stale rows not refreshed per frame.

`sot/car/027/gt/10_1_331.txt` is worse: filename claims 331 frames, file has
328 rows, **298 of them are `[0,0,0,0]`**; the few non-zero rows at the tail
again miss the first column.

### 3.2 Track-level consistency spot-check (`car/001`)

As a sanity check on a training-split sequence that is reportedly dense:

- 31,836 boxes, 186 tracks, 260 frames, mean 122.5 objects/frame.
- **10 degenerate boxes** (w ≤ 0 or h ≤ 0).
- 13 / 186 tracks have frame gaps inside their lifespan (track disappears
  and reappears) — 84 missing frames total.
- 11 adjacent-frame jumps exceed 5× object size — either GT errors or very
  fast small objects (plausible for this dataset scale, but worth flagging).
- Track-length median = 183 frames; min 9, max 260; no tracks under 5 frames.

The training split is much cleaner than the test split, but still not bug-free.

---

## 4. Paper ↔ release inconsistencies

### 4.1 MOT test split annotations are sparse to single-track

Verified by counting unique `obj_id` values per `mot/<cat>/<seq>/gt/gt.txt`:

| Sequence | Format | Frames | Tracks |
|---|---|---:|---:|
| `car/001..024` (train) | comma-delim | 260–481 | 151–212 (dense MOT) |
| `car/025` (val) | comma-delim | 480 | 113 |
| `car/028..038` (val→test) | **space-delim** | 197–358 | **1** |
| `plane/044` (test) | space-delim | 500 | 2 |
| `ship/047` (test) | comma-delim | 300 | 1 |

So **the `mot/` test split is not actually multi-object** — each held-out
sequence has exactly one (rarely two) track, despite the paper framing VISO as
a MOT benchmark. The `sot/` folder mirrors this: `sot/car/001/gt/` has 186
per-track files, `sot/car/029/gt/` has 1.

Practical consequence: any "MOT" score reported on the VISO test split is
effectively a SOT score, and metrics like MOTA / IDF1 lose their meaning.

### 4.2 Tracker class per sequence vs. scene contents

Folder naming (`car/`, `plane/`, `ship/`, `train/`) is treated by our loader
(`datasets/viso.py:157`) as the single class for every annotation in that
sequence. But the underlying satellite frames typically show multiple classes
simultaneously (e.g. a port with ships + cars + planes). VISO's annotators
only labelled the folder class; cross-class objects in the same frame are
**not annotated**.

This means that for a 4-class detector trained on VISO:
- Training treats genuine cross-class objects as negatives, which hurts recall.
- Evaluation counts correct cross-class detections as false positives,
  depressing AP for classes other than the folder class.

### 4.3 README vs paper: MOT results disagreement (Issue #5)

Per Issue #5 (2022-09, no reply): paper Table IX reports Kalman Filter MOTA =
**73.6**; README reports the same method at **5.6**. A ~13× discrepancy on
the benchmark's headline metric, unaddressed for 3+ years.

### 4.4 No `gt.txt` documentation (Issue #7)

A 2024 user asked what the columns of `gt.txt` mean. The release contains **no
README, no data dictionary, no format spec**. The MOT16 convention is
recognizable to practitioners, but the `r1, r2, r3` trailing columns
(always `-1`) are never explained. Issue remains open.

---

## 5. Recommended handling in this repo

Short version: treat VISO as a **qualitative dataset**, not a benchmark.

1. **Do not report MOT metrics (MOTA / IDF1 / HOTA) on the VISO test split.**
   Each test sequence has ≤ 2 annotated tracks; metrics are ill-defined.
2. **If SOT evaluation is needed**, use only the training-split sequences
   (`car/001..024`, `plane/039..042`, `ship/045`, `train/046`) and run a
   quality pre-filter (drop tracks where the first row is `[0,0,0,0]`, drop
   tracks where ≥ N % of rows are all-zero, drop any file with malformed
   rows). Document the exclusion count in the paper.
3. **Do not compute per-class detection metrics (AP per class) on VISO.** The
   one-class-per-folder labelling guarantees systematic false positives for
   classes other than the folder class.
4. **Cite the quality issues explicitly.** A footnote of the form:
   > "VISO's test-split annotations are sparse (1–2 tracks per sequence) and
   > contain known bugs (see upstream issues #4, #6); results on VISO are
   > reported for qualitative reference only."
5. **Ignore any claim of per-frame attribute labels on VISO** — there are
   none in the release. Any attribute-wise plot on VISO must be based on
   attributes the user generates themselves (manual labelling or automatic
   derivation from bboxes / images).

### 5.1 Suggested automated audit script

Not yet written; would live at `tools/viso_quality_audit.py` and emit a CSV
per track with columns: `video_id, track_id, n_frames_claimed,
n_rows_present, n_all_zero, n_malformed_rows, n_duplicate_rows,
first_row_all_zero, verdict ∈ {ok, warn, drop}`. Downstream,
`datasets/viso.py` could consume that CSV to filter bad tracks at load time.

---

## 6. Reproduction

### Count all-zero rows on SOT test sequences
```bash
python3 -c "
import os, glob, re
for seq in ['024','025','026','027']:
    d = f'/data/ESA_DLSTEM_2025/data/trafic/VISO/sot/car/{seq}/gt'
    zero_first = any_zero = total_zero = 0
    for fp in glob.glob(f'{d}/*.txt'):
        lines = [l.strip() for l in open(fp) if l.strip()]
        rows = []
        for l in lines:
            parts = re.split(r'[,\s]+', l)
            if len(parts) >= 4:
                try:    rows.append([float(x) for x in parts[:4]])
                except: pass
        if rows and rows[0] == [0,0,0,0]: zero_first += 1
        z = sum(r == [0,0,0,0] for r in rows)
        if z: any_zero += 1; total_zero += z
    print(f'{seq}: first-row-zero={zero_first}  any-zero={any_zero}  total-zero-rows={total_zero}')
"
```

### Count per-sequence MOT track density
```bash
for cat in car plane ship train; do
  for f in /data/ESA_DLSTEM_2025/data/trafic/VISO/mot/$cat/*/gt/gt.txt; do
    first=$(head -1 "$f")
    if [[ "$first" == *","* ]]; then delim=comma
    else                             delim=space
    fi
    if [ "$delim" = comma ]; then
      tracks=$(awk -F',' '{print $2}' "$f" | sort -nu | wc -l)
    else
      tracks=$(awk       '{print $2}' "$f" | sort -nu | wc -l)
    fi
    seq=$(basename $(dirname $(dirname "$f")))
    printf "%s/%s  delim=%-5s  tracks=%d\n" "$cat" "$seq" "$delim" "$tracks"
  done
done
```

### Dump the extreme case (`car/026` track 114)
```bash
cat /data/ESA_DLSTEM_2025/data/trafic/VISO/sot/car/026/gt/114_17_112.txt
```
