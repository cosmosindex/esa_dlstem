---
name: dataset-resolution
description: Analyze image/video resolutions in a dataset and write a statistics report to docs/. Use when you need to understand the spatial resolution distribution of a dataset.
argument-hint: [DatasetName] [DatasetRootPath]
allowed-tools: Read, Glob, Grep, Bash, Write, Edit
---

# Dataset Resolution Analysis: $ARGUMENTS

Analyze image resolutions for the given dataset and produce a markdown report in `docs/`.

## Inputs

The user provides:
1. **Dataset name** — used for the output filename and report title.
2. **Dataset root path** — the directory containing the dataset.

If the user only provides a name, check whether a dataset class already exists in `datasets/` to find the root path, or ask the user.

## Step 1: Understand the dataset layout

Before scanning images, read the dataset class file (`datasets/<name>.py`) if it exists. This tells you:
- How sequences/videos are organized (subdirectories, flat, etc.)
- Where image files live (e.g. `<seq>/img/*.jpg`, `<seq>/*.png`)
- How categories are derived from directory names
- Any files to skip (metadata JSON, hidden files like `._*`)

If no dataset class exists, explore the directory manually:
```bash
ls <root>/
ls <root>/<first_subdir>/
```

## Step 2: Scan resolutions

Write and run a Python script using the `esa_dlstem` micromamba environment. The script must:

1. Iterate over all sequences/videos in the dataset.
2. For each sequence, read the **first frame only** (sufficient for resolution since frames within a sequence share the same resolution).
3. **Skip hidden files** (filenames starting with `.`) — macOS `._` resource forks are common in transferred datasets.
4. Record `(width, height)` per sequence, along with the category name (derived from directory structure or filename pattern).
5. Print results to stdout for capture.

### Script template

```python
micromamba run -n esa_dlstem python3 -c "
import cv2, os, re
import numpy as np
from collections import defaultdict

root = '<DATASET_ROOT>'
widths, heights = [], []
cat_res = defaultdict(list)
resolutions = defaultdict(list)

for seq in sorted(os.listdir(root)):
    img_dir = os.path.join(root, seq, '<IMG_SUBDIR>')  # e.g. 'img', or just seq itself
    if not os.path.isdir(img_dir):
        continue
    frames = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    if not frames:
        continue
    img = cv2.imread(os.path.join(img_dir, frames[0]))
    if img is None:
        continue
    h, w = img.shape[:2]
    widths.append(w); heights.append(h)
    cat = re.sub(r'_\d+$', '', seq)  # adjust category extraction as needed
    cat_res[cat].append((w, h))
    resolutions[(w, h)].append(seq)

w_arr, h_arr = np.array(widths), np.array(heights)
print(f'Total: {len(widths)} sequences')
print(f'Width  — min: {w_arr.min()}, max: {w_arr.max()}, mean: {w_arr.mean():.0f}, median: {np.median(w_arr):.0f}')
print(f'Height — min: {h_arr.min()}, max: {h_arr.max()}, mean: {h_arr.mean():.0f}, median: {np.median(h_arr):.0f}')
print(f'Unique resolutions: {len(set(zip(widths, heights)))}')
print()
for cat in sorted(cat_res):
    ws = [r[0] for r in cat_res[cat]]
    hs = [r[1] for r in cat_res[cat]]
    print(f'{cat} ({len(ws)} seqs): W [{min(ws)}-{max(ws)}], H [{min(hs)}-{max(hs)}], mean {np.mean(ws):.0f}x{np.mean(hs):.0f}')
print()
print('=== Per-sequence resolutions ===')
for (w, h), seqs in sorted(resolutions.items(), key=lambda x: -len(x[1])):
    for s in seqs:
        print(f'{s}: {w}x{h}')
"
```

Adapt the script for the specific dataset:
- **Image subdirectory**: may be `img/`, `images/`, or frames directly in the sequence dir.
- **Category extraction**: may come from directory name prefix (`car_01` → `car`), parent directory, metadata file, or class annotation files.
- **Frame extension**: may be `.jpg`, `.png`, `.tif`, etc. Use the appropriate glob.

## Step 3: Write the markdown report

Create or update `docs/<dataset_name_lowercase>_resolution.md` (or append a "Resolution Statistics" section to an existing `docs/<dataset_name_lowercase>.md` if one already exists).

### Report structure

```markdown
# <DatasetName> — Resolution Statistics

Dataset path: `<root_path>`

## Summary

| | Width (px) | Height (px) |
|---|---|---|
| Min | ... | ... |
| Max | ... | ... |
| Mean | ... | ... |
| Median | ... | ... |

- **Total sequences:** N
- **Unique resolutions:** M / N

## Per-Category Breakdown

| Category | Sequences | Width Range | Height Range | Mean Resolution |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Per-Sequence Resolutions

### <category_1> (N sequences)

| Sequence | Width | Height |
|---|---|---|
| ... | ... | ... |

### <category_2> ...
```

### Additional sections (if applicable)

If the user provides GSD (ground sampling distance) info, satellite platform details, or other resolution-related context, add a dedicated section:

```markdown
## Ground Sampling Distance (GSD) Notes

<context provided by user>
```

## Step 4: Verify

After writing the report, do a quick sanity check:
- Total sequence count matches the dataset class or directory listing.
- No sequences were silently skipped (compare script output count vs `ls | wc -l`).
- Category names match what the dataset class uses.

Report the summary statistics to the user when done.

## Checklist

- [ ] Dataset layout understood (from dataset class or manual exploration)
- [ ] Resolution scan script adapted and run successfully
- [ ] Hidden/resource-fork files filtered out
- [ ] Markdown report written to `docs/`
- [ ] Sequence counts verified
