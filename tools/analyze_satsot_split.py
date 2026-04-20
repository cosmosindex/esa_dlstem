"""
Analyze SatSOT split balance across sequence attributes.

SatSOT sequence attributes (from the paper, stored in ``SatSOT.json`` as a
per-sequence ``attr`` list of strings) — canonical 11-attribute set, in order:

    ARC, BC, BJT, DEF, FOC, IV, LQ, POC, ROT, SOB, TO

We compare the current class-stratified 80/10/10 split (seed=42) against a
multi-label iterative-stratification split that also targets balance across
the 11 binary attributes. Categories are very imbalanced (car 65 / train 26 /
plane 9 / ship 5), so the recommended variant is the hybrid one that pre-
assigns tiny classes round-robin so every split still gets ≥1 plane and ship.

Usage:
    python tools/analyze_satsot_split.py [--root /data/ESA_DLSTEM_2025/data/trafic/SatSOT]
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ATTR_NAMES = ["ARC", "BC", "BJT", "DEF", "FOC", "IV", "LQ", "POC", "ROT", "SOB", "TO"]
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42
_META_FILENAME = "SatSOT.json"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_satsot(root: Path) -> list[dict]:
    """Return a list of per-sequence dicts: {vid, category, attrs (11,)}."""
    meta_path = root / _META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    attr_to_idx = {a: i for i, a in enumerate(ATTR_NAMES)}

    out = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        gt_path = seq_dir / "groundtruth.txt"
        img_dir = seq_dir / "img"
        if not gt_path.exists() or not img_dir.exists():
            continue

        vid = seq_dir.name
        attrs_raw = meta.get(vid, {}).get("attr", [])
        attrs = np.zeros(len(ATTR_NAMES), dtype=np.int32)
        for a in attrs_raw:
            if a in attr_to_idx:
                attrs[attr_to_idx[a]] = 1

        category = re.sub(r"_\d+$", "", vid)  # "car_01" → "car"
        out.append({"vid": vid, "category": category, "attrs": attrs})
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def class_stratified_split(seqs: list[dict], seed: int = SEED) -> dict[str, str]:
    """Legacy production split: shuffle by class, take 80/10/10."""
    rng = np.random.RandomState(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for s in seqs:
        by_cat[s["category"]].append(s["vid"])

    split_map: dict[str, str] = {}
    tr_r, va_r, _ = RATIOS
    for cat in sorted(by_cat):
        ids = by_cat[cat][:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, round(n * tr_r))
        n_val = max(1, round(n * va_r))
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        for v in ids[:n_train]:
            split_map[v] = "train"
        for v in ids[n_train:n_train + n_val]:
            split_map[v] = "val"
        for v in ids[n_train + n_val:]:
            split_map[v] = "test"
    return split_map


def iterative_stratified_split(
    seqs: list[dict],
    label_matrix: np.ndarray,           # (N, L) binary labels
    seed: int = SEED,
    preassigned: dict[int, int] | None = None,  # sample_idx -> split_idx (0/1/2)
) -> dict[str, str]:
    """Iterative stratification (Sechidis et al. 2011) for multi-label data."""
    rng = np.random.RandomState(seed)

    N, _ = label_matrix.shape
    targets = np.array(RATIOS) * N
    per_label_targets = label_matrix.sum(axis=0)[:, None] * np.array(RATIOS)

    remaining = np.ones(N, dtype=bool)
    split_idx = np.full(N, -1, dtype=np.int32)
    split_counts = np.zeros(3, dtype=np.float64)
    label_counts = np.zeros_like(per_label_targets)

    if preassigned:
        for i, s in preassigned.items():
            split_idx[i] = s
            split_counts[s] += 1
            for l in np.where(label_matrix[i] > 0)[0]:
                label_counts[l, s] += 1
            remaining[i] = False

    remaining_label_counts = label_matrix[remaining].sum(axis=0).astype(np.float64)

    while remaining.any():
        active = remaining_label_counts > 0
        if not active.any():
            deficit = targets - split_counts
            for idx in np.where(remaining)[0]:
                s = int(np.argmax(deficit))
                split_idx[idx] = s
                split_counts[s] += 1
                deficit = targets - split_counts
                remaining[idx] = False
            break

        rare_counts = np.where(active, remaining_label_counts, np.inf)
        label = int(np.argmin(rare_counts))
        candidates = np.where(remaining & (label_matrix[:, label] > 0))[0]
        if candidates.size == 0:
            remaining_label_counts[label] = 0
            continue

        for idx in candidates:
            label_deficit = per_label_targets[label] - label_counts[label]
            best = np.where(label_deficit == label_deficit.max())[0]
            if best.size > 1:
                total_deficit = targets - split_counts
                td = total_deficit[best]
                best = best[np.where(td == td.max())[0]]
            s = int(best[rng.randint(best.size)])

            split_idx[idx] = s
            split_counts[s] += 1
            for l in np.where(label_matrix[idx] > 0)[0]:
                label_counts[l, s] += 1
                remaining_label_counts[l] -= 1
            remaining[idx] = False

    name_map = {0: "train", 1: "val", 2: "test"}
    return {seqs[i]["vid"]: name_map[int(split_idx[i])] for i in range(N)}


def class_plus_attr_stratified_split(seqs: list[dict], seed: int = SEED) -> dict[str, str]:
    """Iterative stratification using (one-hot category) ⊕ (11 binary attrs) as labels."""
    cats = sorted({s["category"] for s in seqs})
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    N = len(seqs)
    L = len(cats) + len(ATTR_NAMES)
    labels = np.zeros((N, L), dtype=np.int32)
    for i, s in enumerate(seqs):
        labels[i, cat_to_idx[s["category"]]] = 1
        labels[i, len(cats):] = s["attrs"]
    return iterative_stratified_split(seqs, labels, seed=seed)


def hybrid_split(
    seqs: list[dict],
    seed: int = SEED,
    small_cat_thresh: int = 10,
    rare_attr_thresh: int = 9,
) -> dict[str, str]:
    """Hybrid split that guarantees each split gets every category and every attribute.

    Steps (each later step only fills in samples not yet preassigned):

      1. **Tiny classes** (n ≤ ``small_cat_thresh``): round-robin their samples
         across splits so every split gets ≥1 sequence of the class.
      2. **Rare attributes** (positive count ≤ ``rare_attr_thresh``): for each
         such attribute, pick up to 3 positive-carrying samples that are not yet
         assigned, and force one into each split. Prevents the iterative-strat
         greedy from dumping all rare positives into train.
      3. Run iterative stratification on all remaining samples with
         (one-hot category) ⊕ (binary attrs) as the label matrix, honouring the
         pre-assignments from steps 1–2.
    """
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        by_cat[s["category"]].append(i)

    preassigned: dict[int, int] = {}
    test_first_cycle = [2, 1, 0]

    # Step 1 — tiny classes
    for cat, members in sorted(by_cat.items()):
        if len(members) > small_cat_thresh:
            continue
        idx_list = members[:]
        rng2 = np.random.RandomState(seed + (zlib.crc32(cat.encode()) % 10_000))
        rng2.shuffle(idx_list)
        for k, idx in enumerate(idx_list):
            preassigned[idx] = test_first_cycle[k % 3]

    # Step 2 — rare attributes: ensure every split has ≥1 positive for each
    attr_matrix = np.stack([s["attrs"] for s in seqs])
    for a_idx, a_name in enumerate(ATTR_NAMES):
        positives = np.where(attr_matrix[:, a_idx] > 0)[0].tolist()
        if not positives or len(positives) > rare_attr_thresh:
            continue
        # Which splits still need coverage for this attribute?
        covered = {preassigned[i] for i in positives if i in preassigned}
        missing = [s for s in (2, 1, 0) if s not in covered]
        if not missing:
            continue
        # Candidates: positives not yet preassigned
        unassigned = [i for i in positives if i not in preassigned]
        if not unassigned:
            continue
        rng3 = np.random.RandomState(seed + (zlib.crc32(a_name.encode()) % 10_000))
        rng3.shuffle(unassigned)
        for s in missing:
            if not unassigned:
                break
            preassigned[unassigned.pop()] = s

    # Step 3 — iterative stratification on class ⊕ attrs labels
    cats = sorted({s["category"] for s in seqs})
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    N = len(seqs)
    labels = np.zeros((N, len(cats) + len(ATTR_NAMES)), dtype=np.int32)
    for i, s in enumerate(seqs):
        labels[i, cat_to_idx[s["category"]]] = 1
        labels[i, len(cats):] = s["attrs"]

    return iterative_stratified_split(seqs, labels, seed=seed, preassigned=preassigned)


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

def summarise(seqs: list[dict], split_map: dict[str, str], title: str) -> None:
    print(f"\n=== {title} ===")
    by_split = defaultdict(list)
    for s in seqs:
        by_split[split_map[s["vid"]]].append(s)

    cats = sorted({s["category"] for s in seqs})
    print("\nCategory counts:")
    print(f"{'category':<10} " + " ".join(f"{k:>6}" for k in SPLITS) + "  total")
    for cat in cats:
        row = {k: sum(1 for s in by_split[k] if s["category"] == cat) for k in SPLITS}
        total = sum(row.values())
        print(f"{cat:<10} " + " ".join(f"{row[k]:>6}" for k in SPLITS) + f"  {total:>5}")
    totals = {k: len(by_split[k]) for k in SPLITS}
    print(f"{'TOTAL':<10} " + " ".join(f"{totals[k]:>6}" for k in SPLITS) + f"  {sum(totals.values()):>5}")

    print("\nAttribute positive counts (sequences with flag=1):")
    total_pos = {a: sum(int(s["attrs"][i] > 0) for s in seqs) for i, a in enumerate(ATTR_NAMES)}
    print(f"{'attr':<6} {'total':>6} " + " ".join(f"{k:>6}" for k in SPLITS) + "   "
          + " ".join(f"{k+'%':>7}" for k in SPLITS) + "   dev")
    for i, a in enumerate(ATTR_NAMES):
        counts = {k: sum(int(s["attrs"][i] > 0) for s in by_split[k]) for k in SPLITS}
        tot = total_pos[a]
        frac = {k: counts[k] / tot if tot else 0.0 for k in SPLITS}
        dev = np.sqrt(np.mean([(frac[k] - r) ** 2 for k, r in zip(SPLITS, RATIOS)]))
        print(
            f"{a:<6} {tot:>6} "
            + " ".join(f"{counts[k]:>6}" for k in SPLITS)
            + "   "
            + " ".join(f"{frac[k]*100:>6.1f}%" for k in SPLITS)
            + f"   {dev:.03f}"
        )

    devs = []
    for i in range(len(ATTR_NAMES)):
        counts = {k: sum(int(s["attrs"][i] > 0) for s in by_split[k]) for k in SPLITS}
        if sum(counts.values()) == 0:
            continue
        frac = {k: counts[k] / sum(counts.values()) for k in SPLITS}
        devs.append(np.sqrt(np.mean([(frac[k] - r) ** 2 for k, r in zip(SPLITS, RATIOS)])))
    print(f"\nMean RMS deviation across {len(devs)} attrs: {np.mean(devs):.04f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ESA_DLSTEM_2025/data/trafic/SatSOT", type=Path)
    args = ap.parse_args()

    seqs = load_satsot(args.root)
    print(f"Loaded {len(seqs)} sequences with attribute metadata.")

    attr_matrix = np.stack([s["attrs"] for s in seqs])
    print("Attribute positive rate in full dataset:")
    for i, a in enumerate(ATTR_NAMES):
        n = int((attr_matrix[:, i] > 0).sum())
        print(f"  {a:<4}  {n:>3} / {len(seqs)}  ({n/len(seqs)*100:.1f}%)")

    current = class_stratified_split(seqs, seed=SEED)
    summarise(seqs, current, "Current: class-stratified 80/10/10 (seed=42)")

    multi = class_plus_attr_stratified_split(seqs, seed=SEED)
    summarise(seqs, multi, "Option A: iterative stratification on class + 11 attrs (seed=42)")

    hyb = hybrid_split(seqs, seed=SEED, small_cat_thresh=10)
    summarise(seqs, hyb, "Option B (recommended): hybrid — round-robin tiny classes, iterative strat on rest")


if __name__ == "__main__":
    main()
