"""
Report the actual SV248S split produced by `SV248SDataset` for docs.

Runs the dataset loader for each of the three splits, then prints:
  1. Per-category counts + frames per split (for `docs/split_statistics.md`).
  2. Attribute positive counts per split (for `docs/sv248s_split_attributes.md`).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.sv248s import ATTR_NAMES, SV248SDataset  # noqa: E402

ROOT = Path("/data/ESA_DLSTEM_2025/data/trafic/SV248S")
RATIOS = (0.8, 0.1, 0.1)
SPLITS = ("train", "val", "test")


def main() -> None:
    per_split: dict[str, list] = {}
    for split in SPLITS:
        ds = SV248SDataset(root=ROOT, split=split, mode="video", clip_len=1)
        per_split[split] = ds.videos
        # keep a reference so attr cache is alive
        per_split[f"{split}_attr"] = ds._attr_cache

    # Category & frame counts
    print("Category counts:")
    cats = sorted({v.category for s in SPLITS for v in per_split[s]})
    print(f"{'Category':<10} " + " ".join(f"{k:>6}" for k in SPLITS) + "  total")
    totals = {s: 0 for s in SPLITS}
    for cat in cats:
        row = {s: sum(1 for v in per_split[s] if v.category == cat) for s in SPLITS}
        total = sum(row.values())
        for s in SPLITS:
            totals[s] += row[s]
        print(f"{cat:<10} " + " ".join(f"{row[s]:>6}" for s in SPLITS) + f"  {total:>5}")
    print(f"{'TOTAL':<10} " + " ".join(f"{totals[s]:>6}" for s in SPLITS) + f"  {sum(totals.values()):>5}")

    print("\nFrame counts:")
    print(f"{'Split':<8} {'Videos':>7} {'Frames':>8}")
    for s in SPLITS:
        n_vids = len(per_split[s])
        n_frames = sum(v.num_frames for v in per_split[s])
        print(f"{s:<8} {n_vids:>7} {n_frames:>8}")

    # Attribute positive counts
    print("\nAttribute positive counts:")
    print(f"{'Attr':<6} {'total':>6} " + " ".join(f"{k:>6}" for k in SPLITS)
          + "  " + " ".join(f"{k+'%':>7}" for k in SPLITS) + "   RMS")
    devs = []
    for i, a in enumerate(ATTR_NAMES):
        counts = {
            s: sum(
                1 for v in per_split[s]
                if (per_split[f"{s}_attr"].get(v.video_id) is not None
                    and per_split[f"{s}_attr"][v.video_id][i] > 0)
            )
            for s in SPLITS
        }
        total = sum(counts.values())
        frac = {s: counts[s] / total if total else 0.0 for s in SPLITS}
        dev = np.sqrt(np.mean([(frac[s] - r) ** 2 for s, r in zip(SPLITS, RATIOS)]))
        devs.append(dev)
        print(
            f"{a:<6} {total:>6} "
            + " ".join(f"{counts[s]:>6}" for s in SPLITS) + "  "
            + " ".join(f"{frac[s]*100:>6.1f}%" for s in SPLITS)
            + f"   {dev:.3f}"
        )
    print(f"\nMean RMS deviation: {np.mean(devs):.4f}")


if __name__ == "__main__":
    main()
