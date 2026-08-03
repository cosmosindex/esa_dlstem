"""Measure how fast a GOES fire mask actually changes, as a function of lead time.

A spread-forecasting task is only meaningful where the trivial persistence
baseline -- predict the mask at t+delta to be exactly the mask at t -- starts to
fail. This script sweeps the lead time and reports persistence IoU / F1 plus
raw pixel churn, so the forecast horizon can be chosen from evidence instead of
guessed.

Usage:
    python tools/analyze_goes_persistence.py <crop_dir> [-o out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Any FDC category that denotes fire, including the time-filtered (+20) tiers.
FIRE_VALUES = (10, 11, 13, 14, 15, 30, 31, 33, 34, 35)


def load_masks(crop_dir: Path):
    times, masks = [], []
    for f in sorted(crop_dir.glob("*.nc")):
        with xr.open_dataset(f) as ds:
            if "Mask" not in ds:
                continue
            m = ds["Mask"].isel(time=0).values
            masks.append(np.isin(m, FIRE_VALUES))
            times.append(ds["time"].values[0])
    return np.array(times), np.stack(masks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("crop_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    times, masks = load_masks(args.crop_dir)
    n = len(times)
    step = float(np.median(np.diff(times) / np.timedelta64(1, "m")))
    sizes = masks.sum((1, 2))
    print(f"{n} frames, cadence {step:.0f} min, "
          f"fire px min/med/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")

    rows = []
    for lag in range(1, n // 2):
        a, b = masks[:-lag], masks[lag:]
        inter = (a & b).sum((1, 2)).astype(float)
        union = (a | b).sum((1, 2)).astype(float)
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        rows.append(dict(
            lead=lag * step,
            iou=np.nanmean(iou),
            new=(b & ~a).sum((1, 2)).mean(),    # newly ignited
            gone=(a & ~b).sum((1, 2)).mean(),   # burnt out / no longer detected
            base=a.sum((1, 2)).mean(),
        ))

    print(f"\n{'lead(min)':>9} {'persistIoU':>11} {'new px':>7} {'lost px':>8} {'base px':>8}")
    for r in rows:
        if r["lead"] <= 15 or r["lead"] % 30 < step:
            print(f"{r['lead']:>9.0f} {r['iou']:>11.3f} {r['new']:>7.1f} "
                  f"{r['gone']:>8.1f} {r['base']:>8.1f}")

    leads = [r["lead"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax[0].plot(leads, [r["iou"] for r in rows], lw=2)
    ax[0].axhline(0.9, ls="--", c="gray", lw=1)
    ax[0].set(xlabel="lead time (min)", ylabel="persistence IoU",
              title="Copying the previous frame\n(a forecast must beat this)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(leads, [r["new"] for r in rows], lw=2, label="newly ignited")
    ax[1].plot(leads, [r["gone"] for r in rows], lw=2, label="no longer detected")
    ax[1].axhline(np.mean([r["base"] for r in rows]), ls="--", c="gray", lw=1,
                  label="mean fire size")
    ax[1].set(xlabel="lead time (min)", ylabel="pixels changed",
              title="How much actually moves")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    out = args.out or args.crop_dir / "persistence.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
