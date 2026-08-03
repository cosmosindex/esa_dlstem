"""Quick-look panels for a GOES fire crop produced by download_goes_fire.py.

Answers the question you have to settle before designing anything else: what
does a wildfire actually look like in each ABI channel, and how many pixels
does it occupy at 2 km?

Panels: C07 (3.9 um) brightness temperature, the C07-C14 split-window
difference the operational detector keys on, the C02 visible channel, and the
L2 fire mask. Prints per-frame pixel counts for each fire-mask category.

Usage:
    python tools/plot_goes_fire_quicklook.py <crop_dir> [-o out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# From the ABI L2 FDC product definition. 10/11 are the high-confidence
# categories; 13/14/15 are probability tiers with progressively more false
# alarms. Values +20 are the "filtered" counterparts of 10-15.
MASK_LABELS = {
    10: "processed fire",
    11: "saturated fire",
    12: "cloud contaminated",
    13: "high probability",
    14: "medium probability",
    15: "low probability",
    30: "filtered processed",
    31: "filtered saturated",
    32: "filtered cloud contam.",
    33: "filtered high prob.",
    34: "filtered medium prob.",
    35: "filtered low prob.",
}
FIRE_VALUES = (10, 11, 13, 14, 15, 30, 31, 33, 34, 35)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("crop_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--frame", type=int, default=0, help="which timestep to plot")
    args = ap.parse_args()

    files = sorted(args.crop_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"no .nc files in {args.crop_dir}")
    print(f"{len(files)} timesteps in {args.crop_dir}")

    # Fire-pixel census across every frame, so the plotted frame can be judged
    # against the whole window rather than in isolation.
    print(f"\n{'timestep':>18}  {'fire px':>7}  breakdown")
    for f in files:
        with xr.open_dataset(f) as ds:
            if "Mask" not in ds:
                print(f"{f.stem:>18}  {'-':>7}  (no fire product)")
                continue
            m = ds["Mask"].values
            vals, counts = np.unique(m[np.isfinite(m)].astype(int), return_counts=True)
            hits = {int(v): int(c) for v, c in zip(vals, counts) if int(v) in FIRE_VALUES}
            total = sum(hits.values())
            desc = ", ".join(
                f"{MASK_LABELS.get(v, v)}={c}" for v, c in sorted(hits.items())
            )
            print(f"{f.stem:>18}  {total:>7}  {desc or '(none)'}")

    ds = xr.open_dataset(files[args.frame]).isel(time=0)
    c07 = ds["CMI_C07"]
    c14 = ds["CMI_C14"]
    btd = c07 - c14

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)

    im = axes[0].imshow(c07, cmap="inferno")
    axes[0].set_title("C07  3.9 um BT (K)\nemissive: works day AND night")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(btd, cmap="magma")
    axes[1].set_title("C07 - C14  split-window BTD (K)\nwhat the operational detector keys on")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    if "CMI_C02" in ds:
        im = axes[2].imshow(ds["CMI_C02"], cmap="gray")
        axes[2].set_title("C02  0.64 um reflectance\nsmoke + cloud; dark at night")
        fig.colorbar(im, ax=axes[2], fraction=0.046)

    if "Mask" in ds:
        mask = ds["Mask"].values
        fire = np.isin(mask, FIRE_VALUES)
        axes[3].imshow(c07, cmap="gray")
        axes[3].imshow(np.where(fire, 1.0, np.nan), cmap="autumn", vmin=0, vmax=1)
        axes[3].set_title(f"L2 fire mask over C07\n{int(fire.sum())} fire pixels @ 2 km")

    stamp = str(ds["time"].values)[:19]
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"{args.crop_dir.name}   {stamp} UTC   "
        f"{ds.sizes['y']}x{ds.sizes['x']} px @ 2 km",
        fontsize=13,
    )

    out = args.out or args.crop_dir / "quicklook.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
