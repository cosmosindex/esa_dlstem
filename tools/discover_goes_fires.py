"""Discover wildfire events directly from the GOES L2 fire product.

Rather than hand-listing named fires and their coordinates, this sweeps the
ABI-L2-FDC fire mask over a date range and clusters the detections into events.
The fire product is tiny (~0.2 MB per scan) so a whole season can be scanned
cheaply, and what comes out is by construction the set of fires GOES can
actually see -- which is exactly the population a GOES-based model is trained
and evaluated on.

Detections are binned onto a coarse lat/lon grid, accumulated over the range,
and split into spatially connected components. Each component's daily time
series gives its start, end and peak.

Writes a CSV consumable by download_goes_fire.py.

Usage:
    python tools/discover_goes_fires.py --sat 18 \
        --start 2024-07-01 --end 2024-08-01 --every 2 -o events.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from pyproj import Proj
from scipy import ndimage

FIRE_VALUES = (10, 11, 13, 14, 15, 30, 31, 33, 34, 35)
# High-confidence tiers only; the probability tiers carry most of the false
# alarms and would seed spurious events.
STRONG_VALUES = (10, 11, 30, 31)

_SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def scan_time(key: str):
    m = _SCAN_RE.search(key)
    if not m:
        return None
    y, doy, hh, mm, ss = (int(g) for g in m.groups())
    return dt.datetime(y, 1, 1) + dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def fire_lonlat(ds: xr.Dataset, values=STRONG_VALUES):
    """Return lon/lat of every fire pixel in a FDC granule."""
    mask = np.isin(ds["Mask"].values, values)
    if not mask.any():
        return np.empty(0), np.empty(0)
    p = ds["goes_imager_projection"].attrs
    h = float(p["perspective_point_height"])
    proj = Proj(proj="geos", h=h,
                lon_0=float(p["longitude_of_projection_origin"]),
                a=float(p["semi_major_axis"]), b=float(p["semi_minor_axis"]),
                sweep=str(p["sweep_angle_axis"]))
    yi, xi = np.nonzero(mask)
    # x/y are scan angles in radians on the fixed grid; scale to metres.
    xm = ds["x"].values[xi] * h
    ym = ds["y"].values[yi] * h
    lon, lat = proj(xm, ym, inverse=True)
    ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180)
    return np.asarray(lon)[ok], np.asarray(lat)[ok]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sat", type=int, default=18, choices=[16, 17, 18, 19])
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, exclusive")
    ap.add_argument("--every", type=int, default=2, help="sample one scan every N hours")
    ap.add_argument("--sector", default="C", choices=["C", "F"])
    ap.add_argument("--cell", type=float, default=0.25, help="binning grid, degrees")
    ap.add_argument("--min-pixhours", type=float, default=200.0,
                    help="drop events below this accumulated fire-pixel count")
    ap.add_argument("--min-days", type=int, default=2,
                    help="drop events shorter than this many days")
    ap.add_argument("--max-span", type=float, default=3.0,
                    help="flag events wider than this many degrees as likely merges")
    ap.add_argument("--lat-range", type=float, nargs=2, default=(25.0, 52.0),
                    metavar=("MIN", "MAX"),
                    help="keep events inside this latitude band; the default "
                         "trims the CONUS sector edges where coverage is partial")
    ap.add_argument("-o", "--out", type=Path, default=Path("goes_fire_events.csv"))
    ap.add_argument("--gap-days", type=int, default=2,
                    help="tolerate this many days without detections before "
                         "splitting one event into two")
    ap.add_argument("--persist-frac", type=float, default=0.15,
                    help="mask cells detected on more than this fraction of "
                         "days as persistent industrial/geothermal sources")
    ap.add_argument("--counts", type=Path,
                    help="cache the scan here; reused instead of re-scanning "
                         "if the file already exists")
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel S3 fetches; the scan is I/O bound")
    ap.add_argument("--cache", type=Path,
                    default=Path("/tmp/goes_fdc_cache"))
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    product = f"ABI-L2-FDC{args.sector}"
    args.cache.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)

    # daily[(day, iy, ix)] -> fire pixel count
    daily: dict[tuple, int] = defaultdict(int)
    lat0, lon0 = 15.0, -170.0  # grid origin covering the CONUS/PACUS sectors

    hours = []
    cursor = start
    while cursor < end:
        hours.append(cursor)
        cursor += dt.timedelta(hours=args.every)

    def scan_hour(when: dt.datetime):
        """Fetch one representative scan for this hour and bin its detections."""
        prefix = (f"noaa-goes{args.sat}/{product}/{when.year}/"
                  f"{when.timetuple().tm_yday:03d}/{when.hour:02d}/")
        try:
            keys = fs.ls(prefix)
        except FileNotFoundError:
            return when, None, "no-listing"
        if not keys:
            return when, None, "empty"
        key = keys[0]
        local = args.cache / key.rsplit("/", 1)[-1]
        try:
            if not local.exists():
                fs.get(key, str(local))
            with xr.open_dataset(local, engine="h5netcdf") as ds:
                lon, lat = fire_lonlat(ds)
            return when, (lon, lat), None
        except Exception as e:  # a few granules are truncated on S3
            return when, None, type(e).__name__
        finally:
            local.unlink(missing_ok=True)

    if args.counts and args.counts.exists():
        # The S3 sweep dominates runtime, so keep it separate from clustering:
        # re-tuning the cluster parameters should not mean re-scanning.
        z = np.load(args.counts, allow_pickle=True)
        for (ds, a, b), c in zip(z["keys"], z["counts"]):
            daily[(dt.date.fromisoformat(str(ds)), int(a), int(b))] = int(c)
        print(f"loaded {len(daily)} cell-days from {args.counts}")
    else:
        scanned = skipped = 0
        print(f"scanning {len(hours)} hourly samples with {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for when, res, err in pool.map(scan_hour, hours):
                if err is not None:
                    skipped += 1
                    continue
                lon, lat = res
                if len(lon):
                    iy = ((lat - lat0) / args.cell).astype(int)
                    ix = ((lon - lon0) / args.cell).astype(int)
                    day = when.date()
                    for a, b in zip(iy, ix):
                        daily[(day, int(a), int(b))] += 1
                scanned += 1
                if scanned % 200 == 0:
                    print(f"  {scanned}/{len(hours)} scanned, "
                          f"{len(daily)} cell-days")
        print(f"scanned {scanned} granules ({skipped} skipped), "
              f"{len(daily)} cell-days")
        if args.counts:
            np.savez_compressed(
                args.counts,
                keys=np.array([(d.isoformat(), a, b) for (d, a, b) in daily],
                              dtype=object),
                counts=np.array(list(daily.values())),
            )
            print(f"cached scan -> {args.counts}")
    if not daily:
        raise SystemExit("no fire detections found in this range")

    # Clustering must be spatio-TEMPORAL. Accumulating the whole range into a
    # single map and labelling that merges every fire that ever burned in the
    # same place across different years, and once enough seasons pile up the
    # components bridge into one continent-sized blob.
    all_days = sorted({d for (d, _, _) in daily})
    day_index = {d: i for i, d in enumerate(all_days)}
    D = len(all_days)

    iy_lo = int((args.lat_range[0] - lat0) / args.cell)
    iy_hi = int((args.lat_range[1] - lat0) / args.cell)
    xs = [k[2] for k in daily]
    x_lo, x_hi = min(xs), max(xs)
    H, W = iy_hi - iy_lo + 1, x_hi - x_lo + 1

    cube = np.zeros((D, H, W), dtype=np.int16)
    for (d, a, b), c in daily.items():
        if iy_lo <= a <= iy_hi:
            cube[day_index[d], a - iy_lo, b - x_lo] += c
    occ = cube > 0
    print(f"cube {D}d x {H} x {W}, {int(occ.sum())} occupied cell-days")

    # Geothermal plants, gas flares and industrial heat sources trip the fire
    # detector at a fixed location all year round. They are not wildfires, and
    # left in they act as permanent bridges joining unrelated events.
    active_frac = occ.sum(0) / max(D, 1)
    persistent = active_frac > args.persist_frac
    print(f"masking {int(persistent.sum())} persistent hot-source cells "
          f"(active on >{args.persist_frac:.0%} of days)")
    occ &= ~persistent[None, :, :]
    cube = cube * occ

    # Allow a fire to go undetected for a few days (cloud, smoke, sampling)
    # without splitting into separate events: dilate along time only.
    grown = ndimage.binary_dilation(
        occ, structure=np.ones((2 * args.gap_days + 1, 1, 1), dtype=bool))
    lab, n = ndimage.label(grown, structure=np.ones((3, 3, 3)))
    lab = lab * occ
    print(f"{n} candidate events before filtering")

    events = []
    for i in range(1, n + 1):
        sel = lab == i
        if not sel.any():
            continue
        pixhours = float(cube[sel].sum())
        if pixhours < args.min_pixhours:
            continue
        di, yy, xx = np.nonzero(sel)
        days = sorted({all_days[k] for k in np.unique(di)})
        if len(days) < args.min_days:
            continue
        w = cube[sel].astype(float)
        clat = lat0 + (yy + iy_lo + 0.5).dot(w) / w.sum() * args.cell
        clon = lon0 + (xx + x_lo + 0.5).dot(w) / w.sum() * args.cell
        span = max(
            (yy.max() - yy.min() + 1) * args.cell,
            (xx.max() - xx.min() + 1) * args.cell,
        )
        per_day = np.bincount(di, weights=w)
        peak = all_days[int(np.argmax(per_day))]
        events.append(dict(
            lat=round(clat, 3), lon=round(clon, 3), sat=args.sat,
            start=days[0].isoformat(), end=days[-1].isoformat(),
            peak_day=peak.isoformat(), n_days=len(days),
            span_days=(days[-1] - days[0]).days + 1,
            pixhours=int(pixhours), span_deg=round(span, 2),
            half_deg=round(max(0.5, span * 0.75), 2),
            # A very wide component is usually several fires that happen to
            # touch on the coarse grid rather than one big event.
            likely_merge=int(span > args.max_span),
        ))

    events.sort(key=lambda e: -e["pixhours"])
    cols = ["lat", "lon", "sat", "start", "end", "peak_day", "n_days",
            "span_days", "pixhours", "span_deg", "half_deg", "likely_merge"]
    with args.out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for e in events:
            fh.write(",".join(str(e[c]) for c in cols) + "\n")

    print(f"\n{len(events)} events kept -> {args.out}\n")
    print(f"{'lat':>8} {'lon':>9} {'start':>11} {'end':>11} {'act':>4} "
          f"{'dur':>4} {'pixhrs':>7} {'span':>5}  flag")
    for e in events[:30]:
        print(f"{e['lat']:>8.2f} {e['lon']:>9.2f} {e['start']:>11} {e['end']:>11} "
              f"{e['n_days']:>4} {e['span_days']:>4} {e['pixhours']:>7} "
              f"{e['span_deg']:>5.2f}{'  MERGE?' if e['likely_merge'] else ''}")


if __name__ == "__main__":
    main()
