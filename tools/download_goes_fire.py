"""Download GOES-R ABI imagery + fire product for a wildfire event window.

Pulls two co-registered CONUS products straight from the NOAA open-data S3
buckets (anonymous, no quota, full radiometric fidelity):

  ABI-L2-MCMIPC  all 16 ABI channels as Cloud & Moisture Imagery, 2 km, 5 min
  ABI-L2-FDCC    Fire Detection & Characterization: Mask/Area/Temp/Power/DQF

Both live on the same fixed-grid geostationary projection, so one lat/lon bbox
crops to identical pixel windows and FDC can serve directly as a per-pixel
label raster for the imagery.

Each timestep is written as a small cropped NetCDF holding the requested
channels plus the fire product. Nothing is rescaled or 8-bit quantised.

Usage:
    python tools/download_goes_fire.py --preset park_fire_2024 --hours 6
    python tools/download_goes_fire.py --lat 39.9 --lon -121.6 \
        --start 2024-07-25T18:00 --end 2024-07-26T06:00 --sat 18
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from pyproj import Proj

# Channels worth keeping for fire work. C07/C14 are the operational detection
# pair and are emissive (usable day AND night); C05/C06 are reflective and only
# respond to intense fires in daylight; C02 carries smoke; C13/C15 support the
# cloud mask. See ABI fire ATBD.
DEFAULT_BANDS = ["C02", "C05", "C06", "C07", "C13", "C14", "C15"]

FIRE_VARS = ["Mask", "Area", "Temp", "Power", "DQF"]

# lat, lon, default satellite, default window.
# Satellite choice matters: view zenith angle grows fast away from the sub-point,
# stretching the nominal 2 km pixel. Prefer the satellite whose longitude is
# closest to the target.
PRESETS = {
    # Large, fast-spreading, well inside the CONUS sector -> 5 min cadence and
    # good geometry from GOES-West.
    "park_fire_2024": dict(
        lat=39.9, lon=-121.6, sat=18, start="2024-07-25T18:00"
    ),
    # Original area of interest. NOT in the CONUS sector, so this preset only
    # works with the full-disk products (10 min) -- see --sector.
    "donnie_creek_2023": dict(
        lat=58.5, lon=-122.0, sat=18, start="2023-06-10T18:00"
    ),
}

_SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def _scan_time(key: str) -> dt.datetime | None:
    """Parse the scan-start stamp sYYYYDDDHHMMSSS out of an ABI filename."""
    m = _SCAN_RE.search(key)
    if not m:
        return None
    year, doy, hh, mm, ss = (int(g) for g in m.groups())
    return dt.datetime(year, 1, 1) + dt.timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss
    )


def list_product(fs, sat: int, product: str, start: dt.datetime, end: dt.datetime):
    """List S3 keys for `product` whose scan start falls in [start, end)."""
    out = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        prefix = (
            f"noaa-goes{sat}/{product}/{cursor.year}/"
            f"{cursor.timetuple().tm_yday:03d}/{cursor.hour:02d}/"
        )
        try:
            keys = fs.ls(prefix)
        except FileNotFoundError:
            keys = []
        for k in keys:
            t = _scan_time(k)
            if t is not None and start <= t < end:
                out.append((t, k))
        cursor += dt.timedelta(hours=1)
    return sorted(out)


def geos_bbox(ds: xr.Dataset, lat: float, lon: float, half_deg: float):
    """Convert a lat/lon box to fixed-grid scan-angle slices for this dataset."""
    p = ds["goes_imager_projection"].attrs
    h = float(p["perspective_point_height"])
    proj = Proj(
        proj="geos",
        h=h,
        lon_0=float(p["longitude_of_projection_origin"]),
        a=float(p["semi_major_axis"]),
        b=float(p["semi_minor_axis"]),
        sweep=str(p["sweep_angle_axis"]),
    )
    lons = [lon - half_deg, lon + half_deg, lon - half_deg, lon + half_deg]
    lats = [lat - half_deg, lat - half_deg, lat + half_deg, lat + half_deg]
    xs, ys = proj(lons, lats)
    xs, ys = np.asarray(xs) / h, np.asarray(ys) / h  # metres -> scan radians
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError(
            f"lat/lon ({lat}, {lon}) does not project onto GOES-{p['longitude_of_projection_origin']}"
            " -- the point is off the visible disk. Pick the other satellite."
        )
    # y descends in the file, so the slice must be reversed.
    return slice(xs.min(), xs.max()), slice(ys.max(), ys.min())


def crop(ds: xr.Dataset, xsl, ysl, variables):
    sub = ds[variables].sel(x=xsl, y=ysl)
    sub.attrs = dict(ds.attrs)
    return sub


def fetch(fs, key: str, cache: Path) -> Path:
    """Copy an S3 object to local disk before opening it.

    HDF5 random access straight off S3 issues thousands of small range requests
    and is ~100x slower than pulling the whole object once, even though we only
    keep a small crop of it.
    """
    local = cache / key.rsplit("/", 1)[-1]
    if not local.exists():
        fs.get(key, str(local))
    return local


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--sat", type=int, choices=[16, 17, 18, 19])
    ap.add_argument("--start", help="UTC, e.g. 2024-07-25T18:00")
    ap.add_argument("--end", help="UTC; defaults to start + --hours")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument(
        "--half-deg", type=float, default=1.0, help="half-width of the crop box"
    )
    ap.add_argument(
        "--sector",
        default="C",
        choices=["C", "F"],
        help="C = CONUS (5 min), F = full disk (10 min, needed outside CONUS)",
    )
    ap.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    ap.add_argument("--out", default="data/goes")
    ap.add_argument("--limit", type=int, help="stop after N timesteps (smoke test)")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset]) if args.preset else {}
    lat = args.lat if args.lat is not None else cfg.get("lat")
    lon = args.lon if args.lon is not None else cfg.get("lon")
    sat = args.sat if args.sat is not None else cfg.get("sat")
    start_s = args.start or cfg.get("start")
    if lat is None or lon is None or sat is None or start_s is None:
        ap.error("need --preset, or all of --lat --lon --sat --start")

    start = dt.datetime.fromisoformat(start_s)
    end = (
        dt.datetime.fromisoformat(args.end)
        if args.end
        else start + dt.timedelta(hours=args.hours)
    )

    img_product = f"ABI-L2-MCMIP{args.sector}"
    fire_product = f"ABI-L2-FDC{args.sector}"

    name = args.preset or f"g{sat}_{lat:.2f}_{lon:.2f}"
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / ".cache"
    cache.mkdir(exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)
    print(f"listing {img_product} / {fire_product} on GOES-{sat} "
          f"{start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC")
    imgs = list_product(fs, sat, img_product, start, end)
    fires = dict(list_product(fs, sat, fire_product, start, end))
    print(f"  {len(imgs)} imagery scans, {len(fires)} fire scans")
    if not imgs:
        raise SystemExit("no scans found -- check the date and satellite")

    fire_keys = sorted(fires)
    fire_times = np.array(fire_keys, dtype="datetime64[s]")
    img_vars = [f"CMI_{b}" for b in args.bands]

    written = 0
    for t, key in imgs:
        if args.limit and written >= args.limit:
            break
        stamp = t.strftime("%Y%m%dT%H%M%S")
        dest = outdir / f"{stamp}.nc"
        if dest.exists():
            written += 1
            continue

        local = fetch(fs, key, cache)
        with xr.open_dataset(local, engine="h5netcdf") as ds:
            xsl, ysl = geos_bbox(ds, lat, lon, args.half_deg)
            keep = [v for v in img_vars if v in ds]
            sub = crop(ds, xsl, ysl, keep + ["goes_imager_projection"]).load()
        local.unlink()

        # Pair with the nearest fire scan (same cadence, stamps can differ by
        # a few seconds between products).
        j = int(np.argmin(np.abs(fire_times - np.datetime64(t, "s"))))
        if abs((fire_times[j] - np.datetime64(t, "s")) / np.timedelta64(1, "s")) < 300:
            flocal = fetch(fs, fires[fire_keys[j]], cache)
            with xr.open_dataset(flocal, engine="h5netcdf") as fds:
                fkeep = [v for v in FIRE_VARS if v in fds]
                fsub = crop(fds, xsl, ysl, fkeep).load()
            flocal.unlink()
            # Reindex onto the imagery grid; the two products share the fixed
            # grid but float rounding can shift the coordinate values.
            fsub = fsub.assign_coords(x=sub.x[: fsub.sizes["x"]],
                                      y=sub.y[: fsub.sizes["y"]])
            sub = sub.merge(fsub, compat="override", join="left")
        else:
            print(f"  {stamp}: no fire scan within 5 min")

        sub = sub.expand_dims(time=[np.datetime64(t, "s")])
        # The source files store x/y as scaled int16; carrying that encoding
        # over would quantise the cropped coordinates.
        for c in ("x", "y"):
            sub[c].encoding = {}
        sub.to_netcdf(dest)
        written += 1
        mb = dest.stat().st_size / 1e6
        ny, nx = sub.sizes["y"], sub.sizes["x"]
        print(f"  {stamp}  {ny}x{nx}px  {mb:.1f} MB  -> {dest}")

    print(f"done: {written} timesteps in {outdir}")


if __name__ == "__main__":
    main()
