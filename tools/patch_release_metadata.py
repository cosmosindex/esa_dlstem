#!/usr/bin/env python
"""Bring an already-built Space-Tracker release up to the current metadata spec.

Two changes, both of which ``build_space_tracker_release.py`` now makes on its
own. This script exists so they can be applied to a package that is already on
disk, without re-copying ninety gigabytes of frames:

1. **The attribute taxonomy is three groups, not four.** ``DEF`` and ``ARC``
   are each defined by two source datasets, but the 32 px size filter leaves
   DEF in OOTB only and ARC in SatSOT only, so neither can be pooled across
   sources. Pooled is five rows (``SOB``, ``ROT``, ``OCC``, ``IV``, ``BC``);
   the other thirteen are reported on their single annotating source. The
   18-attribute taxonomy is unchanged -- only which of its rows may be scored
   across datasets. Nothing detaches from a sequence: ``native_attrs`` and
   ``taxonomy_attrs`` keep every label the sources gave it, so all 18
   attributes (and the five occlusion sub-types under OCC) stay filterable and
   evaluable. Only ``unified_attrs``, which lists the pooled rows alone,
   stops naming DEF. The script refuses to write if any sequence would lose an
   attribute from its full list.

2. **Acquisition constants are published per sequence.** ``platform``,
   ``gsd_m`` and ``fps``, each with a provenance tag, in the manifests, in the
   COCO-VID ``videos`` table and in ``seqinfo.ini``. Ground sample distance and
   frame rate are what turn a displacement in pixels into one in metres per
   second, which is how the released ``motion_state`` was computed; without
   them that computation cannot be reproduced from the package.

Idempotent: running it twice changes nothing the second time.

Usage::

    python tools/patch_release_metadata.py --release $DATA_ROOT/release/space_tracker
    python tools/patch_release_metadata.py --release ... --dry-run
"""

from __future__ import annotations

import argparse
import collections
import configparser
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import DATA_ROOT  # noqa: E402
from tools.build_space_tracker_manifest import (  # noqa: E402
    TAXONOMY_GROUPS, TAXONOMY_SPEC, UNIFIED_ATTR_SPEC)
from tools.build_space_tracker_release import acquisition, _sot_taxonomy  # noqa: E402

#: Attributes a sequence may no longer claim as unified: they are defined by a
#: second source that the size filter removed from the release.
DEMOTED = {a for a in ("DEF", "ARC") if a not in UNIFIED_ATTR_SPEC}

ACQ_FIELDS = ("platform", "gsd_m", "gsd_source", "fps", "fps_source")


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _save(p: Path, doc: dict, dry: bool) -> None:
    if dry:
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    tmp.replace(p)


def _full_attrs(record: dict) -> tuple[list, list]:
    """The two fields that must survive untouched: every label, every taxonomy row."""
    return (record.get("native_attrs", record.get("native_attributes", [])),
            record.get("taxonomy_attrs", record.get("taxonomy_attributes", [])))


def patch_manifest(path: Path, half: str, dry: bool) -> dict[str, int]:
    doc = _load(path)
    n = {"unified_attrs": 0, "acquisition": 0}
    if half == "sot":
        doc["unified_attributes"] = UNIFIED_ATTR_SPEC
        doc["attribute_taxonomy"] = _sot_taxonomy()["attribute_taxonomy"]
    for s in doc["sequences"]:
        if half == "sot":
            before = _full_attrs(s)
            keep = [a for a in s.get("unified_attrs", []) if a not in DEMOTED]
            if keep != s.get("unified_attrs"):
                s["unified_attrs"] = keep
                n["unified_attrs"] += 1
            if _full_attrs(s) != before:
                raise AssertionError(f"{s['name']} lost an attribute: {before}")
        acq = acquisition(half, s["name"])
        if {k: s.get(k) for k in ACQ_FIELDS} != acq:
            s.update(acq)
            n["acquisition"] += 1
    _save(path, doc, dry)
    return n


def patch_coco(path: Path, half: str, dry: bool) -> dict[str, int]:
    doc = _load(path)
    n = {"unified_attrs": 0, "acquisition": 0}
    for v in doc["videos"]:
        if half == "sot":
            before = _full_attrs(v)
            keep = [a for a in v.get("unified_attributes", []) if a not in DEMOTED]
            if keep != v.get("unified_attributes"):
                v["unified_attributes"] = keep
                n["unified_attrs"] += 1
            if _full_attrs(v) != before:
                raise AssertionError(f"{v['name']} lost an attribute: {before}")
        acq = acquisition(half, v["name"])
        if {k: v.get(k) for k in ACQ_FIELDS} != acq:
            v.update(acq)
            n["acquisition"] += 1
    _save(path, doc, dry)
    return n


def patch_seqinfo(path: Path, half: str, name: str, dry: bool) -> bool:
    """Rewrite one seqinfo.ini, preserving key order and adding the constants."""
    acq = acquisition(half, name)
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    sec = cfg["Sequence"]
    want = {
        "frameRate": f"{acq['fps']:g}" if acq["fps"] else "-1",
        **({"gsd": f"{acq['gsd_m']:g}"} if acq["gsd_m"] else {}),
        **({"platform": acq["platform"]} if acq["platform"] else {}),
    }
    if all(sec.get(k) == v for k, v in want.items()):
        return False
    # imExt is the last key the builder writes before its extras, so the two
    # new constants go in after it -- rebuild the section to keep that order.
    ordered = {}
    for k, v in sec.items():
        ordered[k] = want.get(k, v)
        if k == "imExt":
            ordered.update({k2: v2 for k2, v2 in want.items() if k2 != "frameRate"})
    cfg["Sequence"] = ordered
    if not dry:
        with open(path, "w") as f:
            cfg.write(f)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path,
                    default=Path(DATA_ROOT) / "release" / "space_tracker")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root, dry = args.release, args.dry_run

    if not (root / "sot").is_dir() or not (root / "mot").is_dir():
        sys.exit(f"not a Space-Tracker release: {root}")

    print(f"{'would patch' if dry else 'patching'} {root}")
    print(f"  taxonomy: {len(UNIFIED_ATTR_SPEC)} pooled, "
          f"{len(TAXONOMY_GROUPS['single_source']['members'])} single-source, "
          f"{sum(1 for v in TAXONOMY_SPEC.values() if v['group'] != 'occlusion_subtypes')} "
          f"attributes in total"
          + (f"; demoting {', '.join(sorted(DEMOTED))}" if DEMOTED else ""))

    for half in ("sot", "mot"):
        m = patch_manifest(root / half / f"space_tracker_{half}.json", half, dry)
        print(f"  {half}/space_tracker_{half}.json: "
              f"{m['acquisition']} sequences gained constants, "
              f"{m['unified_attrs']} lost a demoted attribute")

        ann = root / half / "annotations"
        for p in [ann / f"space_tracker_{half}.json", *sorted((ann / "per_class").glob("*.json"))]:
            c = patch_coco(p, half, dry)
            print(f"  {p.relative_to(root)}: {c['acquisition']} videos updated, "
                  f"{c['unified_attrs']} attribute lists trimmed")

        seqinfos = sorted((root / half).glob("*/*/seqinfo.ini"))
        changed = sum(patch_seqinfo(p, half, p.parent.name, dry) for p in seqinfos)
        print(f"  {half}: {changed}/{len(seqinfos)} seqinfo.ini rewritten")

    # Every taxonomy row must still be carried by the sequences that carry it:
    # demoting DEF and ARC changes how they are scored, never whether they are
    # attached. Counted after the rewrite, against the manifest as it now is.
    m = _load(root / "sot" / "space_tracker_sot.json")
    carried = collections.Counter(
        a for s in m["sequences"] for a in s["taxonomy_attrs"])
    missing = [a for a in TAXONOMY_SPEC if a not in carried]
    print(f"  attributes attached to at least one released sequence: "
          f"{len(carried)}/{len(TAXONOMY_SPEC)}"
          + (f" -- MISSING {missing}" if missing else ""))
    print("     " + "  ".join(f"{a}={carried[a]}" for a in TAXONOMY_SPEC))
    if missing:
        sys.exit("an attribute detached from every sequence; refusing to claim success")

    print("done" + (" (dry run, nothing written)" if dry else ""))


if __name__ == "__main__":
    main()
