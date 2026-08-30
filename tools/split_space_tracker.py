#!/usr/bin/env python
"""Train / val / test split for Space-Tracker, scene-disjoint and stratified.

Two rules shape it.

**The unit is the parent scene, not the sequence.**  Half the SOT half is six
Jilin-1 parent videos cut into hundreds of crops; putting two crops of one scene
on opposite sides of the split lets a tracker meet the test background during
training.  So every sequence sharing a `parent_scene_group` moves together.  The
two task halves are split independently — an SOT sequence and an MOT sequence
benchmark different algorithms, so sharing a scene across them is not leakage.

**Balance is checked on every axis at once.**  Source dataset, object category,
object size, motion, density, and — for SOT, where they exist — the unified
sequence attributes.  Greedy assignment, largest group first, each group going
to whichever split its arrival leaves closest to the target proportions, with a
coverage pass first so no category is missing from test.

The cost of rule one is real and is reported rather than hidden: with six parent
scenes carrying 246 of the 395 SOT sequences, SV248S can be divided at most six
ways, and the achieved proportions cannot land exactly on the targets.

Usage
-----
    python tools/split_space_tracker.py --ratios 0.7 0.1 0.2 \
        --out docs/space_tracker/splits.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

SPLITS = ("train", "val", "test")

#: Size buckets on the median sqrt(area) of a sequence's boxes, in pixels.
#: 8 px is the sub-8-pixel regime the benchmark is built around; 32 px is the
#: small-object cutoff used to select the release.
SIZE_EDGES = (8.0, 16.0, 32.0)

#: Speed buckets on the fastest track in the sequence, px/frame. The 4.7 px
#: boundary is the 75th percentile of the MOT distribution and recovers 14 of
#: the 19 sequences a human tagged ``fast_motion`` during review; the five it
#: misses sit between 2.9 and 4.2 px/frame.
SPEED_EDGES = (1.5, 4.7)

#: Crowding buckets on the busiest frame.
DENSITY_EDGES = (5, 30, 100)


def bucket(value: float | None, edges) -> str:
    if value is None:
        return "na"
    for i, e in enumerate(edges):
        if value < e:
            return f"<{e:g}" if i == 0 else f"{edges[i - 1]:g}-{e:g}"
    return f">={edges[-1]:g}"


def strata_of(rec: dict, attrs: dict[str, list[str]]) -> list[str]:
    """Every label this sequence contributes to. Balance is checked on each."""
    out = [
        "all",
        f"ds={rec['source_dataset']}",
        f"cat={rec['category']}",
        f"size={bucket(rec.get('sqrt_area_px_median'), SIZE_EDGES)}",
        f"speed={bucket(rec.get('px_per_frame_max'), SPEED_EDGES)}",
        f"dens={bucket(rec.get('objects_per_frame_max'), DENSITY_EDGES)}",
        # The joint dataset x category cell is what actually goes wrong when a
        # split is balanced on each axis separately.
        f"ds+cat={rec['source_dataset']}/{rec['category']}",
    ]
    out += [f"attr={a}" for a in attrs.get(rec["key"], ())]
    return out


def load_attributes(release: Path) -> dict[str, list[str]]:
    """Unified sequence attributes, which only the SOT half carries."""
    path = release / "sot" / "annotations" / "space_tracker_sot.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {f"sot/{v['name']}": list(v.get("unified_attributes") or [])
            for v in data["videos"]}


#: Not every axis matters equally. A class missing from test makes the benchmark
#: unusable for that class; a slightly lopsided density bucket does not. Without
#: this the optimiser trades ``car-large`` away to keep the overall val count
#: tidy, because one SV248S parent scene happens to be exactly the size of a 10%
#: val split.
STRATUM_WEIGHT = {"cat=": 6.0, "ds+cat=": 3.0, "attr=": 1.5}


def weight_of(label: str) -> float:
    for prefix, w in STRATUM_WEIGHT.items():
        if label.startswith(prefix):
            return w
    return 1.0


def cost(counts: dict, totals: dict, ratios: dict) -> float:
    """Squared deviation from the target share, summed over every stratum.

    Divided by the stratum total so a 10-sequence category weighs as much as a
    300-sequence one — otherwise ``all`` and ``ds=sv248s`` drown out everything
    the split is actually at risk of getting wrong.
    """
    c = 0.0
    for k, tot in totals.items():
        w = weight_of(k) / tot
        for s in SPLITS:
            want = ratios[s] * tot
            c += w * (counts[s].get(k, 0) - want) ** 2
    return c


def assign(groups: list[tuple], totals: dict, ratios: dict) -> dict[int, str]:
    """Greedy: biggest group first, into whichever split it hurts least."""
    counts = {s: collections.Counter() for s in SPLITS}
    out: dict[int, str] = {}
    for gid, labels, size in sorted(groups, key=lambda g: -g[2]):
        best, best_cost = None, math.inf
        for s in SPLITS:
            counts[s].update(labels)
            c = cost(counts, totals, ratios)
            counts[s].subtract(labels)
            if c < best_cost:
                best, best_cost = s, c
        counts[best].update(labels)
        out[gid] = best
    return out


def hill_climb(assignment: dict[int, str], groups: list[tuple], totals: dict,
               ratios: dict, rounds: int = 200) -> dict[int, str]:
    """Move one group at a time, keeping any move that lowers the cost.

    Greedy alone places the largest groups first, when it has almost no
    information to place them with. On the SOT half that decides everything:
    five parent scenes hold 219 of the 395 sequences, and one of them holds 24
    of the 37 ``car-large`` sequences, so the first placement either works or
    the whole axis is lost.
    """
    counts = {s: collections.Counter() for s in SPLITS}
    by_gid = {g[0]: g for g in groups}
    for gid, s in assignment.items():
        counts[s].update(by_gid[gid][1])
    best = cost(counts, totals, ratios)
    for _ in range(rounds):
        improved = False
        for gid, labels, _size in groups:
            cur = assignment[gid]
            for s in SPLITS:
                if s == cur:
                    continue
                counts[cur].subtract(labels)
                counts[s].update(labels)
                c = cost(counts, totals, ratios)
                if c < best - 1e-9:
                    best, assignment[gid], cur, improved = c, s, s, True
                else:
                    counts[s].subtract(labels)
                    counts[cur].update(labels)
        if not improved:
            break
    return assignment


def repair_coverage(assignment: dict[int, str], groups: list[tuple],
                    totals: dict, ratios: dict, must_cover: list[str]) -> dict[int, str]:
    """Make sure test, then val, sees every label that has enough groups.

    A label carried by only one group cannot appear in more than one split, so
    it is left where the greedy pass put it.
    """
    by_gid = {g[0]: g for g in groups}
    for label in must_cover:
        holders = [g for g in groups if label in g[1]]
        if len(holders) < 2:
            continue
        for target in ("test", "val"):
            if any(assignment[g[0]] == target for g in holders):
                continue
            donors = [g for g in holders if assignment[g[0]] == "train"]
            if not donors:
                donors = [g for g in holders if assignment[g[0]] != target]
            if not donors:
                continue
            # Move the smallest donor: it disturbs the other axes least.
            move = min(donors, key=lambda g: g[2])
            assignment[move[0]] = target
    return assignment


def report(recs: list[dict], assignment: dict[int, str], half: str,
           attrs: dict) -> None:
    per = collections.Counter(assignment[r["parent_scene_group"]] for r in recs)
    n = len(recs)
    print(f"\n=== {half.upper()}  {n} sequences, "
          f"{len({r['parent_scene_group'] for r in recs})} scene groups ===")
    print("  " + "  ".join(f"{s}={per[s]} ({100*per[s]/n:.1f}%)" for s in SPLITS))

    for axis in ("source_dataset", "category"):
        print(f"  -- by {axis}")
        keys = sorted({r[axis] for r in recs}, key=str)
        for k in keys:
            rows = [r for r in recs if r[axis] == k]
            c = collections.Counter(assignment[r["parent_scene_group"]] for r in rows)
            print(f"     {str(k):12s} n={len(rows):4d}  " +
                  "  ".join(f"{s}={c[s]:4d}" for s in SPLITS))

    for axis, edges, key in (("size", SIZE_EDGES, "sqrt_area_px_median"),
                             ("speed", SPEED_EDGES, "px_per_frame_max")):
        print(f"  -- by {axis}")
        rows_by = collections.defaultdict(list)
        for r in recs:
            rows_by[bucket(r.get(key), edges)].append(r)
        for k in sorted(rows_by):
            c = collections.Counter(assignment[r["parent_scene_group"]] for r in rows_by[k])
            print(f"     {k:12s} n={len(rows_by[k]):4d}  " +
                  "  ".join(f"{s}={c[s]:4d}" for s in SPLITS))

    if half == "sot":
        print("  -- by unified attribute")
        used = collections.Counter(a for r in recs for a in attrs.get(r["key"], ()))
        for a in sorted(used):
            rows = [r for r in recs if a in attrs.get(r["key"], ())]
            c = collections.Counter(assignment[r["parent_scene_group"]] for r in rows)
            print(f"     {a:12s} n={len(rows):4d}  " +
                  "  ".join(f"{s}={c[s]:4d}" for s in SPLITS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path,
                    default=Path("docs/space_tracker/sequence_metadata.json"))
    ap.add_argument("--release", type=Path,
                    default=Path("/data/anon/release/space_tracker"))
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--out", type=Path, default=Path("docs/space_tracker/splits.json"))
    args = ap.parse_args()

    ratios = dict(zip(SPLITS, args.ratios))
    recs = json.loads(args.metadata.read_text())["sequences"]
    attrs = load_attributes(args.release)

    result: dict[str, dict[str, str]] = {}
    for half in ("sot", "mot"):
        rows = [r for r in recs if r["half"] == half]
        labels_by_group: dict[int, collections.Counter] = collections.defaultdict(
            collections.Counter)
        size_by_group: collections.Counter = collections.Counter()
        for r in rows:
            g = r["parent_scene_group"]
            labels_by_group[g].update(strata_of(r, attrs))
            size_by_group[g] += 1
        totals: collections.Counter = collections.Counter()
        for g, lab in labels_by_group.items():
            totals.update(lab)

        groups = [(g, labels_by_group[g], size_by_group[g]) for g in labels_by_group]
        assignment = assign(groups, totals, ratios)
        assignment = hill_climb(assignment, groups, totals, ratios)
        must = [k for k in totals if k.startswith(("cat=", "ds=", "attr="))]
        assignment = repair_coverage(assignment, groups, totals, ratios, must)
        # Coverage repair moves groups for a reason the cost function does not
        # know about, so let the search settle again afterwards. Anything it
        # would undo was not needed for coverage in the first place.
        assignment = hill_climb(assignment, groups, totals, ratios)
        assignment = repair_coverage(assignment, groups, totals, ratios, must)

        for r in rows:
            result.setdefault(half, {})[r["key"]] = assignment[r["parent_scene_group"]]
        report(rows, assignment, half, attrs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ratios": ratios,
        "unit": "parent_scene_group, within each task half",
        "splits": result,
    }, indent=2) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
