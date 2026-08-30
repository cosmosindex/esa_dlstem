"""
Build the SOT clip-length ablation table (T = 16 / 32 / 64).

All three points are scored on the SAME basis, which is the only thing that
makes them comparable:

* the **71-sequence Space-Tracker SOT test split** (not the source datasets'
  own test splits -- those overlap ours in only 5 of 71 sequences),
* tau = 0, so nothing is filtered and the clip length is the only variable,
* OOTB reduced to horizontal boxes, exactly as in the main table.

T=16 and T=64 were run with `SOT_SEQ_WHITELIST` pointing at the test split.
T=32 is not re-run: it is extracted from the existing whole-dataset tau=0 dumps
by applying the same whitelist, which costs no GPU and guarantees the baseline
is the identical inference the main table reports.

Usage:
    python tools/make_wacv_clip_table.py \
        --clip16 /work/anon/experiments/SOT_clip16 \
        --clip32 /work/anon/experiments/SOT_tau0 \
        --clip64 /work/anon/experiments/SOT_clip64 \
        --whitelist docs/space_tracker/sot_test_sequences.txt \
        --out wacv-2027-author-kit-template/tables/sot_clip_ablation.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning_modules.sot_metrics import _compute_from_sequences
from tools.aggregate_sot_release import records_by_sequence

DATASETS = ["ootb", "satsot", "sv248s"]
TRACKERS = [
    ("siamfc",     "SiamFC"),
    ("siamrpn",    "SiamRPN++"),
    ("smalltrack", "SmallTrack"),
    ("ostrack",    "OSTrack-384"),
    ("odtrack",    "ODTrack"),
    ("lorat",      "LoRAT-g378"),
    ("sam2",       "SAM 2"),
    ("samurai",    "SAMURAI"),
    ("sam3",       "SAM 3"),
]
CLIPS = [16, 32, 64]


def find_run(root: Path, tracker: str, dataset: str) -> Path | None:
    c = sorted(root.glob(f"{tracker}*_first_frame_{dataset}_*/per_image_metrics.json"))
    return c[-1] if c else None


def collect(root: Path, tracker: str, keep: dict[str, set[str]]):
    """dataset/video -> {frame_id: record}, or None if a run is missing."""
    out = {}
    for ds in DATASETS:
        pim = find_run(root, tracker, ds)
        if pim is None:
            return None
        per_seq, _ = records_by_sequence(pim, keep[ds], rescore_aabb=(ds == "ootb"))
        for vid, recs in per_seq.items():
            out[f"{ds}/{vid}"] = {r.frame_id: r for r in recs}
    return out or None


def score_on(collected: dict, frames: dict[str, set]) -> dict:
    """Score only the frames in `frames`, so every T sees an identical frame set."""
    seqs = []
    for key, byfid in collected.items():
        sel = [byfid[f] for f in sorted(frames.get(key, ())) if f in byfid]
        if sel:
            seqs.append(sel)
    m = _compute_from_sequences(seqs)
    m["_n_seq"] = len(seqs)
    m["_n_frame"] = sum(len(x) for x in seqs)
    return m


def _unused_score(root: Path, tracker: str, keep: dict[str, set[str]]) -> dict | None:
    """keep maps dataset -> the bare video ids of the test split IN that dataset.

    It must be per-dataset. Video ids are only unique within a dataset -- both
    OOTB and SatSOT have a `car_10` -- so a flat set of bare ids silently admits
    same-named sequences from the wrong dataset. That does not affect T=16/64,
    whose runs were already restricted to the test split at inference time, but
    it inflated the T=32 baseline (extracted from the whole-dataset dumps) from
    71 to 82 sequences and made the three points incomparable.
    """
    seqs = {}
    for ds in DATASETS:
        pim = find_run(root, tracker, ds)
        if pim is None:
            return None
        per_seq, _ = records_by_sequence(pim, keep[ds], rescore_aabb=(ds == "ootb"))
        for vid, recs in per_seq.items():
            seqs[f"{ds}/{vid}"] = recs
    if not seqs:
        return None
    m = _compute_from_sequences(list(seqs.values()))
    m["_n_seq"] = len(seqs)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip16", required=True)
    ap.add_argument("--clip32", required=True)
    ap.add_argument("--clip64", required=True)
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wl_raw = [l.strip() for l in Path(args.whitelist).read_text().splitlines() if l.strip()]
    keep: dict[str, set[str]] = {ds: set() for ds in DATASETS}
    for w in wl_raw:
        ds, vid = w.split("/", 1)
        if ds not in keep:
            raise SystemExit(f"whitelist entry {w!r} names an unknown dataset")
        keep[ds].add(vid)
    print(f"whitelist: {len(wl_raw)} sequences "
          + str({d: len(v) for d, v in keep.items()}))

    roots = {16: Path(args.clip16), 32: Path(args.clip32), 64: Path(args.clip64)}
    data: dict[str, dict[int, dict]] = {}
    for key, name in TRACKERS:
        got = {T: collect(roots[T], key, keep) for T in CLIPS}
        missing = [T for T, v in got.items() if v is None]
        if missing:
            print(f"  {name:12s} MISSING at T={missing}")
            data[key] = {}
            continue

        # Clip slicing drops the tail of a sequence that cannot fill a whole
        # clip, so a larger T evaluates FEWER frames -- and the dropped frames
        # are the late ones, where tracking error has accumulated most. Scoring
        # each T on its own frames therefore rewards long clips for skipping the
        # hard part: it moves even SiamFC, which has no temporal state and
        # cannot respond to T at all. Intersecting the frame sets removes that
        # artefact and leaves T as the only variable.
        common: dict[str, set] = {}
        for kseq in set.intersection(*(set(v) for v in got.values())):
            fs = set.intersection(*(set(got[T][kseq]) for T in CLIPS))
            if fs:
                common[kseq] = fs
        data[key] = {T: score_on(got[T], common) for T in CLIPS}
        raw = {T: sum(len(x) for x in got[T].values()) for T in CLIPS}
        d = data[key]
        row = "  ".join(f"T={T}: SR={d[T]['success_auc']:.4f}" for T in CLIPS)
        print(f"  {name:12s} {row}   common {d[32]['_n_seq']} seq / "
              f"{d[32]['_n_frame']} frames (raw {raw})")

    L = []
    L.append("% Generated by tools/make_wacv_clip_table.py -- do not edit by hand.")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Clip-length ablation on the Space-Tracker-SOT \textbf{test} "
        f"split ({len(wl_raw)} sequences). "
        r"$T$ is the number of frames a tracker is given per clip. All three "
        r"settings are scored on the same sequences at $\tau=0$ with OOTB "
        r"reduced to horizontal boxes, so $T$ is the only variable; $T=32$ is "
        r"the setting used throughout the paper. $\Delta$ columns are relative "
        r"to $T=32$. $T$ is not merely a batching parameter: ground truth "
        r"prompts the tracker once, on the first frame of the video, and at "
        r"every subsequent clip boundary the tracker is re-prompted with "
        r"\emph{its own prediction} from the previous clip's last frame. A "
        r"shorter $T$ therefore replaces the template with the tracker's own "
        r"output more often, compounding any drift already present, and this "
        r"applies whether or not a method carries temporal state -- SiamFC, a "
        r"purely frame-wise matcher, moves by $0.115$ SR between $T{=}16$ and "
        r"$T{=}64$. Because clip slicing discards the tail of a sequence that "
        r"cannot fill a whole clip, and those late frames are the hardest, each "
        r"$T$ would otherwise be scored on a different frame set; all three "
        r"columns are therefore computed on the intersection of the frames "
        r"evaluated at every $T$ ($30{,}245$ frames over $71$ sequences).}")
    L.append(r"\label{tab:sot_clip}")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\renewcommand{\arraystretch}{1.12}")
    L.append(r"\resizebox{\columnwidth}{!}{%")
    L.append(r"\begin{tabular}{l *{3}{>{\centering\arraybackslash}p{3.0em}} "
             r"*{2}{>{\centering\arraybackslash}p{3.2em}}}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c}{\textbf{SR}} "
             r"& \multicolumn{2}{c}{\textbf{$\Delta$ vs. $T{=}32$}} \\")
    L.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}")
    L.append(r" & $T{=}16$ & $T{=}32$ & $T{=}64$ & $T{=}16$ & $T{=}64$ \\")
    L.append(r"\midrule")
    for key, name in TRACKERS:
        d = data.get(key, {})
        if 32 not in d:
            L.append(f"{name} & -- & -- & -- & -- & -- \\\\")
            continue
        base = d[32]["success_auc"]
        cells = []
        for T in CLIPS:
            cells.append(f"{d[T]['success_auc']:.3f}" if T in d else "--")
        for T in (16, 64):
            cells.append(f"{d[T]['success_auc'] - base:+.3f}" if T in d else "--")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
