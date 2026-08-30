"""
Build the WACV headline SOT table for Space-Tracker-SOT.

Two deliberate departures from the NeurIPS tables:

1. **Space-Tracker-SOT is one dataset.** The NeurIPS table pooled the three
   source datasets with an equal-weight macro-of-macro mean, which keeps the
   source boundaries alive in the metric. Here every released sequence carries
   equal weight: we aggregate straight over the 395 sequences. SV248S supplies
   246 of them, so it naturally dominates -- that is a property of the benchmark
   we built, not something to average away.

2. **OOTB is scored axis-aligned for everyone.** The NeurIPS numbers scored all
   trackers with polygon IoU (see docs/space_tracker/sot_release_metrics.md),
   while the text claimed a two-branch protocol. Rather than restore the split,
   we collapse the OOTB oriented GT and every prediction to axis-aligned boxes,
   so all methods solve the same localisation task. Polygon/OBB scoring becomes
   a supplementary analysis instead of a branch inside the headline number.

Outputs a booktabs table matching the NeurIPS layout, plus an abstention column
(the fraction of frames a tracker emitted no box; our protocol scores those as
IoU 0, which structurally favours trackers that never declare loss).

Usage:
    python tools/make_wacv_sot_table.py \
        --runs   /path/to/SOT_whole_dataset_<date> \
        --release /path/to/release/space_tracker \
        --out    wacv-2027-author-kit-template/tables/sot_main_table.tex
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning_modules.sot_metrics import _compute_from_sequences
from tools.aggregate_sot_release import records_by_sequence

TINY_THRESH_PX = 8.0

# display name, venue, and the citation key used in the WACV bibliography
TRACKERS = [
    ("siamrpn",    "SiamRPN++",   "CVPR 2019",  "li2019siamrpn++",  True),
    ("siamfc",     "SiamFC",      "ECCVW 2016", "bertinetto2016siamfc", True),
    ("ostrack",    "OSTrack-384", "ECCV 2022",  "ye2022joint",      False),
    ("odtrack",    "ODTrack",     "AAAI 2024",  "zheng2024odtrack", False),
    ("lorat",      "LoRAT-g378",  "ECCV 2024",  "lin2024tracking",  False),
    ("smalltrack", "SmallTrack",  "TGRS 2023",  "xue2023smalltrack", True),
    ("sam2",       "SAM 2",       "ICCV 2024",  "ravi2024sam",      False),
    ("samurai",    "SAMURAI",     "TIP 2026",   "yang2026samurai",  False),
    ("sam3",       "SAM 3",       "ICLR 2026",  "carion2026sam3",   False),
]

# visual row groups in the headline table (label, tracker keys in display order)
GROUPS = [
    ("Siamese correlation",    ["siamrpn", "siamfc"]),
    ("Transformer trackers",   ["ostrack", "odtrack", "lorat"]),
    ("Small-object specific",  ["smalltrack"]),
    ("Video foundation models", ["sam2", "samurai", "sam3"]),
]

DATASETS = ["ootb", "satsot", "sv248s"]


def released_sequences(release_root: Path) -> dict[str, dict[str, float]]:
    """(source dataset) → {source video id: median sqrt(area) in px}."""
    ann = release_root / "sot" / "annotations" / "space_tracker_sot.json"
    data = json.loads(ann.read_text())
    out: dict[str, dict[str, float]] = {}
    for v in data["videos"]:
        ds = v["source_dataset"]
        sid = v["source_sequence_id"]
        vid = sid[len(ds) + 1:] if sid.startswith(ds + "/") else sid
        out.setdefault(ds, {})[vid] = float(v["median_sqrt_area_px"])
    return out


def find_run(runs_root: Path, tracker: str, dataset: str) -> Path | None:
    """Newest run directory for this tracker/dataset that has a per-frame dump."""
    cands = sorted(runs_root.glob(f"{tracker}/*_first_frame_{dataset}_*/per_image_metrics.json"))
    return cands[-1] if cands else None


def collect(runs_root: Path, tracker: str, released: dict[str, dict[str, float]]):
    """
    Gather every released sequence for one tracker across the three sources.

    OOTB is always rescored axis-aligned, for every tracker, so the headline
    number measures one localisation task.
    """
    seqs: dict[str, list] = {}
    answered: dict[str, list] = {}
    sizes: dict[str, float] = {}
    for ds in DATASETS:
        pim = find_run(runs_root, tracker, ds)
        if pim is None:
            raise FileNotFoundError(f"no run found for {tracker}/{ds}")
        keep = set(released[ds])
        s, a = records_by_sequence(pim, keep, rescore_aabb=(ds == "ootb"))
        for vid, recs in s.items():
            key = f"{ds}/{vid}"
            seqs[key] = recs
            answered[key] = a[vid]
            sizes[key] = released[ds][vid]
    return seqs, answered, sizes


def summarise(seqs, answered, sizes, tiny_only: bool) -> dict:
    keys = [k for k in seqs if (sizes[k] < TINY_THRESH_PX) or not tiny_only]
    if tiny_only:
        keys = [k for k in seqs if sizes[k] < TINY_THRESH_PX]
    metrics = _compute_from_sequences([seqs[k] for k in keys])
    per_seq_abstain = [
        1.0 - sum(answered[k]) / len(answered[k]) for k in keys if answered[k]
    ]
    metrics["abstention"] = sum(per_seq_abstain) / len(per_seq_abstain)
    metrics["n_seq"] = len(keys)
    return metrics


def fmt_col(values: list[float], higher_better: bool = True) -> list[str]:
    """Bold the best, underline the second best."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher_better)
    best, second = order[0], order[1]
    out = []
    for i, v in enumerate(values):
        s = f"{v:.3f}"
        if i == best:
            out.append(rf"\textbf{{{s}}}")
        elif i == second:
            out.append(rf"\underline{{{s}}}")
        else:
            out.append(s)
    return out


def pad_pct(v: float) -> str:
    """Right-align one-digit percentages under two-digit ones."""
    s = f"{v:.1f}"
    return (r"\phantom{1}" + s) if v < 10 else s


def fmt_abst(values: list[float]) -> list[str]:
    """Abstention column: two decimals, lowest is best, ties share a rank.

    Two decimals because several trackers abstain on the same 0.8% of frames at
    one decimal and the reader cannot tell them apart (reviewer request).
    Ranking is done on the rounded strings, so anything printed as equal is
    marked as equal.
    """
    shown = [f"{v:.2f}" for v in values]
    uniq = sorted({float(x) for x in shown})
    best = uniq[0]
    second = uniq[1] if len(uniq) > 1 else None
    out = []
    for s in shown:
        padded = (r"\phantom{1}" + s) if float(s) < 10 else s
        if float(s) == best:
            out.append(rf"\textbf{{{padded}}}")
        elif second is not None and float(s) == second:
            out.append(rf"\underline{{{padded}}}")
        else:
            out.append(padded)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    released = released_sequences(Path(args.release))
    n_total = sum(len(v) for v in released.values())
    print(f"Released sequences: {n_total} "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(released.items())))

    rows = []
    for key, name, venue, cite, third_party in TRACKERS:
        seqs, answered, sizes = collect(Path(args.runs), key, released)
        whole = summarise(seqs, answered, sizes, tiny_only=False)
        tiny = summarise(seqs, answered, sizes, tiny_only=True)
        rows.append((key, name, venue, cite, third_party, whole, tiny))
        print(f"  {name:12s} {whole['n_seq']:3d} seq  SR={whole['success_auc']:.4f}  "
              f"tiny({tiny['n_seq']}) SR={tiny['success_auc']:.4f}  "
              f"abstain={whole['abstention']*100:.1f}%")

    cols = {
        "w_sr":  fmt_col([r[5]["success_auc"] for r in rows]),
        "w_npr": fmt_col([r[5]["norm_precision_auc"] for r in rows]),
        "w_pr":  fmt_col([r[5]["precision_auc"] for r in rows]),
        "w_ab":  fmt_abst([r[5]["abstention"] * 100 for r in rows]),
        "t_sr":  fmt_col([r[6]["success_auc"] for r in rows]),
        "t_npr": fmt_col([r[6]["norm_precision_auc"] for r in rows]),
        "t_p5":  fmt_col([r[6]["precision_5"] for r in rows]),
    }

    n_tiny = rows[0][6]["n_seq"]
    L = []
    L.append("% Generated by tools/make_wacv_sot_table.py -- do not edit by hand.")
    L.append("% Requires booktabs (already loaded by wacv.sty).")
    # Single column, not table*. A table* costs full-page-width vertical space
    # however narrow its tabular is, so widening the column gaps to fill
    # \textwidth bought nothing and only spread the numbers out. At
    # \footnotesize with 2pt gaps and no Year column the tabular measures
    # 229pt against a 237pt column, so it fits natively -- no \resizebox, whose
    # arbitrary scale factor would leave the digits at a size unrelated to the
    # body text. Venue/year is dropped rather than scaled away: it is metadata
    # the citation already carries, and it was the cheapest column to lose.
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{2pt}")
    L.append(r"\renewcommand{\arraystretch}{1.05}")
    # The caption used to restate the definition of every metric. All five are
    # given in full, with equations, in the supplementary metrics section, so
    # the restatement cost ten lines of full-width text and said nothing new.
    L.append(
        r"\caption{\textbf{Single-object tracking on Space-Tracker-SOT.} "
        f"Metrics are averaged over the ${n_total}$ released sequences with equal "
        r"weight per sequence (\cref{sec:suppl_metrics_sot}); the \textbf{tiny} "
        f"half restricts them to the ${n_tiny}$ sequences whose median "
        r"ground-truth $\sqrt{\mathrm{area}}$ is below $8$\,px, where "
        r"CLE-based PR saturates and is replaced by \textbf{P@5}, the fraction "
        r"of frames with $\mathrm{CLE}<5$\,px. \textbf{Abst.} is the "
        r"percentage of frames with no predicted box; those frames already "
        r"count as zero overlap inside SR, so it imposes no second penalty and "
        r"is shown only because it separates a tracker that localises badly "
        r"from one that declines to answer. Predictions and OOTB's oriented "
        r"ground truth are converted to horizontal boxes before scoring, so all "
        r"methods face one localisation task; \cref{tab:sot_obb_supp} gives the "
        r"oriented-box analysis. \textbf{Bold} / \underline{underline}: best / "
        r"second best. $^\dag$~third-party re-implementation or checkpoint.}"
    )
    L.append(r"\label{tab:sot_results}")
    # an empty spacer column (7) separates the two halves without a vertical rule
    L.append(r"\begin{tabular}{@{}lcccc c ccc@{}}")
    L.append(r"\toprule")
    L.append(
        r"& \multicolumn{4}{c}{\textbf{All} \, "
        f"({n_total} seq)"
        r"} & & \multicolumn{3}{c}{\textbf{Tiny} \, "
        f"({n_tiny} seq, "
        r"$<8$\,px)} \\"
    )
    L.append(r"\cmidrule(lr){2-5} \cmidrule(lr){7-9}")
    L.append(
        r"\textbf{Method} & SR\,$\uparrow$ & NPR\,$\uparrow$ & "
        r"PR\,$\uparrow$ & Abst.\,$\downarrow$ & & SR\,$\uparrow$ & NPR\,$\uparrow$ & "
        r"P@5\,$\uparrow$ \\"
    )
    L.append(r"\midrule")

    index = {r[0]: i for i, r in enumerate(rows)}
    for g, (group_label, keys) in enumerate(GROUPS):
        if g:
            L.append(r"\addlinespace[1pt]")
        L.append(r"\multicolumn{9}{@{}l}{\itshape " + group_label + r"} \\")
        for key in keys:
            i = index[key]
            _, name, venue, cite, third_party, _, _ = rows[i]
            dag = r"$^\dag$" if third_party else ""
            L.append(
                f"{name}~\\cite{{{cite}}}{dag} & "
                f"{cols['w_sr'][i]} & {cols['w_npr'][i]} & {cols['w_pr'][i]} & "
                f"{cols['w_ab'][i]} & & "
                f"{cols['t_sr'][i]} & {cols['t_npr'][i]} & {cols['t_p5'][i]} \\\\"
            )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
