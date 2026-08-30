"""
Size-stratified HOTA decomposition for the Space-Tracker-MOT benchmark.

The headline MOT tables give one HOTA per tracker over a whole test half. This
script re-derives the *same* numbers detection-by-detection and attributes each
one to the pixel scale of the ground-truth object it belongs to, so the two
axes of HOTA can be read as a function of object size:

    DetRe(s) = TP(s) / GT(s)                 -- can the detector see it at all
    AssA(s)  = mean over TP(s) of A(c)       -- once seen, is the identity kept

Both are exact restrictions of TrackEval's own quantities. HOTA's third
ingredient, precision, is deliberately NOT split by size: a false positive has
no ground-truth object and therefore no ground-truth scale, so any per-bin
"DetPr" has to invent one from the predicted box. That column is still written
to the CSV (attributing each FP to its own predicted scale) because it is the
only way to form a per-bin DetA, but the figure uses DetRe / AssA, which need
no such convention.

Mechanics. TrackEval is used verbatim for loading and preprocessing, so the
inputs are byte-identical to the ones behind the published tables. The HOTA
inner loop is then re-implemented here -- copied from
``trackeval/metrics/hota.py`` -- because the upstream metric returns only
sequence totals and the attribution has to happen inside the per-timestep
Hungarian matching. ``--check`` re-aggregates the unbinned totals and compares
them against a published CSV; that gate is what makes the re-implementation
trustworthy.

Aggregation across sequences follows ``HOTA.combine_sequences``: counts are
summed, and AssA / LocA are HOTA_TP-weighted, which here is just "accumulate
numerator and denominator separately". Every quantity is finally averaged over
the 19 alpha thresholds, as in the standard HOTA.

Usage:
    python tools/compute_mot_size_curve.py \
        --workspace /tmp/hota_car_ws2 --benchmark spacetracker_car_car \
        --half car --edges 0:16:1 \
        --out docs/space_tracker/mot_size_curve_car.csv \
        --check docs/space_tracker/hota_car.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo("float").eps


def parse_edges(spec: str) -> np.ndarray:
    """'0:16:1' -> uniform edges; '0,5,8,12,20,32' -> explicit edges."""
    if ":" in spec:
        lo, hi, w = (float(v) for v in spec.split(":"))
        return np.arange(lo, hi + w / 2, w)
    return np.array([float(v) for v in spec.split(",")])


def _dataset(workspace: Path, benchmark: str, split: str, tracker: str):
    import trackeval

    cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg.update({
        "GT_FOLDER":        str(workspace / "gt"),
        "TRACKERS_FOLDER":  str(workspace / "trackers"),
        "OUTPUT_FOLDER":    str(workspace / "output"),
        "TRACKERS_TO_EVAL": [tracker],
        "CLASSES_TO_EVAL":  ["pedestrian"],
        "BENCHMARK":        benchmark,
        "SPLIT_TO_EVAL":    split,
        "PRINT_CONFIG":     False,
        "DO_PREPROC":       False,
        "SEQMAP_FOLDER":    str(workspace / "seqmaps"),
        "SEQMAP_FILE":      str(workspace / "seqmaps" / f"{benchmark}-{split}.txt"),
        "SKIP_SPLIT_FOL":   False,
    })
    return trackeval.datasets.MotChallenge2DBox(cfg)


def _scale(dets: np.ndarray) -> np.ndarray:
    """sqrt(area) of xywh boxes -- the same scale definition as the SOT figure."""
    if len(dets) == 0:
        return np.zeros(0)
    return np.sqrt(np.maximum(dets[:, 2], 0) * np.maximum(dets[:, 3], 0))


def eval_sequence(data: dict, edges: np.ndarray) -> dict:
    """HOTA for one sequence, with TP/FN/FP/AssA/LocA also split by GT scale.

    Mirrors trackeval.metrics.hota.HOTA.eval_sequence; the additions are the
    ``np.digitize`` calls and the per-bin accumulators.
    """
    nb = len(edges) - 1
    na = len(ALPHAS)
    z = lambda: np.zeros((na, nb))
    out = {
        "TP": np.zeros(na), "FN": np.zeros(na), "FP": np.zeros(na),
        "ass_num": np.zeros(na), "loc_num": np.zeros(na),
        "gt_bin": np.zeros(nb),
        "tp_bin": z(), "fp_bin": z(), "ass_num_bin": z(), "loc_num_bin": z(),
    }

    n_gt, n_tr = data["num_gt_ids"], data["num_tracker_ids"]
    # A GT box that falls outside the plotted range is dropped from the binned
    # accumulators but still counted in the unbinned totals, so --check stays
    # exact even when the range is narrower than the data.
    gt_bins_t, tr_bins_t = [], []
    for t in range(data["num_timesteps"]):
        gb = np.digitize(_scale(data["gt_dets"][t]), edges) - 1
        tb = np.digitize(_scale(data["tracker_dets"][t]), edges) - 1
        gt_bins_t.append(gb)
        tr_bins_t.append(tb)
        ok = (gb >= 0) & (gb < nb)
        if ok.any():
            out["gt_bin"] += np.bincount(gb[ok], minlength=nb)

    if data["num_tracker_dets"] == 0 or data["num_gt_dets"] == 0:
        out["FN"] += data["num_gt_dets"]
        out["FP"] += data["num_tracker_dets"]
        for t in range(data["num_timesteps"]):
            tb = tr_bins_t[t]
            ok = (tb >= 0) & (tb < nb)
            if ok.any():
                out["fp_bin"] += np.bincount(tb[ok], minlength=nb)[None, :]
        return out

    # --- global alignment (verbatim from TrackEval) -------------------------
    potential = np.zeros((n_gt, n_tr))
    gt_id_count = np.zeros((n_gt, 1))
    tr_id_count = np.zeros((1, n_tr))
    for t, (g, r) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        sim = data["similarity_scores"][t]
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        m = denom > EPS
        sim_iou[m] = sim[m] / denom[m]
        potential[g[:, None], r[None, :]] += sim_iou
        gt_id_count[g] += 1
        tr_id_count[0, r] += 1
    align = potential / (gt_id_count + tr_id_count - potential)

    matches_counts = [np.zeros((n_gt, n_tr)) for _ in ALPHAS]
    # flat records of every TP, so ass_a (only known after the loop) can be
    # gathered back onto size bins without a per-bin match matrix.
    rec_gt = [[] for _ in ALPHAS]
    rec_tr = [[] for _ in ALPHAS]
    rec_bin = [[] for _ in ALPHAS]

    from scipy.optimize import linear_sum_assignment

    for t, (g, r) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        tb, gb = tr_bins_t[t], gt_bins_t[t]
        if len(g) == 0:
            out["FP"] += len(r)
            ok = (tb >= 0) & (tb < nb)
            if ok.any():
                out["fp_bin"] += np.bincount(tb[ok], minlength=nb)[None, :]
            continue
        if len(r) == 0:
            out["FN"] += len(g)
            continue
        sim = data["similarity_scores"][t]
        score = align[g[:, None], r[None, :]] * sim
        mr, mc = linear_sum_assignment(-score)
        fp_all = np.bincount(tb[(tb >= 0) & (tb < nb)], minlength=nb)
        for a, alpha in enumerate(ALPHAS):
            keep = sim[mr, mc] >= alpha - EPS
            ar, ac = mr[keep], mc[keep]
            n = len(ar)
            out["TP"][a] += n
            out["FN"][a] += len(g) - n
            out["FP"][a] += len(r) - n
            gbin = gb[ar]
            ok = (gbin >= 0) & (gbin < nb)
            if n:
                out["loc_num"][a] += sim[ar, ac].sum()
                matches_counts[a][g[ar], r[ac]] += 1
                if ok.any():
                    out["tp_bin"][a] += np.bincount(gbin[ok], minlength=nb)
                    out["loc_num_bin"][a] += np.bincount(
                        gbin[ok], weights=sim[ar, ac][ok], minlength=nb)
                    rec_gt[a].append(g[ar][ok])
                    rec_tr[a].append(r[ac][ok])
                    rec_bin[a].append(gbin[ok])
            # FPs are attributed by their own predicted scale: matched tracker
            # dets are subtracted from the frame's predicted-scale histogram.
            matched = np.bincount(tb[ac][(tb[ac] >= 0) & (tb[ac] < nb)], minlength=nb)
            out["fp_bin"][a] += fp_all - matched

    for a in range(len(ALPHAS)):
        mc = matches_counts[a]
        ass_a = mc / np.maximum(1, gt_id_count + tr_id_count - mc)
        out["ass_num"][a] = np.sum(mc * ass_a)
        if rec_gt[a]:
            gi = np.concatenate(rec_gt[a])
            ri = np.concatenate(rec_tr[a])
            bi = np.concatenate(rec_bin[a])
            out["ass_num_bin"][a] = np.bincount(bi, weights=ass_a[gi, ri], minlength=nb)
    return out


def run_tracker(ds, tracker: str, seqs: list[str], edges: np.ndarray) -> dict:
    nb, na = len(edges) - 1, len(ALPHAS)
    tot = {"TP": np.zeros(na), "FN": np.zeros(na), "FP": np.zeros(na),
           "ass_num": np.zeros(na), "loc_num": np.zeros(na),
           "gt_bin": np.zeros(nb), "tp_bin": np.zeros((na, nb)),
           "fp_bin": np.zeros((na, nb)), "ass_num_bin": np.zeros((na, nb)),
           "loc_num_bin": np.zeros((na, nb))}
    for seq in seqs:
        raw = ds.get_raw_seq_data(tracker, seq)
        data = ds.get_preprocessed_seq_data(raw, "pedestrian")
        r = eval_sequence(data, edges)
        for k in tot:
            tot[k] = tot[k] + r[k]
    return tot


def summarise(tot: dict, edges: np.ndarray) -> tuple[dict, list[dict]]:
    tp, fn, fp = tot["TP"], tot["FN"], tot["FP"]
    det_a = tp / np.maximum(1, tp + fn + fp)
    ass_a = tot["ass_num"] / np.maximum(1, tp)
    overall = {
        "HOTA": float(np.mean(np.sqrt(det_a * ass_a))),
        "DetA": float(np.mean(det_a)),
        "AssA": float(np.mean(ass_a)),
        "DetRe": float(np.mean(tp / np.maximum(1, tp + fn))),
        "LocA": float(np.mean(tot["loc_num"] / np.maximum(1e-10, tp))),
    }
    rows = []
    nb = len(edges) - 1
    for b in range(nb):
        gt = tot["gt_bin"][b]
        tpb, fpb = tot["tp_bin"][:, b], tot["fp_bin"][:, b]
        detre = tpb / max(gt, 1)
        assa = tot["ass_num_bin"][:, b] / np.maximum(1, tpb)
        deta = tpb / np.maximum(1, tpb + (gt - tpb) + fpb)
        loca = tot["loc_num_bin"][:, b] / np.maximum(1e-10, tpb)
        # AssA is a mean over true positives, so a bin a tracker recovered
        # nothing in has no AssA -- not an AssA of zero. Writing 0 there would
        # read as "associates terribly" instead of "was never given the chance",
        # and TrackTrack, which discards a third of the ground-truth boxes even
        # under the oracle, has exactly such bins.
        rows.append({
            "bin_lo": edges[b], "bin_hi": edges[b + 1],
            "n_gt": int(round(gt)), "n_tp": int(round(tpb.mean())),
            "DetRe": float(np.mean(detre)),
            "AssA": float(np.mean(assa)) if gt and tpb.sum() else float("nan"),
            "DetA": float(np.mean(deta)),
            "LocA": float(np.mean(loca)) if gt and tpb.sum() else float("nan"),
            "HOTA_re": float(np.mean(np.sqrt(detre * assa))),
        })
    return overall, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--half", required=True, help="tag written into the CSV (car / nocar)")
    ap.add_argument("--edges", default="0:16:1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", type=Path, help="published HOTA csv to verify against")
    ap.add_argument("--trackers", nargs="*")
    args = ap.parse_args()

    edges = parse_edges(args.edges)
    troot = args.workspace / "trackers" / f"{args.benchmark}-{args.split}"
    trackers = args.trackers or sorted(d.name for d in troot.iterdir() if (d / "data").is_dir())
    seqmap = args.workspace / "seqmaps" / f"{args.benchmark}-{args.split}.txt"
    seqs = [l.strip() for i, l in enumerate(seqmap.read_text().splitlines())
            if i and l.strip()]
    print(f"[{args.half}] {len(trackers)} trackers x {len(seqs)} sequences, "
          f"{len(edges) - 1} bins over [{edges[0]:g},{edges[-1]:g}] px")

    published = {}
    if args.check:
        for r in csv.DictReader(open(args.check)):
            published[r["tracker"]] = r

    out_rows = []
    for tr in trackers:
        ds = _dataset(args.workspace, args.benchmark, args.split, tr)
        tot = run_tracker(ds, tr, seqs, edges)
        overall, rows = summarise(tot, edges)
        msg = (f"  {tr:<13} HOTA={overall['HOTA']:.4f} DetA={overall['DetA']:.4f} "
               f"AssA={overall['AssA']:.4f}")
        if tr in published:
            d = max(abs(overall[k] - float(published[tr][k])) for k in ("HOTA", "DetA", "AssA"))
            msg += f"   max|delta| vs published = {d:.2e}"
        print(msg)
        for row in rows:
            out_rows.append({"half": args.half, "tracker": tr, **row})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[save] {args.out}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
