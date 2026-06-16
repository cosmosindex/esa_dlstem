#!/usr/bin/env python
"""Write a human-readable FAIRMOT_RESULTS.md for the unattended FairMOT
pipeline (run_fairmot_pipeline.sh), so the results + any problems are visible
at a glance the next morning.

Reads, if present, inside <pipeline_dir>:
  pipeline_status.json   stage statuses + messages the orchestrator wrote
  hota_summary.csv       compute_hota.py output (HOTA/DetA/AssA/MOTA/IDF1 rows)
  fairmot_<ds>_*/test_metrics.json   per-dataset eval detail (Pr/Re/F1, fps)

Usage: python tools/fairmot_write_summary.py <pipeline_dir>
"""
import csv
import glob
import json
import os
import sys
from datetime import datetime


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    pdir = sys.argv[1]
    status = _load_json(os.path.join(pdir, "pipeline_status.json")) or {}
    lines = []
    A = lines.append

    A(f"# FairMOT union pipeline — results\n")
    A(f"_generated {datetime.now():%Y-%m-%d %H:%M:%S} — dir `{pdir}`_\n")

    overall = status.get("overall", "unknown")
    A(f"**Overall status: {overall}**\n")

    # ---- stage table ----
    A("## Stages\n")
    A("| stage | status | note |")
    A("|---|---|---|")
    for st in status.get("stages", []):
        A(f"| {st.get('name','')} | {st.get('status','')} | {st.get('note','')} |")
    A("")

    # ---- training ----
    tr = status.get("train", {})
    if tr:
        A("## Training\n")
        A(f"- best epoch: **{tr.get('best_epoch','?')}**, "
          f"val det_loss = **{tr.get('best_val_det','?')}**")
        A(f"- checkpoint: `{tr.get('best_ckpt','?')}`")
        A(f"- W&B run: `{tr.get('wandb','esa-dlstem / fairmot_hrnet18_union')}`")
        A(f"- train log: `{tr.get('log','?')}`")
        A("")

    # ---- HOTA table ----
    hota_csv = os.path.join(pdir, "hota_summary.csv")
    if os.path.isfile(hota_csv):
        A("## Tracking metrics (compute_hota.py)\n")
        with open(hota_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            cols = ["dataset", "tracker", "HOTA", "DetA", "AssA",
                    "MOTA", "IDF1", "IDsw"]
            cols = [c for c in cols if c in rows[0]]
            A("| " + " | ".join(cols) + " |")
            A("|" + "|".join(["---"] * len(cols)) + "|")
            for r in rows:
                A("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        A("")
    else:
        A("## Tracking metrics\n\n_hota_summary.csv not found — HOTA stage did "
          "not complete (see stages above)._\n")

    # ---- per-dataset detection detail ----
    A("## Per-dataset detection detail (eval test_metrics.json)\n")
    detail_rows = []
    for d in sorted(glob.glob(os.path.join(pdir, "fairmot_*"))):
        tm = _load_json(os.path.join(d, "test_metrics.json"))
        if tm:
            detail_rows.append((tm.get("dataset", os.path.basename(d)), tm))
    if detail_rows:
        A("| dataset | Pr | Re | F1 | MOTA | IDF1 | IDsw | videos | frames | fps |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for ds, tm in detail_rows:
            A(f"| {ds} | {tm.get('Precision',0):.3f} | {tm.get('Recall',0):.3f} | "
              f"{tm.get('F1',0):.3f} | {tm.get('MOTA',0):.3f} | {tm.get('IDF1',0):.3f} | "
              f"{tm.get('ID_switches','?')} | {tm.get('total_videos','?')} | "
              f"{tm.get('total_frames','?')} | {tm.get('fps',0):.1f} |")
    else:
        A("_no per-dataset eval results found._")
    A("")

    # ---- problems ----
    probs = status.get("problems", [])
    A("## Problems / notes\n")
    if probs:
        for p in probs:
            A(f"- {p}")
    else:
        A("- none recorded")
    A("")

    out = os.path.join(pdir, "FAIRMOT_RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
