#!/usr/bin/env python
"""Unattended TGraM UNION pipeline: train -> eval x3 -> HOTA -> summary.

The TGraM sibling of tools/fairmot_pipeline.py. Designed to run in tmux. Records
every stage to ``<pipe_dir>/pipeline_status.json`` and (re)writes
``TGRAM_RESULTS.md`` after each stage, so results AND problems are visible even
if a later stage fails. Resilient: a failing stage is recorded and the pipeline
still attempts later stages with whatever artifacts exist.

Stages
  1. train   TGraM/src/tgram_train_union.py (single-GPU, native-res buckets,
             num_frames=3, W&B, model_best.pth by val detection loss)
  2. eval    eval_tgram.py per test split (rscardata/satmtb/sdmcar) using
             model_best.pth -> mot_format/ + test_metrics.json (no viz)
  3. hota    compute_hota.py over the three eval run dirs -> hota_summary.csv
  4. summary inline -> TGRAM_RESULTS.md

Run:  python tools/tgram_pipeline.py --epochs 30 --gpus 0
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TGRAM_SRC = REPO / "TGraM" / "src"
# NB: TGraM saves to TGraM/exp/mot/<exp> (opts root_dir = src/lib/../.. = TGraM),
# NOT TGraM/src/exp — avoiding the src/exp path bug fairmot_pipeline.py hit.
EXP_ID = "tgram_mbseg_union"
UNION_CKPT_DIR = REPO / "TGraM" / "exp" / "mot" / EXP_ID
DATASETS = ["rscardata", "satmtb", "sdmcar"]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Status:
    def __init__(self, path):
        self.path = path
        self.d = {"overall": "running", "started": _now(), "stages": [],
                  "problems": [], "train": {}}

    def stage(self, name, status, note=""):
        self.d["stages"].append({"name": name, "status": status,
                                 "note": note, "t": _now()})
        self.flush()

    def problem(self, msg):
        self.d["problems"].append(f"[{_now()}] {msg}")
        self.flush()

    def flush(self):
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=2)


def _run(cmd, log_path, cwd, env_extra=None):
    env = dict(os.environ)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if env_extra:
        env.update(env_extra)
    with open(log_path, "a") as log:
        log.write(f"\n===== {_now()} :: {' '.join(map(str, cmd))} (cwd={cwd}) =====\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        return p.wait()


def write_summary(pipe_dir, st):
    """Render TGRAM_RESULTS.md from the live status + hota_summary.csv (if any)."""
    lines = [f"# TGraM UNION pipeline — {st.d.get('overall', '?')}",
             "", f"- started: {st.d.get('started')}",
             f"- finished: {st.d.get('finished', '(running)')}",
             f"- pipe dir: `{pipe_dir}`",
             f"- ckpt dir: `{UNION_CKPT_DIR}`",
             f"- W&B: esa-dlstem / {EXP_ID}", ""]
    tr = st.d.get("train", {})
    if tr:
        lines += ["## Train",
                  f"- rc={tr.get('rc')} hours={tr.get('hours')} "
                  f"best_epoch={tr.get('best_epoch')} "
                  f"best_val_det={tr.get('best_val_det')}",
                  f"- log: `{tr.get('log')}`", ""]
    csv_path = pipe_dir / "hota_summary.csv"
    if csv_path.is_file():
        lines += ["## HOTA", "", "| dataset | tracker | HOTA | DetA | AssA | "
                  "MOTA | IDF1 | IDsw |", "|---|---|---|---|---|---|---|---|"]
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                lines.append(
                    f"| {r.get('dataset','')} | {r.get('tracker','')} | "
                    f"{r.get('HOTA','')} | {r.get('DetA','')} | {r.get('AssA','')} | "
                    f"{r.get('MOTA','')} | {r.get('IDF1','')} | {r.get('IDsw','')} |")
        lines.append("")
    if st.d.get("problems"):
        lines += ["## Problems", ""] + [f"- {p}" for p in st.d["problems"]] + [""]
    stage_rows = [f"- {s['t']}  {s['name']}: {s['status']} {s.get('note','')}"
                  for s in st.d.get("stages", [])]
    lines += ["## Stage log", ""] + stage_rows + [""]
    (pipe_dir / "TGRAM_RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpus", default="0")        # single GPU (box hangs on DataParallel)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--master_batch_size", type=int, default=4)
    ap.add_argument("--lr_step", default="20")
    ap.add_argument("--pipe_dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")
    pipe_dir = Path(args.pipe_dir or f"{exp_root}/tgram_union_{ts}")
    pipe_dir.mkdir(parents=True, exist_ok=True)
    st = Status(pipe_dir / "pipeline_status.json")
    print(f"[pipeline] dir = {pipe_dir}")
    st.flush(); write_summary(pipe_dir, st)

    # ---------- stage 1: train ----------
    train_log = pipe_dir / "train.log"
    cfg = TGRAM_SRC / "lib" / "cfg" / "union_car.json"
    train_cmd = [
        sys.executable, "tgram_train_union.py",
        "--exp_id", EXP_ID, "--arch", "tgrammbseg",
        "--dataloader", "tgram", "--load_model", "",
        "--data_cfg", str(cfg), "--input_w", "1024", "--input_h", "1024",
        "--num_frames", "3", "--aug_scale_min", "0.8", "--gpus", args.gpus,
        "--batch_size", str(args.batch_size),
        "--master_batch_size", str(args.master_batch_size),
        "--num_epochs", str(args.epochs), "--lr", "1e-4",
        "--lr_step", args.lr_step, "--num_workers", "0", "--print_iter", "50",
    ]
    st.stage("train", "running", f"epochs={args.epochs} gpus={args.gpus}")
    write_summary(pipe_dir, st)
    t0 = time.time()
    rc = _run(train_cmd, train_log, TGRAM_SRC)
    dt = (time.time() - t0) / 3600.0
    best_ckpt = UNION_CKPT_DIR / "model_best.pth"
    best_epoch = best_val = None
    try:
        m = re.findall(r"best epoch=(\S+)\s+val_det_loss=(\S+)", train_log.read_text())
        if m:
            best_epoch, best_val = m[-1]
    except Exception:
        pass
    st.d["train"] = {
        "rc": rc, "hours": round(dt, 2),
        "best_epoch": best_epoch, "best_val_det": best_val,
        "best_ckpt": str(best_ckpt), "log": str(train_log),
        "wandb": f"esa-dlstem / {EXP_ID}",
    }
    if rc != 0:
        st.problem(f"training exited rc={rc} after {dt:.2f}h — see {train_log}")
    if not best_ckpt.is_file():
        st.problem(f"no model_best.pth at {best_ckpt}; falling back to model_last.pth")
        last = UNION_CKPT_DIR / "model_last.pth"
        best_ckpt = last if last.is_file() else best_ckpt
    st.stage("train", "done" if best_ckpt.is_file() else "failed",
             f"{dt:.2f}h rc={rc} best_epoch={best_epoch}")
    st.flush(); write_summary(pipe_dir, st)

    if not best_ckpt.is_file():
        st.problem("no usable checkpoint — cannot eval. Stopping.")
        st.d["overall"] = "failed (no checkpoint)"
        st.flush(); write_summary(pipe_dir, st)
        return

    # ---------- stage 2: eval per split ----------
    eval_env = {"EXPERIMENT_ROOT": str(pipe_dir)}
    ok_splits = []
    for ds in DATASETS:
        cfg_yaml = REPO / "configs" / "MOT" / f"tgram_{ds}.yaml"
        elog = pipe_dir / f"eval_{ds}.log"
        st.stage(f"eval:{ds}", "running")
        rc = _run([sys.executable, "eval_tgram.py", "--config", str(cfg_yaml),
                   "--dataset", ds, "--checkpoint", str(best_ckpt)],
                  elog, REPO, env_extra=eval_env)
        if rc == 0:
            ok_splits.append(ds)
            st.stage(f"eval:{ds}", "done")
        else:
            st.problem(f"eval {ds} exited rc={rc} — see {elog}")
            st.stage(f"eval:{ds}", "failed", f"rc={rc}")
        write_summary(pipe_dir, st)

    if not ok_splits:
        st.problem("all eval splits failed — no mot_format to score. Stopping.")
        st.d["overall"] = "failed (eval)"
        st.flush(); write_summary(pipe_dir, st)
        return

    # ---------- stage 3: HOTA ----------
    hlog = pipe_dir / "hota.log"
    st.stage("hota", "running", f"splits={ok_splits}")
    rc = _run([sys.executable, "compute_hota.py",
               "--tracker-output-root", str(pipe_dir),
               "--workspace", str(pipe_dir / "hota_ws"),
               "--output", str(pipe_dir / "hota_summary.csv")],
              hlog, REPO)
    if rc == 0 and (pipe_dir / "hota_summary.csv").is_file():
        st.stage("hota", "done")
    else:
        st.problem(f"compute_hota exited rc={rc} — see {hlog}")
        st.stage("hota", "failed", f"rc={rc}")

    st.d["overall"] = "complete" if (pipe_dir / "hota_summary.csv").is_file() \
        else "complete-with-problems"
    st.d["finished"] = _now()
    st.flush(); write_summary(pipe_dir, st)
    print(f"[pipeline] DONE -> {pipe_dir}/TGRAM_RESULTS.md")


if __name__ == "__main__":
    main()
