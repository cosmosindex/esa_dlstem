#!/usr/bin/env python
"""Unattended FairMOT UNION pipeline: train -> eval x3 -> HOTA -> summary.

Designed to run in tmux overnight. Records EVERY stage to
``<pipe_dir>/pipeline_status.json`` and (re)writes ``FAIRMOT_RESULTS.md`` after
each stage, so the results AND any problems are visible the next morning even
if a later stage fails or the box reboots.

Resilient by design: a failing stage is recorded in ``problems`` and the
pipeline still attempts the later stages with whatever artifacts exist (e.g.
HOTA over however many eval splits succeeded).

Stages
  1. train   FairMOT/src/train_union.py (dual-GPU, native-res buckets, W&B,
             model_best.pth by val detection loss)
  2. eval    eval_fairmot.py per test split (rscardata/satmtb/sdmcar) using
             model_best.pth -> mot_format/ + test_metrics.json (no viz)
  3. hota    compute_hota.py over the three eval run dirs -> hota_summary.csv
  4. summary tools/fairmot_write_summary.py -> FAIRMOT_RESULTS.md

Run:  python tools/fairmot_pipeline.py --epochs 30 --gpus 0,1
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAIRMOT_SRC = REPO / "FairMOT" / "src"
UNION_CKPT_DIR = FAIRMOT_SRC / "exp" / "mot" / "fairmot_hrnet18_union"
DATASETS = ["rscardata", "satmtb", "sdmcar"]


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


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd, log_path, cwd, env_extra=None):
    """Run cmd, tee combined output to log_path, return returncode."""
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


def write_summary(pipe_dir):
    try:
        subprocess.run([sys.executable, str(REPO / "tools" / "fairmot_write_summary.py"),
                        str(pipe_dir)], cwd=str(REPO), check=False)
    except Exception as exc:
        print(f"[summary] failed: {exc!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    # single GPU: FairMOT's multi-GPU DataParallel hangs on this box (the first
    # batch never completes, GPU stuck ~900MB). Single-GPU training is verified
    # working. bs=4 fits 1920x1088 (the big bucket) on one 32GB card.
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--master_batch_size", type=int, default=4)
    ap.add_argument("--lr_step", default="20")
    ap.add_argument("--pipe_dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.environ.get("EXPERIMENT_ROOT", "/work/anon/experiments")
    pipe_dir = Path(args.pipe_dir or f"{exp_root}/fairmot_union_{ts}")
    pipe_dir.mkdir(parents=True, exist_ok=True)
    st = Status(pipe_dir / "pipeline_status.json")
    print(f"[pipeline] dir = {pipe_dir}")
    st.flush()
    write_summary(pipe_dir)

    # ---------- stage 1: train ----------
    train_log = pipe_dir / "train.log"
    cfg = FAIRMOT_SRC / "lib" / "cfg" / "union_car.json"
    train_cmd = [
        sys.executable, "train_union.py", "mot",
        "--exp_id", "fairmot_hrnet18_union", "--arch", "hrnet_18",
        "--data_cfg", str(cfg), "--input_w", "1024", "--input_h", "1024",
        "--aug_scale_min", "0.8", "--gpus", args.gpus,
        "--batch_size", str(args.batch_size),
        "--master_batch_size", str(args.master_batch_size),
        "--num_epochs", str(args.epochs), "--lr", "1e-4",
        # num_workers=0: this box deadlocks DataLoader workers (fork+CUDA, and
        # spawn hangs too with the bucket sampler). Synchronous load is slower
        # but rock-solid for an unattended overnight run. (CLAUDE.md guidance.)
        "--lr_step", args.lr_step, "--num_workers", "0", "--print_iter", "50",
    ]
    st.stage("train", "running", f"epochs={args.epochs} gpus={args.gpus}")
    write_summary(pipe_dir)
    t0 = time.time()
    rc = _run(train_cmd, train_log, FAIRMOT_SRC)
    dt = (time.time() - t0) / 3600.0
    best_ckpt = UNION_CKPT_DIR / "model_best.pth"
    # parse best epoch / val det loss from the train log
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
        "wandb": "esa-dlstem / fairmot_hrnet18_union",
    }
    if rc != 0:
        st.problem(f"training exited rc={rc} after {dt:.2f}h — see {train_log}")
    if not best_ckpt.is_file():
        st.problem(f"no model_best.pth at {best_ckpt}; falling back to model_last.pth")
        last = UNION_CKPT_DIR / "model_last.pth"
        best_ckpt = last if last.is_file() else best_ckpt
    st.stage("train", "done" if best_ckpt.is_file() else "failed",
             f"{dt:.2f}h rc={rc} best_epoch={best_epoch}")
    st.flush(); write_summary(pipe_dir)

    if not best_ckpt.is_file():
        st.problem("no usable checkpoint — cannot eval. Stopping.")
        st.d["overall"] = "failed (no checkpoint)"
        st.flush(); write_summary(pipe_dir)
        return

    # ---------- stage 2: eval per split ----------
    eval_env = {"EXPERIMENT_ROOT": str(pipe_dir)}
    ok_splits = []
    for ds in DATASETS:
        cfg_yaml = REPO / "configs" / "MOT" / f"fairmot_{ds}.yaml"
        elog = pipe_dir / f"eval_{ds}.log"
        st.stage(f"eval:{ds}", "running")
        rc = _run([sys.executable, "eval_fairmot.py", "--config", str(cfg_yaml),
                   "--dataset", ds, "--checkpoint", str(best_ckpt)],
                  elog, REPO, env_extra=eval_env)
        if rc == 0:
            ok_splits.append(ds)
            st.stage(f"eval:{ds}", "done")
        else:
            st.problem(f"eval {ds} exited rc={rc} — see {elog}")
            st.stage(f"eval:{ds}", "failed", f"rc={rc}")
        write_summary(pipe_dir)

    if not ok_splits:
        st.problem("all eval splits failed — no mot_format to score. Stopping.")
        st.d["overall"] = "failed (eval)"
        st.flush(); write_summary(pipe_dir)
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

    # ---------- done ----------
    st.d["overall"] = "complete" if (pipe_dir / "hota_summary.csv").is_file() \
        else "complete-with-problems"
    st.d["finished"] = _now()
    st.flush()
    write_summary(pipe_dir)
    print(f"[pipeline] DONE -> {pipe_dir}/FAIRMOT_RESULTS.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # last-resort: record and still write summary
        import traceback
        tb = traceback.format_exc()
        print(tb)
        # best-effort status note
        try:
            pd = os.environ.get("FAIRMOT_PIPE_DIR")
            if pd:
                with open(Path(pd) / "pipeline_status.json", "r+") as f:
                    d = json.load(f)
                    d.setdefault("problems", []).append(f"FATAL: {exc!r}")
                    d["overall"] = "crashed"
                    f.seek(0); json.dump(d, f, indent=2); f.truncate()
        except Exception:
            pass
        raise
