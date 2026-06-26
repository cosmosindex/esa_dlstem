"""
SMOKE TEST (Path B feasibility): does a single GT-box exemplar on frame 0 change
/ rescue SAM3's find-grounding detections vs text-only -- on the small species
where text-only fails? In-video box = oracle (test GT), diagnostic only.
SAM3 visual prompt accepts exactly ONE box per concept.

    CUDA_VISIBLE_DEVICES=0 python evaluation/_sam3_exemplar_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.birdsai_mot import BIRDSAIMOTDataset
from models.sam3 import SAM3TextTracker

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
PROMPT = {0: "person", 1: "elephant", 2: "giraffe", 3: "lion", 4: "animal"}
CLIP = 24
# (video_id, class_id) cases: lion, giraffe, unknown, + one elephant control
CASES = [("0000000012_0000000000", 3), ("0000000065_0000000000", 2),
         ("0000000060_0000000000", 4), ("0000000055_0000000000", 1)]


def run_prompt(pred, state, text, box_xywh, nframes):
    pred.reset_state(state)
    kw = dict(inference_state=state, frame_idx=0, text_str=text)
    if box_xywh is not None:
        kw["boxes_xywh"] = torch.tensor(box_xywh, dtype=torch.float32).view(1, 4)
        kw["box_labels"] = torch.ones(1, dtype=torch.long)
    pred.add_prompt(**kw)
    counts = np.zeros(nframes, int); scoresum = np.zeros(nframes)
    for fidx, out in pred.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=nframes, reverse=False):
        m = out["out_binary_masks"]; p = out["out_probs"]
        for i in range(len(p)):
            if m[i].any():
                counts[int(fidx)] += 1; scoresum[int(fidx)] += float(p[i])
    return counts, scoresum


def main():
    ds = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine",
                           class_map={v: k for k, v in CANON.items()})
    vmap = {v.video_id: v for v in ds.videos}
    trk = SAM3TextTracker(class_names=list(PROMPT.values()),
                          label_to_id={v: k for k, v in PROMPT.items()})
    pred = trk.predictor

    for vid, c in CASES:
        video = vmap[vid]
        clip_fids = video.frame_ids[:CLIP]
        frames = [ds._load_frame(video, fid) for fid in clip_fids]
        H, W = frames[0].shape[:2]
        ann0 = ds._load_annotations(video, clip_fids[0])
        bxs = [b for b, l in zip(ann0["boxes"], ann0["labels"]) if int(l) == c]
        if not bxs:
            print(f"\n[{CANON[c]}] {vid}: no GT in frame0, skip"); continue
        b = np.asarray(bxs[0], np.float32)
        xywh = np.array([b[0] / W, b[1] / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H])
        xywh = np.clip(xywh, 0, 1)
        diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))

        trk.init_video(frames); state = trk._inference_state
        text = PROMPT[c]
        ct_t, ss_t = run_prompt(pred, state, text, None, len(frames))
        ct_e, ss_e = run_prompt(pred, state, text, xywh, len(frames))
        ngt = sum(1 for l in ann0["labels"] if int(l) == c)
        print(f"\n[{CANON[c]}] {vid}  prompt='{text}'  exemplar diag={diag:.0f}px  "
              f"(frame0 GT count={ngt})")
        print(f"  text-only : dets={ct_t.sum():4d}  fr_w_det={(ct_t>0).sum():2d}/{len(frames)}  "
              f"mean_sc={ss_t.sum()/max(ct_t.sum(),1):.3f}")
        print(f"  text+exemp: dets={ct_e.sum():4d}  fr_w_det={(ct_e>0).sum():2d}/{len(frames)}  "
              f"mean_sc={ss_e.sum()/max(ct_e.sum(),1):.3f}")
        trk.reset_state()


if __name__ == "__main__":
    main()
