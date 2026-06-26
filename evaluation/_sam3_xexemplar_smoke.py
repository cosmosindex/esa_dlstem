"""FEASIBILITY PROBE — cross-image exemplar detection with SAM3 (v2).

Question the user raised: can we use annotations from the *training* set as
visual exemplars to make SAM3 detect that concept on a *test* image?  (a FAIR
few-shot detector, not the test-GT oracle.)

v1 proved a single train exemplar already recovers tiny thermal species that
text + all 3 trained detectors score ~0 on. v2 validates the two top improvement
levers, both motivated by SAM3's scoring head (`DotProductScoring` mean-pools ALL
prompt tokens into one "concept prototype" — model_misc.py:743):
    (1) K positive exemplars  -> richer, less single-instance-overfit prototype
    (2) M negative exemplars  -> push the prototype away from thermal background
                                 (the FP-flood cause); uses label=False tokens

Mechanism recap — the geometry encoder RoI-aligns the exemplar box over whatever
`img_feats` it is given and returns prompt token(s); the encoder/decoder ground
those tokens on a (possibly different) image. So we pool each exemplar token from
ITS OWN training frame, concat the tokens, then ground on the test frame. Output
box coordinates come from the decoder's own anchors over the test image — the
reference box location does NOT leak into output coords, only into the prototype.

    CUDA_VISIBLE_DEVICES=0 python evaluation/_sam3_xexemplar_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.birdsai_mot import BIRDSAIMOTDataset

BIRDSAI_ROOT = "/data/ESA_DLSTEM_2025/data/wild_animal/BIRDSAI"
CANON = {0: "human", 1: "elephant", 2: "giraffe", 3: "lion", 4: "unknown"}
CANON_MAP = {v: k for k, v in CANON.items()}
THR_BASE = 0.2         # detector runs at this floor; we also post-filter @0.4
THRS = (0.2, 0.4)
IOU_HIT = 0.3
N_TEST_FRAMES = 12
K_POS = 4              # positive exemplars per class
M_NEG = 2              # negative (background) exemplars  (M<K: equal-weight pool)
NMS_IOU = 0.5          # per-image class-agnostic NMS on the exemplar dets
SEED = 0


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _norm_cxcywh(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W; h = (y2 - y1) / H
    return np.clip([cx, cy, w, h], 0, 1)


def main():
    from models.sam3 import SAM3Tracker
    SAM3Tracker._ensure_pkg_resources_shim()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    rng = np.random.default_rng(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(load_from_HF=True)
    proc = Sam3Processor(model, confidence_threshold=THR_BASE)
    dev = proc.device

    ds_tr = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="train", granularity="fine", class_map=CANON_MAP)
    ds_te = BIRDSAIMOTDataset(root=BIRDSAI_ROOT, split="test", granularity="fine", class_map=CANON_MAP)

    # ---- gather K diverse positive training boxes for a class ---------------
    def gather_pos(cls, k):
        cands = []  # (diag, video, fid, box)
        for v in ds_tr.videos:
            for fid in v.frame_ids[::12]:
                ann = ds_tr._load_annotations(v, fid)
                for b, l in zip(ann["boxes"], ann["labels"]):
                    if int(l) != cls:
                        continue
                    b = np.asarray(b, np.float32)
                    cands.append((float(np.hypot(b[2] - b[0], b[3] - b[1])), v, fid, b))
            if len(cands) > 400:
                break
        if not cands:
            return []
        cands.sort(key=lambda t: -t[0])
        # take the largest, but spread across the size range for diversity
        idx = np.linspace(0, min(len(cands), 120) - 1, k).astype(int)
        return [cands[i] for i in idx]

    # ---- sample M background (negative) boxes from training frames ----------
    def gather_neg(cls, m):
        out = []
        vids = list(ds_tr.videos)
        tries = 0
        while len(out) < m and tries < 2000:
            tries += 1
            v = vids[rng.integers(len(vids))]
            fid = v.frame_ids[rng.integers(len(v.frame_ids))]
            ann = ds_tr._load_annotations(v, fid)
            gt = np.asarray(ann["boxes"], np.float32).reshape(-1, 4) if len(ann["boxes"]) else np.zeros((0, 4))
            H, W = ds_tr._load_frame(v, fid).shape[:2]
            s = int(rng.integers(14, 30))
            x = int(rng.integers(0, max(1, W - s))); y = int(rng.integers(0, max(1, H - s)))
            box = np.array([x, y, x + s, y + s], np.float32)
            if len(gt) and (_iou(box[None], gt) > 0.05).any():
                continue
            out.append((float(s * 1.41), v, fid, box))
        return out

    @torch.inference_mode()
    def token_for(ref_pil, box_xyxy, positive):
        W, H = ref_pil.size
        st = proc.set_image(ref_pil)
        proc.reset_all_prompts(st)
        geo = model._get_dummy_prompt()
        cxcywh = _norm_cxcywh(box_xyxy, W, H)
        boxes = torch.tensor(cxcywh, dtype=torch.float32, device=dev).view(1, 1, 4)
        labels = torch.tensor([bool(positive)], device=dev).view(1, 1)
        geo.append_boxes(boxes, labels)
        _, img_feats, img_pos, sizes = model._get_img_feats(st["backbone_out"], proc.find_stage.img_ids)
        feats, masks = model.geometry_encoder(geo, img_feats, sizes, img_pos)
        # forward output order is [box_token, cls_token]; keep the box token only
        # (drop the per-frame CLS so K frames don't over-weight the generic CLS)
        return feats[:1], masks[:, :1]

    def build_bank(exemplars):
        """exemplars: list of (ref_pil, box, positive). Returns concatenated
        (geo_feats [S,1,C], geo_masks [1,S])."""
        feats, masks = [], []
        for ref_pil, box, pos in exemplars:
            f, m = token_for(ref_pil, box, pos)
            feats.append(f); masks.append(m)
        return torch.cat(feats, dim=0), torch.cat(masks, dim=1)

    @torch.inference_mode()
    def detect(query_pil, mode, bank=None, text="visual"):
        st = proc.set_image(query_pil)
        proc.reset_all_prompts(st)
        st = proc.set_text_prompt(text, st)
        if mode == "text":
            return _nms(st["boxes"].float(), st["scores"].float())
        geo_feats, geo_masks = bank
        orig = model.geometry_encoder.forward
        model.geometry_encoder.forward = lambda *a, **k: (geo_feats, geo_masks)
        try:
            st["geometric_prompt"] = model._get_dummy_prompt()
            st = proc._forward_grounding(st)
        finally:
            model.geometry_encoder.forward = orig
        return _nms(st["boxes"].float(), st["scores"].float())

    import torchvision

    def _nms(boxes, scores):
        if len(boxes) == 0:
            return boxes.cpu().numpy(), scores.cpu().numpy()
        keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
        return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()

    def evaluate(conditions, tests):
        """conditions: dict name -> (mode, bank, text). Returns per-cond, per-thr stats."""
        stat = {n: {t: [0, 0, 0] for t in THRS} for n in conditions}  # [dets, prec_num, rec_num]
        ngt = 0
        for v, fid, gt in tests:
            q = Image.fromarray(ds_te._load_frame(v, fid))
            ngt += len(gt)
            for name, (mode, bank, text) in conditions.items():
                bx, sc = detect(q, mode, bank, text)
                for t in THRS:
                    keep = sc >= t
                    b = bx[keep]
                    stat[name][t][0] += len(b)
                    if len(b):
                        iou = _iou(b, gt)
                        stat[name][t][1] += int((iou >= IOU_HIT).any(1).sum())  # dets hitting
                        stat[name][t][2] += int((iou >= IOU_HIT).any(0).sum())  # GT matched
        return stat, ngt

    for cls in [1, 2, 3, 4]:   # elephant(control), giraffe, lion, unknown
        pos = gather_pos(cls, K_POS)
        neg = gather_neg(cls, M_NEG)
        if not pos:
            print(f"\n[{CANON[cls]}] no train pos, skip"); continue
        pos_ex = [(Image.fromarray(ds_tr._load_frame(v, f)), b, True) for _, v, f, b in pos]
        neg_ex = [(Image.fromarray(ds_tr._load_frame(v, f)), b, False) for _, v, f, b in neg]

        bank_1 = build_bank(pos_ex[:1])
        bank_k = build_bank(pos_ex)
        bank_kn = build_bank(pos_ex + neg_ex)

        tests = []
        for v in ds_te.videos:
            for fid in v.frame_ids[::15]:
                ann = ds_te._load_annotations(v, fid)
                g = [np.asarray(b, np.float32) for b, l in zip(ann["boxes"], ann["labels"]) if int(l) == cls]
                if g:
                    tests.append((v, fid, np.stack(g)))
                    if len(tests) >= N_TEST_FRAMES:
                        break
            if len(tests) >= N_TEST_FRAMES:
                break

        conditions = {
            "text     ": ("text", None, CANON[cls]),
            "1pos     ": ("x", bank_1, "visual"),
            f"{K_POS}pos     ": ("x", bank_k, "visual"),
            f"{K_POS}pos+{M_NEG}neg": ("x", bank_kn, "visual"),
        }
        stat, ngt = evaluate(conditions, tests)
        ex_sizes = ",".join(f"{d:.0f}" for d, *_ in pos)
        print(f"\n=== {CANON[cls]} | {len(tests)} test frames, {ngt} GT | "
              f"pos exemplar diags=[{ex_sizes}]px ===")
        print(f"  {'cond':<12} | " + " | ".join(
            f"@{t}: dets / recall / prec" for t in THRS))
        for name in conditions:
            row = f"  {name:<12} |"
            for t in THRS:
                d, pn, rn = stat[name][t]
                rec = rn / max(ngt, 1); prec = pn / max(d, 1)
                row += f"  {d:4d} / {rec:.2f} / {prec:.2f} |"
            print(row)


if __name__ == "__main__":
    main()
