"""Evaluation of per-pixel P(lesion) maps against masks (0 bg, 1 lesion, 255 ignore).

Per positive image : Dice, IoU (threshold 0.5), then averaged over images.
Micro (all pixels) : Dice, pixel recall, pixel precision.
Lesion level       : GT connected components; a lesion is detected when >= 50 % of its
                     pixels are predicted lesion. Recall = detected / all lesions.
Negative images    : frame false-positive rate = share of lesion-free frames with any
                     predicted component >= MIN_FP_AREA of the image.
ECE                : 15 equal-width bins over all valid pixels (both classes).
FP components      : logged with specular / dark fraction so FP location can be reviewed.
Flicker            : on sequences, consecutive frames sorted by frame index:
                     mean |p_t - p_{t-1}| and 1 - IoU(mask_t, mask_{t-1}) on a 112x112 grid,
                     reported next to the same quantity for the GT masks (GT also moves).
"""
from collections import defaultdict

import numpy as np
from scipy import ndimage

N_BINS = 15
MIN_FP_AREA = 0.001
LESION_HIT = 0.5


def specular(rgb):
    mx, mn = rgb.max(2).astype(float), rgb.min(2).astype(float)
    sat = (mx - mn) / np.maximum(mx, 1)
    return (mx > 230) & (sat < 0.15)


class Evaluator:
    def __init__(self):
        self.img = defaultdict(list)
        self.tp = self.fp = self.fn = 0
        self.les_hit = self.les_n = 0
        self.neg_n = self.neg_fp = 0
        self.ece_cnt = np.zeros(N_BINS)
        self.ece_conf = np.zeros(N_BINS)
        self.ece_pos = np.zeros(N_BINS)
        self.fp_rows = []
        self.seq = defaultdict(list)  # sequence -> [(frame_idx, p_small, gt_small)]

    def add(self, prob, mask, name="", rgb=None, sequence="", frame=0):
        valid = mask != 255
        gt = mask == 1
        pred = prob >= 0.5
        p, g, v = pred & valid, gt & valid, valid

        tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
        self.tp += tp; self.fp += fp; self.fn += fn

        if g.any():
            self.img["dice"].append(2 * tp / max(2 * tp + fp + fn, 1))
            self.img["iou"].append(tp / max(tp + fp + fn, 1))
            lab, n = ndimage.label(g)
            if n:
                cover = ndimage.mean(p, lab, index=np.arange(1, n + 1))
                self.les_hit += int((np.asarray(cover) >= LESION_HIT).sum())
                self.les_n += n
        else:
            self.neg_n += 1
            lab, n = ndimage.label(p)
            areas = ndimage.sum(np.ones_like(p), lab, index=np.arange(1, n + 1)) if n else []
            self.neg_fp += int(any(a >= MIN_FP_AREA * v.sum() for a in areas))

        q = prob[v]
        b = np.minimum((q * N_BINS).astype(int), N_BINS - 1)
        self.ece_cnt += np.bincount(b, minlength=N_BINS)
        self.ece_conf += np.bincount(b, weights=q, minlength=N_BINS)
        self.ece_pos += np.bincount(b, weights=g[v].astype(float), minlength=N_BINS)

        if rgb is not None:
            fpm = p & ~g
            lab, n = ndimage.label(fpm)
            if n:
                spec, dark = specular(rgb), rgb.max(2) < 40
                for k, sl in enumerate(ndimage.find_objects(lab), 1):
                    c = lab[sl] == k
                    area = int(c.sum())
                    if area < MIN_FP_AREA * v.sum():
                        continue
                    self.fp_rows.append(dict(
                        name=name, y0=sl[0].start, y1=sl[0].stop, x0=sl[1].start, x1=sl[1].stop,
                        area_frac=area / v.sum(), mean_prob=float(prob[sl][c].mean()),
                        specular_frac=float(spec[sl][c].mean()), dark_frac=float(dark[sl][c].mean()),
                        touches_gt=bool(ndimage.binary_dilation(c, iterations=3).__and__(g[sl]).any())))

        if sequence:
            ps = _small(prob); gs = _small(gt.astype(float))
            vs = _small(valid.astype(float)) > 0.5
            self.seq[sequence].append((frame, ps, gs, vs))

    def summary(self):
        s = {}
        for k, v in self.img.items():
            s[k] = float(np.mean(v)) if v else float("nan")
        s["n_pos_img"] = len(self.img["dice"])
        s["micro_dice"] = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        s["pixel_recall"] = self.tp / max(self.tp + self.fn, 1)
        s["pixel_precision"] = self.tp / max(self.tp + self.fp, 1)
        s["lesion_recall"] = self.les_hit / max(self.les_n, 1)
        s["n_lesions"] = self.les_n
        s["neg_frame_fp_rate"] = self.neg_fp / self.neg_n if self.neg_n else float("nan")
        s["n_neg_img"] = self.neg_n
        tot = self.ece_cnt.sum()
        nz = self.ece_cnt > 0
        s["ece"] = float((np.abs(self.ece_conf[nz] - self.ece_pos[nz])).sum() / tot)
        s.update(self.flicker())
        if self.fp_rows:
            fr = np.array([[r["specular_frac"] > 0.3, r["dark_frac"] > 0.5, r["touches_gt"]]
                           for r in self.fp_rows])
            s["fp_comp_n"] = len(self.fp_rows)
            s["fp_comp_specular"] = float(fr[:, 0].mean())
            s["fp_comp_dark"] = float(fr[:, 1].mean())
            s["fp_comp_near_gt"] = float(fr[:, 2].mean())
        return s

    def flicker(self):
        dp, du, gu = [], [], []
        for frames in self.seq.values():
            frames.sort(key=lambda t: t[0])
            for (_, p0, g0, v0), (_, p1, g1, v1) in zip(frames, frames[1:]):
                v = v0 & v1
                if not v.any():
                    continue
                dp.append(np.abs(p1 - p0)[v].mean())
                du.append(1 - _iou(p0 >= 0.5, p1 >= 0.5, v))
                gu.append(1 - _iou(g0 >= 0.5, g1 >= 0.5, v))
        if not dp:
            return {}
        return {"flicker_prob": float(np.mean(dp)), "flicker_mask": float(np.mean(du)),
                "flicker_mask_gt": float(np.mean(gu)), "n_frame_pairs": len(dp)}


def _small(a, n=112):
    from PIL import Image
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((n, n), Image.BILINEAR))


def _iou(a, b, v):
    a, b = a & v, b & v
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u
