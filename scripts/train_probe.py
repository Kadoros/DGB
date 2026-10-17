"""Train a head on cached frozen features, fit temperature on val, evaluate test sets.

Loss: BCE + soft Dice on non-ignore pixels at the input resolution.
Model selection: best val micro-Dice (checked every epoch).
Temperature: one scalar T fitted on val pixels (NLL), applied as sigmoid(logit / T).
Test evaluation runs at the cached image resolution (long side <= 640), not at `res`.

Writes runs/<encoder>_<res>_<head>_s<seed>/{metrics.json, fp_components.csv, head.pt}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dgb.data import frame_index, load_mask, load_splits, paths
from dgb.heads import HEADS
from dgb.metrics import Evaluator

EVAL_SPLITS = ("val", "test_int", "test_ext")


def masked_loss(logit, mask):
    valid = (mask != 255).float()
    gt = (mask == 1).float()
    bce = F.binary_cross_entropy_with_logits(logit, gt, weight=valid, reduction="sum") / valid.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    dice = 1 - (2 * (p * gt).sum() + 1) / (p.sum() + gt.sum() + 1)
    return bce + dice


def predict(head, feats, idx, res, bs=64):
    """Logits upsampled to res x res, float32."""
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            f = torch.from_numpy(feats[idx[s:s + bs]].astype(np.float32))
            out.append(F.interpolate(head(f), size=(res, res), mode="bilinear", align_corners=False)[:, 0])
    return torch.cat(out)


def fit_temperature(logits, masks):
    v = masks != 255
    z, y = logits[v], (masks[v] == 1).float()
    if len(z) > 4_000_000:
        sel = torch.randperm(len(z), generator=torch.Generator().manual_seed(0))[:4_000_000]
        z, y = z[sel], y[sel]
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(z / logt.exp(), y)
        l.backward()
        return l
    opt.step(closure)
    return float(logt.exp())


def evaluate(head, feats, df, idx, T, collect_fp):
    ev = {"raw": Evaluator(), "ts": Evaluator()}
    head.eval()
    with torch.no_grad():
        for s in range(0, len(idx), 32):
            b = idx[s:s + 32]
            lg = head(torch.from_numpy(feats[b].astype(np.float32)))
            for j, i in enumerate(b):
                row = df.iloc[i]
                mask = load_mask(row)
                z = F.interpolate(lg[j:j + 1], size=mask.shape, mode="bilinear", align_corners=False)[0, 0]
                rgb = np.asarray(Image.open(paths(row)[0]).convert("RGB")) if collect_fp else None
                kw = dict(name=row["name"], sequence=row["sequence"], frame=frame_index(row["name"]))
                ev["raw"].add(torch.sigmoid(z).numpy(), mask, rgb=rgb, **kw)
                ev["ts"].add(torch.sigmoid(z / T).numpy(), mask, **kw)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--head", default="linear", choices=list(HEADS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    df = load_splits()
    feats = np.load(f"cache/{a.encoder}_{a.res}/feats.npy", mmap_mode="r")
    masks = torch.from_numpy(np.load(f"cache/masks_{a.res}.npy"))
    tr = np.where(df.split == "train")[0]
    va = np.where(df.split == "val")[0]

    head = HEADS[a.head](feats.shape[1])
    prior = float((masks[tr] == 1).float().mean())
    torch.nn.init.constant_(head.fc.bias, np.log(prior / (1 - prior)))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        head.train()
        perm = np.random.permutation(tr)
        tot = 0
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            b = np.sort(perm[s:s + a.bs])
            f = torch.from_numpy(feats[b].astype(np.float32))
            if np.random.rand() < 0.5:  # horizontal flip of features and masks
                f, m = f.flip(-1), masks[b].flip(-1)
            else:
                m = masks[b]
            lg = F.interpolate(head(f), size=m.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            loss = masked_loss(lg, m)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        head.eval()
        lv = predict(head, feats, va, a.res)
        mv = masks[va]
        v = mv != 255
        p, g = (lv >= 0) & v, (mv == 1) & v
        d = float(2 * (p & g).sum() / max(int(p.sum() + g.sum()), 1))
        hist.append(dict(epoch=ep, train_loss=tot / (len(perm) // a.bs), val_micro_dice=d))
        print(f"ep {ep} loss {hist[-1]['train_loss']:.4f} val dice {d:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if d > best:
            best, best_state = d, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()

    T = fit_temperature(predict(head, feats, va, a.res), masks[va])
    print("temperature", T)

    out = Path("runs") / f"{a.encoder}_{a.res}_{a.head}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(args=vars(a), temperature=T, history=hist, best_val_micro_dice=best)
    fp_rows = []
    for sp in EVAL_SPLITS:
        idx = np.where(df.split == sp)[0]
        ev = evaluate(head, feats, df, idx, T, collect_fp=sp == "test_ext")
        res[sp] = {k: e.summary() for k, e in ev.items()}
        fp_rows += [dict(split=sp, **r) for r in ev["raw"].fp_rows]
        print(sp, json.dumps({k: round(v, 4) for k, v in res[sp]["raw"].items() if isinstance(v, float)}))
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    pd.DataFrame(fp_rows).to_csv(out / "fp_components.csv", index=False)
    torch.save(head.state_dict(), out / "head.pt")


if __name__ == "__main__":
    main()
