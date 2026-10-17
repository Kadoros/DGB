"""Run a frozen encoder once over every image in data/splits.csv and cache features.

cache/<encoder>_<res>/feats.npy : float16 memmap (N, C, h, w), rows in splits.csv order
cache/masks_<res>.npy           : uint8 (N, res, res), shared by all encoders
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dgb.data import FrameDataset, load_splits
from dgb.encoders import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    df = load_splits()
    enc = build_encoder(a.encoder)
    out = Path("cache") / f"{a.encoder}_{a.res}"
    out.mkdir(parents=True, exist_ok=True)
    h = a.res // enc.stride
    feats = np.lib.format.open_memmap(out / "feats.npy", "w+", np.float16, (len(df), enc.dim, h, h))
    mpath = Path("cache") / f"masks_{a.res}.npy"
    masks = None if mpath.exists() else np.zeros((len(df), a.res, a.res), np.uint8)

    dl = torch.utils.data.DataLoader(FrameDataset(df, a.res, enc.mean, enc.std), batch_size=a.bs,
                                     num_workers=2)
    with torch.inference_mode():
        for x, m, i in tqdm(dl, mininterval=30):
            feats[i.numpy()] = enc(x).numpy().astype(np.float16)
            if masks is not None:
                masks[i.numpy()] = m.numpy().astype(np.uint8)
    feats.flush()
    if masks is not None:
        np.save(mpath, masks)
    json.dump(dict(encoder=a.encoder, res=a.res, stride=enc.stride, dim=enc.dim, n=len(df)),
              open(out / "info.json", "w"))


if __name__ == "__main__":
    main()
