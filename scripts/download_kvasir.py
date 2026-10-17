"""Kvasir-SEG (1,000 images) from the HF mirror kowndinya23/Kvasir-SEG.

datasets.simula.no fails TLS verification behind our proxy, so we use the mirror.
Output: data/kvasir/{images,masks}/<name>.{jpg,png} and data/kvasir/meta.csv
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from dgb.ignore import build_mask
from dgb.io import resize_pair, save_pair

REPO = "kowndinya23/Kvasir-SEG"
FILES = ["data/train-00000-of-00001-7a33e3aba9046bad.parquet",
         "data/validation-00000-of-00001-4d65c2d2950bdc96.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kvasir")
    a = ap.parse_args()
    out = Path(a.out)
    rows = []
    for f in FILES:
        df = pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset"))
        for _, r in df.iterrows():
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            ann = Image.open(io.BytesIO(r["annotation"]["bytes"])).convert("L")
            img, ann = resize_pair(img, ann)
            rgb = np.asarray(img)
            m = build_mask(rgb, np.asarray(ann) > 127)
            name = Path(r["name"]).stem
            save_pair(out, name, img, m)
            rows.append(dict(name=name, dataset="kvasir", center="kvasir", sequence="",
                             positive=int((m == 1).any()), h=rgb.shape[0], w=rgb.shape[1]))
    pd.DataFrame(rows).to_csv(out / "meta.csv", index=False)
    print(len(rows), "images ->", out)


if __name__ == "__main__":
    main()
