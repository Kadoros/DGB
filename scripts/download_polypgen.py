"""PolypGen (8,037 frames) streamed from the HF mirror EmpyreanCz/PolypGen2021-processed.

The official release is on Synapse (login required). The mirror keeps the original
file names in dataset.json, from which the centre is recovered:
  - positive single frames: prefix C1..C6
  - positive sequences seq16..seq23 are named "_C6_" -> C6 (432 frames, matches the paper)
  - other positive sequences: centre not recoverable (paper: mostly C1/C2/C4/C5) -> "unk"
  - negative sequences seq11..seq13 are named "_C5_" -> C5, others "unk"
C3 has no sequence frames (457 = all single frames), so C3 and C6 are fully identified
and are the external test centres.

The 7.4 GB tar is streamed and resized on the fly; it is never stored on disk.
"""
import argparse
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dgb.ignore import build_mask
from dgb.io import MAX_SIDE

URL = ("https://huggingface.co/datasets/EmpyreanCz/PolypGen2021-processed/"
       "resolve/main/PolypGen2021.tar")


def centre_of(src_name: str, positive: bool):
    if positive and re.match(r"C[1-6]_", src_name):
        return src_name[:2], ""
    seq = "pos_" if positive else "neg_"
    seq += src_name.split("_")[0]
    m = re.search(r"_(C[1-6])_", src_name)
    return (m.group(1) if m else "unk"), seq


def resized(im: Image.Image, nearest: bool):
    w, h = im.size
    s = min(1.0, MAX_SIDE / max(w, h))
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.NEAREST if nearest else Image.BICUBIC)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/polypgen")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "raw_masks").mkdir(parents=True, exist_ok=True)
    meta = None
    proc = subprocess.Popen(["curl", "-sSL", "--retry", "5", URL], stdout=subprocess.PIPE)
    n = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.name == "dataset.json":
                meta = json.load(tf.extractfile(m))
                continue
            if p.suffix != ".png" or p.parent.name not in ("references", "targets"):
                continue
            im = Image.open(io.BytesIO(tf.extractfile(m).read()))
            if p.parent.name == "references":
                resized(im.convert("RGB"), False).save(out / "images" / f"{p.stem}.jpg", quality=95)
            else:
                g = np.asarray(im.convert("RGB"))[..., 1] > 127
                resized(Image.fromarray(g.astype(np.uint8)), True).save(out / "raw_masks" / f"{p.stem}.png")
            n += 1
            if n % 2000 == 0:
                print(n, flush=True)
    assert proc.wait() == 0 and meta is not None

    (out / "masks").mkdir(exist_ok=True)
    rows = []
    for r in meta:
        name = Path(r["input_image"]).stem
        src = r["source"].replace("\\", "/")
        positive_dir = "/positive/" in src
        centre, seq = centre_of(Path(src).name, positive_dir)
        img = np.asarray(Image.open(out / "images" / f"{name}.jpg"))
        les = np.asarray(Image.open(out / "raw_masks" / f"{name}.png")) > 0
        mk = build_mask(img, les)
        Image.fromarray(mk).save(out / "masks" / f"{name}.png")
        (out / "raw_masks" / f"{name}.png").unlink()
        rows.append(dict(name=name, dataset="polypgen", center=centre, sequence=seq,
                         positive=int((mk == 1).any()), source_dir="positive" if positive_dir else "negative",
                         h=img.shape[0], w=img.shape[1]))
    (out / "raw_masks").rmdir()
    df = pd.DataFrame(rows)
    df.to_csv(out / "meta.csv", index=False)
    print(df.groupby(["source_dir", "center"]).size())


if __name__ == "__main__":
    main()
