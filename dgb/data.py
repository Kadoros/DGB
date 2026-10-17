"""Dataset over data/splits.csv. Masks: 0 background, 1 lesion, 255 ignore."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path("data")


def load_splits(path=ROOT / "splits.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].fillna("")
    return df


def frame_index(name: str) -> int:
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 0


def paths(row):
    d = ROOT / row["dataset"]
    return d / "images" / f"{row['name']}.jpg", d / "masks" / f"{row['name']}.png"


def load_mask(row) -> np.ndarray:
    return np.asarray(Image.open(paths(row)[1]))


class FrameDataset(torch.utils.data.Dataset):
    """Squash-resizes to res x res; returns (normalised image, mask at res, index)."""

    def __init__(self, df: pd.DataFrame, res: int, mean, std):
        self.df = df.reset_index(drop=True)
        self.res = res
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ip, mp = paths(row)
        img = Image.open(ip).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        msk = Image.open(mp).resize((self.res, self.res), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        x = (x - self.mean) / self.std
        return x, torch.from_numpy(np.asarray(msk).copy()).long(), i
