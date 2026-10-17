"""Shared image/mask I/O for the cached datasets."""
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE = 640  # cached long side; models resize again to 224/448/518


def resize_pair(img: Image.Image, mask: Image.Image, max_side: int = MAX_SIDE):
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        size = (round(w * s), round(h * s))
        img = img.resize(size, Image.BICUBIC)
        mask = mask.resize(size, Image.NEAREST)
    return img, mask


def save_pair(root: Path, name: str, img: Image.Image, mask_bin: np.ndarray):
    """mask_bin: uint8 HxW with 0 = background, 1 = lesion, 255 = ignore."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(root / "images" / f"{name}.jpg", quality=95)
    Image.fromarray(mask_bin.astype(np.uint8)).save(root / "masks" / f"{name}.png")
