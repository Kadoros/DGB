"""Ignore mask for the black frame around the endoscope view.

Only the border-connected, near-black region is ignored. The lumen is also dark
but is not connected to the image border, so it stays as background.
Instruments and blur have no public GT and are not handled yet.
"""
import numpy as np
from scipy import ndimage


def black_border(rgb: np.ndarray, thr: int = 15) -> np.ndarray:
    dark = rgb.max(axis=2) < thr
    dark = ndimage.binary_opening(dark, iterations=2)
    lab, _ = ndimage.label(dark)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    return np.isin(lab, edge)


def build_mask(rgb: np.ndarray, lesion: np.ndarray) -> np.ndarray:
    m = lesion.astype(np.uint8)
    m[black_border(rgb) & (m == 0)] = 255
    return m
