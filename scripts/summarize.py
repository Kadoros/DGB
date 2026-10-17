"""Aggregate runs/*/metrics.json into a markdown table (mean ± std over seeds)."""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [  # (split, calib, key, label)
    ("test_ext", "raw", "dice", "Dice"),
    ("test_ext", "raw", "iou", "IoU"),
    ("test_ext", "raw", "pixel_recall", "픽셀 recall"),
    ("test_ext", "raw", "lesion_recall", "병변 recall"),
    ("test_ext", "raw", "ece", "ECE"),
    ("test_ext", "ts", "ece", "ECE (TS)"),
    ("test_ext", "raw", "flicker_mask", "flicker 1-IoU"),
    ("test_ext", "raw", "fp_comp_specular", "FP 반사 비율"),
    ("test_int", "raw", "dice", "Dice"),
    ("test_int", "raw", "lesion_recall", "병변 recall"),
    ("test_int", "raw", "neg_frame_fp_rate", "음성 프레임 FP율"),
    ("test_int", "ts", "ece", "ECE (TS)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*/metrics.json")
    ap.add_argument("--out", default="results/summary.md")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(glob.glob(a.glob)):
        name = re.sub(r"_s\d+$", "", f.split("/")[-2])
        groups[name].append(json.load(open(f)))
    lines = ["| 설정 | n | " + " | ".join(f"{sp.replace('test_', '')}: {lab}" for sp, _, _, lab in COLS) + " |",
             "|" + "---|" * (len(COLS) + 2)]
    for name, runs in groups.items():
        cells = []
        for sp, cal, key, _ in COLS:
            v = np.array([r[sp][cal].get(key, np.nan) for r in runs], float)
            cells.append(f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}")
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    T = {n: np.mean([r["temperature"] for r in rs]) for n, rs in groups.items()}
    lines += ["", "temperature (val fit): " + ", ".join(f"{n} {t:.2f}" for n, t in T.items())]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
