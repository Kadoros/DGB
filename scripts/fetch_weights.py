"""Download Endo-FM and keep only the teacher backbone (2.3 GB -> 0.5 GB).

ResNet-50, DINOv2 and EndoViT are pulled from the HF Hub on first use.
"""
import subprocess
from pathlib import Path

import torch

GDRIVE_ID = "1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl"  # med-air/Endo-FM README


def main():
    out = Path("weights")
    out.mkdir(exist_ok=True)
    full = out / "endo_fm.pth"
    subprocess.run(["gdown", GDRIVE_ID, "-O", str(full)], check=True)
    ck = torch.load(full, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
    torch.save(sd, out / "endo_fm_teacher_backbone.pth")
    full.unlink()


if __name__ == "__main__":
    main()
