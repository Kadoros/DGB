"""Decoder heads: feature map (B,C,h,w) -> lesion logits (B,1,h,w)."""
import torch.nn as nn


class LinearHead(nn.Module):
    """BN + 1x1 conv, the standard linear probe for dense features."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.fc = nn.Conv2d(dim, 1, 1)

    def forward(self, f):
        return self.fc(self.bn(f))


class PatchMLPHead(nn.Module):
    """Per-patch 2-layer MLP (1x1 convs)."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 1))
        self.fc = self.net[-1]

    def forward(self, f):
        return self.net(f)


HEADS = {"linear": LinearHead, "mlp": PatchMLPHead}
