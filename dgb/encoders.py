"""Swappable frozen encoders: image (B,3,H,W) -> feature map (B,C,h,w).

Every encoder exposes the same interface so that heads, training and
evaluation stay identical across the 1-a.1 comparison:
    enc = build_encoder(name); feats = enc(x)   # x already normalised by enc.mean/std
    enc.dim, enc.stride, enc.mean, enc.std
"""
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class Encoder(nn.Module):
    dim: int
    stride: int
    mean = IMAGENET[0]
    std = IMAGENET[1]


class RawPatch(Encoder):
    """No pretrained encoder: each stride x stride RGB patch flattened (baseline proxy)."""

    def __init__(self, stride=16):
        super().__init__()
        self.stride, self.dim = stride, 3 * stride * stride

    def forward(self, x):
        return F.pixel_unshuffle(x, self.stride)


class ResNet50(Encoder):
    """ImageNet ResNet-50, last stage dilated to output stride 16 (DeepLab style)."""

    def __init__(self):
        super().__init__()
        self.m = timm.create_model("resnet50.a1_in1k", pretrained=True, features_only=True,
                                   out_indices=(4,), output_stride=16)
        self.dim, self.stride = 2048, 16

    def forward(self, x):
        return self.m(x)[0]


class TimmViT(Encoder):
    """Patch tokens of the last block (after final norm) reshaped to a grid."""

    def __init__(self, model, stride, mean=IMAGENET[0], std=IMAGENET[1], state_dict=None):
        super().__init__()
        self.m = model
        self.dim, self.stride, self.mean, self.std = model.embed_dim, stride, mean, std
        if hasattr(self.m, "set_input_size"):
            self.m.dynamic_img_size = True

    def forward(self, x):
        B, _, H, W = x.shape
        t = self.m.forward_features(x)
        t = t[:, self.m.num_prefix_tokens:]
        return t.transpose(1, 2).reshape(B, self.dim, H // self.stride, W // self.stride)


def dinov2_vitb14():
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                          dynamic_img_size=True, num_classes=0)
    return TimmViT(m, 14)


def endovit_vitb16():
    """EndoViT (MAE ViT-B/16 on Endo700k). Weights: HF egeozsoy/EndoViT."""
    from timm.models.vision_transformer import VisionTransformer
    m = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          num_classes=0, dynamic_img_size=True)
    sd = torch.load(hf_hub_download("egeozsoy/EndoViT", "pytorch_model.bin"),
                    map_location="cpu", weights_only=False)["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("head")], missing
    # dataset statistics from the EndoViT demo
    return TimmViT(m, 16, mean=(0.3464, 0.2280, 0.2228), std=(0.2520, 0.2128, 0.2093))


class EndoFMFrame(Encoder):
    """Endo-FM (TimeSformer ViT-B/16, divided space-time) used on single frames (T=1).

    With T=1 the temporal attention reduces to proj(v); the branch is kept so the
    pretrained residual stream is unchanged. Teacher backbone weights extracted from
    the official checkpoint (Google Drive 1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl) by
    scripts/fetch_weights.py.
    """

    def __init__(self, path="weights/endo_fm_teacher_backbone.pth"):
        super().__init__()
        from timm.models.vision_transformer import Attention, Mlp
        sd = torch.load(path, map_location="cpu")
        d, self.dim, self.stride = 768, 768, 16
        self.patch_embed = nn.Conv2d(3, d, 16, 16)
        self.cls_token = nn.Parameter(sd["cls_token"])
        self.pos_embed = nn.Parameter(sd["pos_embed"])
        self.time_embed = nn.Parameter(sd["time_embed"])
        ln = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList()
        for _ in range(12):
            b = nn.Module()
            b.norm1, b.temporal_norm1, b.norm2 = ln(d), ln(d), ln(d)
            b.attn = Attention(d, 12, qkv_bias=True)
            b.temporal_attn = Attention(d, 12, qkv_bias=True)
            b.temporal_fc = nn.Linear(d, d)
            b.mlp = Mlp(d, 4 * d)
            self.blocks.append(b)
        self.norm = ln(d)
        sd = {k.replace("patch_embed.proj", "patch_embed"): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        # Endo-FM pretraining normalisation (ImageNet-style values in their config)
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def _temporal_t1(self, attn, x):
        # softmax over a single key is 1, so the output is proj(v)
        _, v = attn.qkv(x).chunk(3, dim=-1)[1:]
        return attn.proj(v)

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // 16, W // 16
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        pos = self.pos_embed
        if pos.shape[1] - 1 != h * w:
            g = int((pos.shape[1] - 1) ** 0.5)
            grid = pos[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
            pos = torch.cat([pos[:, :1], grid.flatten(2).transpose(1, 2)], 1)
        cls = self.cls_token.expand(B, -1, -1) + pos[:, :1]
        p = p + pos[:, 1:] + self.time_embed[:, :1]
        x = torch.cat([cls, p], 1)
        for b in self.blocks:
            xt = x[:, 1:] + b.temporal_fc(self._temporal_t1(b.temporal_attn, b.temporal_norm1(x[:, 1:])))
            xs = torch.cat([x[:, :1], xt], 1)
            res = b.attn(b.norm1(xs))
            x = torch.cat([x[:, :1], xt], 1) + res
            x = x + b.mlp(b.norm2(x))
        x = self.norm(x)[:, 1:]
        return x.transpose(1, 2).reshape(B, self.dim, h, w)


REGISTRY = {
    "rawpatch": RawPatch,
    "resnet50": ResNet50,
    "dinov2_b14": dinov2_vitb14,
    "endovit_b16": endovit_vitb16,
    "endofm_b16": EndoFMFrame,
}


def build_encoder(name: str) -> Encoder:
    enc = REGISTRY[name]()
    return enc.eval()
