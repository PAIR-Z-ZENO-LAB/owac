"""Fixed fruit-camera ISP and frozen DINOv3 spatial features."""

import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

WB = np.array([2.5, 1.0, 12 / 7], dtype=np.float32)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO_SHA = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"


def linear_rgb(raw):
    import cv2

    # Scale before bilinear interpolation so fractions of one native count survive OpenCV.
    expanded = np.ascontiguousarray(raw, dtype=np.uint16) * np.uint16(64)
    rgb = cv2.cvtColor(expanded, cv2.COLOR_BayerBGGR2RGB).astype(np.float32) / 64
    rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.3)
    return np.maximum(rgb - 64, 0) * (WB / 959)


def luminance_p99(linear):
    return float(np.quantile(linear @ LUMA, 0.99))


def color_image(raw, target=None):
    linear = linear_rgb(raw)
    gain = 1.0
    if target is not None:
        level = luminance_p99(linear)
        gain = float(np.clip(target / level, 1, 32)) if level > 0 else 32.0
    x = np.clip(linear * gain, 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.rint(srgb * 255).astype(np.uint8), gain


def linear_planes(raw, shape):
    """Form linear RGB from native BGGR cells, without a display transform."""
    import cv2

    r = raw[1::2, 1::2]
    g = (raw[::2, 1::2].astype(np.float32) + raw[1::2, ::2]) * 0.5
    b = raw[::2, ::2]
    linear = (np.stack((r, g, b), -1) - 64) / 959
    resized = cv2.resize(linear, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return resized.clip(0, 1).transpose(2, 0, 1).astype(np.float16)


def corp_rgb(raw, size=224):
    rgb = color_image(raw)[0]
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


class IspDino(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True)
        processor = json.loads((path / "preprocessor_config.json").read_text())
        self.register_buffer("mean", torch.tensor(processor["image_mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor["image_std"]).view(1, 3, 1, 1))
        self.scale = processor["rescale_factor"]
        self.target = None

    def pixels(self, rgb):
        x = torch.as_tensor(rgb, device=self.mean.device).permute(0, 3, 1, 2).float()
        return ((x * self.scale - self.mean) / self.std).contiguous()

    def encode_rgb(self, rgb):
        feats = self.encoder(pixel_values=self.pixels(rgb)).last_hidden_state
        patches = feats[:, 1 + self.encoder.config.num_register_tokens :]
        h, w = rgb.shape[1:3]
        if patches.shape[1] != (h // 16) * (w // 16):
            raise ValueError(f"Unexpected DINO patch grid for input {(h, w)}")
        return feats[:, 0], patches

    def forward(self, raw):
        rgb = np.stack([color_image(im, self.target)[0] for im in raw.cpu().numpy()])
        return self.encode_rgb(rgb)
