"""Integer shifts of original RAW views for training, without radiometric changes.

The replicate-padding/random-crop rule follows DrQ-v2 RandomShiftsAug (MIT):
facebookresearch/drqv2, revision c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef.
This implementation uses integer indexing instead of floating-point grid sampling
and accepts rectangular views. It does not import the upstream RL algorithm.
"""

import torch
from torch.nn import functional as F


def shift_raw_views(raw, offsets, padding, *, padding_mode="replicate"):
    """Crop VCHW views at explicit (y, x) offsets in replicate-padded images.

    Offsets are in [0, 2*padding]; (padding, padding) is the original image.
    Reuse the same offsets for paired exposures or successive history frames.
    """
    height, width = raw.shape[-2:]
    padded = F.pad(raw, (padding,) * 4, mode=padding_mode)
    return torch.stack(
        [
            view[:, y : y + height, x : x + width]
            for view, (y, x) in zip(padded, offsets, strict=True)
        ]
    )
