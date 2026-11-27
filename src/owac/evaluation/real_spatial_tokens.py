"""Final, full-field spatial descriptors used for the real tracking and PCA experiments."""

import torch
from torch.nn import functional as F


def encode(model, name, values):
    if name == "owac":
        with torch.autocast(values.device.type, dtype=torch.bfloat16, enabled=values.is_cuda):
            tok = model(values.float())[1]
    else:
        x = values.permute(0, 3, 1, 2).contiguous().float() / 255
        patch = model.config.patch_size
        if patch != 16:
            x = F.interpolate(
                x,
                size=(30 * patch, 40 * patch),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if name == "siglip2":
            mean = std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        x = (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(std)[None, :, None, None]
        with torch.autocast(values.device.type, dtype=torch.bfloat16, enabled=values.is_cuda):
            if name == "siglip2":
                tok = model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            else:
                tok = model(pixel_values=x).last_hidden_state[
                    :, 1 + getattr(model.config, "num_register_tokens", 0) :
                ]
    assert tok.shape[1] == 1200, str(tok.shape)
    return tok.transpose(1, 2).reshape(len(tok), -1, 30, 40).float()
