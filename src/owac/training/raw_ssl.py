"""RAW self-supervision: radiometric tokens and DINO/iBOT objectives.

No pretrained features, RGB target, action, or image reconstruction is used.
The response pivot is learned, not a calibrated sensor noise parameter.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class RadiometricResponse(nn.Module):
    """Keep magnitude, compress dynamic range, and expose relative channel response."""

    def __init__(self, pivot=0.02):
        super().__init__()
        self.log_pivot = nn.Parameter(torch.full((1, 3, 1, 1), math.log(pivot)))

    def forward(self, raw):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            x = raw.float()
            pivot = self.log_pivot.exp()
            # asinh also supports signed measurements; no saturation/clipping of HDR.
            compressed = torch.asinh(x / pivot) / torch.asinh(1 / pivot)
            relative = compressed - compressed.mean(1, keepdim=True)
            return torch.cat((x, compressed, relative), 1)


class RawEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        self.measurement_config = cfg.get("measurement_encoder")
        if self.measurement_config:
            from owac.training.raw_measurements import MeasurementResponse

            response_class = MeasurementResponse
            if self.measurement_config.get("adc_features"):
                from owac.training.raw_adc_features import ADCMeasurementResponse

                response_class = ADCMeasurementResponse
            self.response = response_class(self.measurement_config, cfg["response_pivot"])
        else:
            self.response = RadiometricResponse(cfg["response_pivot"])
        architecture = DINOv3ViTConfig(
            image_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
            num_channels=5 * len(self.response.filter_names) if self.measurement_config else 9,
            hidden_size=cfg["width"],
            intermediate_size=4 * cfg["width"],
            num_hidden_layers=cfg["depth"],
            num_attention_heads=cfg["heads"],
            num_register_tokens=4,
            layerscale_value=1e-5,
            drop_path_rate=cfg["drop_path"],
            pos_embed_rescale=2.0,
        )
        architecture._attn_implementation = "sdpa"
        self.backbone = DINOv3ViTModel(architecture)
        if self.measurement_config:
            from owac.training.raw_sensel import SenselPatchEmbedding

            patch_class = SenselPatchEmbedding
            if self.measurement_config.get("adc_features"):
                from owac.training.raw_adc_features import ADCPatchEmbedding

                patch_class = ADCPatchEmbedding
            self.backbone.embeddings.patch_embeddings = patch_class(
                cfg["width"], cfg["patch_size"], self.measurement_config
            )
        elif cfg.get("photon_evidence"):
            from owac.training.raw_evidence import EvidencePatchEmbedding

            embeddings = self.backbone.embeddings
            embeddings.patch_embeddings = EvidencePatchEmbedding(
                embeddings.patch_embeddings, cfg["photon_evidence"]
            )

    def forward(self, raw, mask=None, *, sensor=None, valid=None):
        values = (
            self.response(raw, sensor=sensor, valid=valid)
            if self.measurement_config
            else self.response(raw)
        )
        tokens = self.backbone(pixel_values=values, bool_masked_pos=mask).last_hidden_state
        return tokens[:, 0], tokens[:, 5:]


def build_encoder(cfg):
    if cfg.get("input_encoding", "owac") == "owac" and not cfg.get("freeze_for_action_readout"):
        return RawEncoder(cfg)
    from owac.training.raw_mechanism import encoder_class

    return encoder_class(RawEncoder)(cfg)


class PrototypeHead(nn.Module):
    def __init__(self, width, prototypes, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.weight = nn.Parameter(torch.randn(prototypes, bottleneck) * 0.02)

    def forward(self, x):
        return F.linear(F.normalize(self.mlp(x), dim=-1), F.normalize(self.weight, dim=-1))


class RawSSL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = build_encoder(cfg["model"])
        width = cfg["model"]["width"]
        self.global_head = PrototypeHead(width, cfg["loss"]["prototypes"])
        self.patch_head = PrototypeHead(width, cfg["loss"]["prototypes"])

    def forward(self, global_views, local_views=None, mask=None):
        cls, patches = self.encoder(global_views, mask)
        result = {"global": self.global_head(cls), "cls": cls, "patches": patches}
        if mask is not None:
            result["masked"] = self.patch_head(patches[mask])
        if local_views is not None:
            local, _ = self.encoder(local_views)
            result["local"] = self.global_head(local)
        return result


@torch.no_grad()
def sinkhorn(logits, temperature, iterations=3):
    """Distributed prototype balancing, with FP32 exponentiation and reductions."""
    logits = logits.float() / temperature
    largest = logits.max()
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    q = (logits - largest).exp().T
    total = q.sum()
    count = q.new_tensor(q.shape[1])
    if distributed:
        dist.all_reduce(total)
        dist.all_reduce(count)
    q /= total
    for _ in range(iterations):
        rows = q.sum(1, keepdim=True)
        if distributed:
            dist.all_reduce(rows)
        q /= rows.clamp_min(torch.finfo(q.dtype).tiny)
        q /= q.shape[0]
        q /= q.sum(0, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        q /= count
    return (q * count).T


def cross_entropy(logits, targets, temperature=0.1):
    return -(targets * F.log_softmax(logits.float() / temperature, dim=-1)).sum(-1).mean()


def koleo(cls):
    x = F.normalize(cls.float(), dim=-1)
    similarity = x @ x.T
    similarity.fill_diagonal_(-torch.inf)
    nearest = similarity.argmax(-1)
    distance = (x - x[nearest]).norm(dim=-1)
    return -distance.clamp_min(torch.finfo(x.dtype).eps).log().mean()


def objective(student, teacher, patch_targets, batch, locals_count, cfg):
    tg = teacher.reshape(2, batch, -1)
    sg = student["global"].reshape(2, batch, -1)
    terms = [cross_entropy(sg[0], tg[1]), cross_entropy(sg[1], tg[0])]
    for local in student["local"].reshape(locals_count, batch, -1):
        terms.extend(cross_entropy(local, target) for target in tg)
    global_loss = torch.stack(terms).mean()
    patch_loss = cross_entropy(student["masked"], patch_targets)
    # Different crops of the same image are not nearest-neighbor negatives.
    entropy = torch.stack([koleo(c) for c in student["cls"].chunk(2)]).mean()
    loss = global_loss + cfg["ibot_weight"] * patch_loss + cfg["koleo_weight"] * entropy
    return loss, {
        "global": global_loss.detach(),
        "ibot": patch_loss.detach(),
        "koleo": entropy.detach(),
    }
