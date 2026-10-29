"""Deterministic source-balanced RAW views, with worker-local read-only mappings."""

import math
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from owac.raw.all_bench_distillation import DOMAIN_RAW_KEYS, read_npz_member


def crop(x, size, scale, rng):
    height, width = x.shape[-2:]
    area = rng.uniform(*scale) * height * width
    ratio = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
    h = min(height, round(math.sqrt(area / ratio)))
    w = min(width, round(math.sqrt(area * ratio)))
    y, z = rng.randrange(height - h + 1), rng.randrange(width - w + 1)
    x = x[:, y : y + h, z : z + w]
    x = F.interpolate(
        x[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )[0]
    if rng.random() < 0.5:
        x = x.flip(-1)
    # A modest exposure change; retain HDR values rather than clipping to display range.
    return x * math.exp(rng.uniform(-0.15, 0.15))


class RawViews(Dataset):
    def __init__(self, sources, cfg):
        self.sources = sources
        self.cfg = cfg
        self.cache = OrderedDict()
        self.groups = {}
        for i, source in enumerate(sources):
            self.groups.setdefault((source.domain, source.lighting), []).append(i)
        self.strata = sorted(self.groups)

    def frame(self, source, endpoint, camera):
        key = (str(source.path), camera)
        if key not in self.cache:
            self.cache[key] = read_npz_member(source.path, camera + ".npy")
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        array = np.array(self.cache[key][endpoint], copy=True)
        x = torch.from_numpy(array).permute(2, 0, 1).float()
        return x / 255 if source.domain == "libero" else x

    def __len__(self):
        return self.cfg["training"]["updates"] * self.cfg["training"]["batch"]

    def __getitem__(self, index):
        rng = random.Random(self.cfg["seed"] + index)
        group = self.groups[self.strata[rng.randrange(len(self.strata))]]
        source = self.sources[rng.choice(group)]
        camera = rng.choice(DOMAIN_RAW_KEYS[source.domain])
        x = self.frame(source, rng.randrange(source.num_frames), camera)
        c = self.cfg["crops"]
        return (
            torch.stack([crop(x, c["global_size"], c["global_scale"], rng) for _ in range(2)]),
            torch.stack(
                [crop(x, c["local_size"], c["local_scale"], rng) for _ in range(c["local_count"])]
            ),
        )


class ResumeSampler(Sampler):
    def __init__(self, start_step, updates, batch, rank, world):
        self.start, self.updates, self.batch = start_step, updates, batch
        self.rank, self.world = rank, world

    def __iter__(self):
        per_rank = self.batch // self.world
        for step in range(self.start, self.updates):
            start = step * self.batch + self.rank * per_rank
            yield from range(start, start + per_rank)

    def __len__(self):
        return (self.updates - self.start) * (self.batch // self.world)


def block_masks(batch, grid, device):
    """Equal masked counts per image; rectangular blocks with random cyclic origins."""
    fraction = float(torch.empty(()).uniform_(0.1, 0.5))
    h = int(torch.randint(grid // 2, grid + 1, ()).item())
    w = min(grid, max(1, round(fraction * grid * grid / h)))
    mask = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    for i in torch.randperm(batch)[: batch // 2].tolist():
        y, x = torch.randint(grid, (2,)).tolist()
        ys = (torch.arange(h, device=device) + y) % grid
        xs = (torch.arange(w, device=device) + x) % grid
        mask[i, ys[:, None], xs[None, :]] = True
    return mask.flatten(1)
