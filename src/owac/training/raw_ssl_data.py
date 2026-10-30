"""Deterministic source-balanced RAW views, with worker-local read-only mappings."""

import math
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from owac.raw.all_bench_distillation import DOMAIN_RAW_KEYS, read_npz_member
from owac.training.raw_exposure import lower_exposure


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


class FrameReader:
    """Read a single stored NPY frame without mmap readahead of the episode."""

    def __init__(self, path, camera):
        array = read_npz_member(path, camera + ".npy")
        self.array = None
        self.fd = None
        if isinstance(array, np.memmap) and array.flags.c_contiguous:
            self.shape, self.dtype = array.shape[1:], array.dtype
            self.offset = array.offset
            self.frame_bytes = math.prod(self.shape) * self.dtype.itemsize
            self.fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
            array._mmap.close()
        else:
            # Existing compressed archives need NumPy decompression.
            self.array = array

    def frame(self, index):
        if self.array is not None:
            return np.array(self.array[index], copy=True)
        offset = self.offset + index * self.frame_bytes
        data = os.pread(self.fd, self.frame_bytes, offset)
        array = np.frombuffer(data, dtype=self.dtype).reshape(self.shape).copy()
        page = os.sysconf("SC_PAGE_SIZE")
        start = offset // page * page
        end = (offset + self.frame_bytes + page - 1) // page * page
        os.posix_fadvise(self.fd, start, end - start, os.POSIX_FADV_DONTNEED)
        return array

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


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
            self.cache[key] = FrameReader(source.path, camera)
            if len(self.cache) > 12:
                self.cache.popitem(last=False)[1].close()
        self.cache.move_to_end(key)
        array = self.cache[key].frame(endpoint)
        x = torch.from_numpy(array).permute(2, 0, 1).float()
        return x / 255 if source.domain == "libero" else x

    def close(self):
        for reader in self.cache.values():
            reader.close()
        self.cache.clear()

    def __len__(self):
        return self.cfg["training"]["updates"] * self.cfg["training"]["batch"]

    def __getitem__(self, index):
        rng = random.Random(self.cfg["seed"] + index)
        stratum = self.strata[rng.randrange(len(self.strata))]
        exposure = self.cfg.get("exposure")
        if exposure and rng.random() < exposure["normal_sampling"]:
            stratum = (stratum[0], "Normal")
        group = self.groups[stratum]
        source = self.sources[rng.choice(group)]
        camera = rng.choice(DOMAIN_RAW_KEYS[source.domain])
        x = self.frame(source, rng.randrange(source.num_frames), camera)
        c = self.cfg["crops"]
        if exposure:
            step = index // self.cfg["training"]["batch"]
            maximum = exposure["start_stops"] + (
                exposure["max_stops"] - exposure["start_stops"]
            ) * min(step / exposure["curriculum_steps"], 1)
            globals_, degraded = [], []
            generator = torch.Generator().manual_seed(self.cfg["seed"] + index)
            for _ in range(2):
                stops = rng.uniform(0, maximum) if source.lighting == "Normal" else 0.0
                dark = (
                    lower_exposure(x, stops, exposure["noise"][source.domain], generator)
                    if stops
                    else x
                )
                # Identical crop, flip, resize, and mild gain; noise is sampled BEFORE resize.
                geometry = rng.getstate()
                globals_.append(crop(x, c["global_size"], c["global_scale"], rng))
                rng.setstate(geometry)
                degraded.append(crop(dark, c["global_size"], c["global_scale"], rng))
            local = torch.stack(
                [crop(x, c["local_size"], c["local_scale"], rng) for _ in range(c["local_count"])]
            )
            return torch.stack(globals_), local, torch.stack(degraded)
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
