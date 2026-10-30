"""Frame-streamed LIBERO RAW demonstrations for the official pi05 action contract."""

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from owac.training.raw_ssl_data import FrameReader


def collate_actions(samples):
    batch = torch.utils.data.default_collate(samples)
    # All language padding is on the right; valid positions do not change.
    length = int(batch["token_mask"].sum(1).max())
    batch["tokens"] = batch["tokens"][:, :length]
    batch["token_mask"] = batch["token_mask"][:, :length]
    return batch


def action_chunk(actions, endpoint, stats, horizon=10):
    """Official quantile normalization, repeated terminal action, then zero padding."""
    ids = np.minimum(np.arange(endpoint, endpoint + horizon), len(actions) - 1)
    values = actions[ids].astype(np.float32)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    values = 2 * (values - low[:7]) / (high[:7] - low[:7] + 1e-6) - 1
    result = np.zeros((horizon, 32), np.float32)
    result[:, :7] = values
    return result


def geometric_augment(image, rng):
    """Official external-camera 95% crop and +/-5 degree rotation, in RAW space."""
    size = image.shape[-1]
    crop = int(size * 0.95)
    y, x = rng.integers(0, size - crop + 1, size=2)
    image = F.interpolate(
        image[None, :, y : y + crop, x : x + crop],
        (size, size),
        mode="bilinear",
        align_corners=False,
    )
    angle = float(rng.uniform(-5, 5)) * math.pi / 180
    c, s = math.cos(angle), math.sin(angle)
    theta = image.new_tensor([[[c, -s, 0], [s, c, 0]]])
    grid = F.affine_grid(theta, image.shape, align_corners=True)
    return F.grid_sample(image, grid, align_corners=True)[0]


class RawActionFrames(Dataset):
    def __init__(self, sources, cfg, *, validation=False):
        import sentencepiece

        self.sources, self.cfg, self.validation = sources, cfg, validation
        self.ends = np.cumsum([s.num_frames for s in sources])
        self.readers = OrderedDict()
        self.labels = OrderedDict()
        self.prompts = {}
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg["assets"]["tokenizer"])
        self.stats = json.loads(Path(cfg["assets"]["statistics"]).read_text())["norm_stats"]

    def __len__(self):
        return (
            len(self.sources)
            if self.validation
            else self.cfg["training"]["updates"] * self.cfg["training"]["batch"]
        )

    def frame(self, source, camera, endpoint):
        key = (str(source.path), camera)
        if key not in self.readers:
            self.readers[key] = FrameReader(source.path, camera)
            if len(self.readers) > 12:
                self.readers.popitem(last=False)[1].close()
        self.readers.move_to_end(key)
        return torch.from_numpy(self.readers[key].frame(endpoint)).permute(2, 0, 1).float() / 255

    def __getitem__(self, index):
        rng = np.random.default_rng(self.cfg["seed"] + index)
        if self.validation:
            source = self.sources[index]
            endpoint = source.num_frames // 2
        else:
            frame = int(rng.integers(self.ends[-1]))
            sid = int(np.searchsorted(self.ends, frame, side="right"))
            source = self.sources[sid]
            endpoint = frame - (int(self.ends[sid - 1]) if sid else 0)
        key = source.group_id
        if key not in self.labels:
            with np.load(source.path, allow_pickle=False) as archive:
                self.labels[key] = (
                    archive["actions"].copy(),
                    json.loads(str(archive["metadata_json"].item()))["task_description"],
                )
            if len(self.labels) > 16:
                self.labels.popitem(last=False)
        self.labels.move_to_end(key)
        actions, prompt = self.labels[key]
        if prompt not in self.prompts:
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            ids = self.tokenizer.encode(cleaned, add_bos=True) + self.tokenizer.encode("\n")
            tokens, mask = np.zeros(200, np.int64), np.zeros(200, bool)
            tokens[: len(ids[:200])] = ids[:200]
            mask[: len(ids[:200])] = True
            self.prompts[prompt] = tokens, mask
        images = torch.stack(
            [self.frame(source, k, endpoint) for k in ("agentview_raw_uint8", "wrist_raw_uint8")]
        )
        if not self.validation:
            images[0] = geometric_augment(images[0], rng)
        tokens, mask = self.prompts[prompt]
        return {
            "raw": images,
            "tokens": tokens,
            "token_mask": mask,
            "actions": action_chunk(actions, endpoint, self.stats["actions"]),
            "lighting": source.lighting,
            "source": key,
            "endpoint": endpoint,
        }
