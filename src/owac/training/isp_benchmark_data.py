"""Bounded frame reads for frozen-policy ISP adaptation on the original split."""

import json
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from owac.raw.all_bench_distillation import load_inventory
from owac.training.raw_ssl_data import FrameReader


def causal_indices(current, *, observations, burst, stride):
    endpoints = np.maximum(current - stride * np.arange(observations - 1, -1, -1), 0)
    return np.maximum(endpoints[:, None] - np.arange(burst - 1, -1, -1), 0)


class ISPData(Dataset):
    def __init__(self, cfg, policy, frontend):
        sources, _ = load_inventory(cfg["inventory"])
        self.sources = [s for s in sources if s.domain == "libero" and s.split == "train"]
        self.policy, self.frontend = policy, frontend
        self.seed = cfg["training"]["seed"]
        self.length = cfg["training"]["updates"] * cfg["training"]["batch"]
        self.horizon = 12 if policy == "corp" else 10
        self.stride = 12 if policy == "corp" else 5
        self.labels = OrderedDict()
        self.readers = OrderedDict()

    def __len__(self):
        return self.length

    def _episode(self, path):
        key = str(path)
        if key not in self.labels:
            with np.load(path, allow_pickle=False) as archive:
                self.labels[key] = {
                    "actions": archive["actions"],
                    "state": archive["robot_states"],
                    "metadata": json.loads(str(archive["metadata_json"].item())),
                }
            if len(self.labels) > 8:
                self.labels.popitem(last=False)
        self.labels.move_to_end(key)
        return self.labels[key]

    def _frames(self, path, camera, indices):
        key = (str(path), camera)
        if key not in self.readers:
            self.readers[key] = FrameReader(path, camera)
            if len(self.readers) > 8:
                self.readers.popitem(last=False)[1].close()
        self.readers.move_to_end(key)
        unique, inverse = np.unique(indices, return_inverse=True)
        frames = np.stack([self.readers[key].frame(int(i)) for i in unique])
        return frames[inverse].reshape(*indices.shape, *frames.shape[1:])

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        source = self.sources[int(rng.integers(len(self.sources)))]
        episode = self._episode(source.path)
        current = int(rng.integers(len(episode["actions"])))
        recurrent, burst = (6, 6) if self.frontend == "rawvla" else (8, 1)
        indices = causal_indices(current, observations=recurrent, burst=burst, stride=self.stride)
        raw = np.stack(
            [
                self._frames(source.path, camera, indices)
                for camera in ("agentview_raw_uint8", "wrist_raw_uint8")
            ]
        )
        endpoints = indices[:, -1] if self.frontend == "rawild" else np.asarray([current])
        action_indices = np.minimum(
            endpoints[:, None] + np.arange(self.horizon), len(episode["actions"]) - 1
        )
        return {
            "raw": raw,
            "state": episode["state"][endpoints].astype(np.float32),
            "actions": episode["actions"][action_indices].astype(np.float32),
            "prompt": episode["metadata"]["task_description"],
            "lighting": source.lighting,
        }


def collate_samples(samples):
    return samples
