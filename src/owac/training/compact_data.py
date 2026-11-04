"""Task/lighting balanced RAW observations and aligned, valid action chunks."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import geometric_augment
from owac.training.raw_ssl_data import FrameReader


def normalize_state(values, statistics):
    return (values - np.asarray(statistics["state_mean"], np.float32)) / (
        np.asarray(statistics["state_std"], np.float32) + 1e-6
    )


def normalized_chunk(actions, endpoint, horizon, statistics):
    indices = np.arange(endpoint, endpoint + horizon)
    valid = indices < len(actions)
    values = actions[np.minimum(indices, len(actions) - 1)].astype(np.float32)
    low = np.asarray(statistics["action_q01"], np.float32)
    high = np.asarray(statistics["action_q99"], np.float32)
    return 2 * (values - low) / (high - low + 1e-6) - 1, valid


def unnormalize_actions(values, statistics):
    low = np.asarray(statistics["action_q01"], np.float32)
    high = np.asarray(statistics["action_q99"], np.float32)
    return (values + 1) * 0.5 * (high - low + 1e-6) + low


def prepare(cfg):
    """Read labels from training trajectories only; never load RGB members."""
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [s for s in sources if s.domain == "libero" and s.split == "train"]
    states, actions, names = [], [], set()
    for source in sources:
        with np.load(source.path, allow_pickle=False) as archive:
            states.append(archive["states"][:, 1:10].astype(np.float32))
            actions.append(archive["actions"][:, :7].astype(np.float32))
            names.add(json.loads(str(archive["metadata_json"].item()))["task_description"])
    state, action = np.concatenate(states), np.concatenate(actions)
    low, high = np.quantile(action, [0.01, 0.99], axis=0)
    cfg["statistics"] = {
        "state_mean": state.mean(0).tolist(),
        "state_std": state.std(0).tolist(),
        "action_q01": low.tolist(),
        "action_q99": high.tolist(),
        "training_sources": len(sources),
        "training_frames": len(state),
    }
    cfg["task_names"] = sorted(names)
    return cfg


class CompactFrames(Dataset):
    def __init__(self, cfg, *, split="train"):
        from transformers import T5TokenizerFast

        self.cfg, self.split = cfg, split
        sources, _ = load_inventory(cfg["data"]["inventory"])
        self.sources = [s for s in sources if s.domain == "libero" and s.split == split]
        self.lighting_names = sorted({s.lighting for s in self.sources})
        self.readers, self.labels = OrderedDict(), OrderedDict()
        tokenizer = T5TokenizerFast.from_pretrained(
            cfg["assets"]["tokenizer"], local_files_only=True
        )
        self.texts = tokenizer(
            cfg["task_names"],
            max_length=cfg["model"]["language_length"],
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        self.task_ids = {s: i for i, s in enumerate(cfg["task_names"])}
        groups = {}
        for index, source in enumerate(self.sources):
            task = (source.suite, source.task_name)
            groups.setdefault(task, {}).setdefault(source.lighting, []).append(index)
        self.groups = [list(v.values()) for _, v in sorted(groups.items())]
        self.validation_indices = [
            (sid, int(frame))
            for sid, source in enumerate(self.sources)
            for frame in np.linspace(
                0, source.num_frames - 1, cfg["validation"]["frames_per_source"], dtype=int
            )
        ]

    def __len__(self):
        return (
            self.cfg["training"]["updates"] * self.cfg["training"]["batch"]
            if self.split == "train"
            else len(self.validation_indices)
        )

    def frame(self, source, name, endpoint):
        key = (str(source.path), name)
        if key not in self.readers:
            self.readers[key] = FrameReader(source.path, name)
            if len(self.readers) > 12:
                self.readers.popitem(last=False)[1].close()
        self.readers.move_to_end(key)
        return torch.from_numpy(self.readers[key].frame(endpoint)).permute(2, 0, 1).float() / 255

    def __getitem__(self, index):
        rng = np.random.default_rng(self.cfg["seed"] + index)
        if self.split == "train":
            task = self.groups[int(rng.integers(len(self.groups)))]
            group = task[int(rng.integers(len(task)))]
            sid = group[int(rng.integers(len(group)))]
            endpoint = int(rng.integers(self.sources[sid].num_frames))
        else:
            sid, endpoint = self.validation_indices[index]
        source = self.sources[sid]
        key = source.group_id
        if key not in self.labels:
            with np.load(source.path, allow_pickle=False) as archive:
                self.labels[key] = (
                    archive["actions"][:, :7].copy(),
                    archive["states"][:, 1:10].astype(np.float32),
                    json.loads(str(archive["metadata_json"].item()))["task_description"],
                )
            if len(self.labels) > 16:
                self.labels.popitem(last=False)
        self.labels.move_to_end(key)
        actions, states, prompt = self.labels[key]
        task_id = self.task_ids[prompt]
        raw = torch.stack(
            [
                self.frame(source, name, endpoint)
                for name in ("agentview_raw_uint8", "wrist_raw_uint8")
            ]
        )
        if self.split == "train":
            raw[0] = geometric_augment(raw[0], rng)
        target, valid = normalized_chunk(
            actions, endpoint, self.cfg["model"]["action_horizon"], self.cfg["statistics"]
        )
        return {
            "raw": raw,
            "state": normalize_state(states[endpoint], self.cfg["statistics"]),
            "tokens": self.texts["input_ids"][task_id].astype(np.int64),
            "token_mask": self.texts["attention_mask"][task_id].astype(bool),
            "actions": target,
            "action_valid": valid,
            "task_id": task_id,
            "source_index": sid,
            "lighting_id": self.lighting_names.index(source.lighting),
            "endpoint": endpoint,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = prepare(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(cfg["statistics"]))


if __name__ == "__main__":
    main()
