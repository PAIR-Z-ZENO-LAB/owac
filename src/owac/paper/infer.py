"""Predict a LIBERO action chunk from two pseudo-RAW views, robot state and an instruction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import T5TokenizerFast

from owac.paper.checkpoint import TOKENIZER, load_policy
from owac.training.compact_data import normalize_state, unnormalize_actions


class LiberoPolicy:
    def __init__(self, checkpoint, device="cpu", tokenizer=TOKENIZER):
        self.device = torch.device(device)
        self.model, self.config, self.step = load_policy(checkpoint, self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer, local_files_only=True)
        self.text = {}

    @torch.inference_mode()
    def predict(self, raw, proprio, prompt, *, seed=0):
        """Return 12 clipped, denormalized 7D delta commands in the benchmark convention."""
        frames = np.asarray(raw)
        state = np.asarray(proprio, dtype=np.float32)
        if frames.shape != (2, 256, 256, 3) or frames.dtype != np.uint8:
            raise ValueError("LIBERO input requires two upright 256x256x3 uint8 pseudo-RAW views")
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError("Expected seven joint angles and two gripper positions")
        if prompt not in self.text:
            self.text[prompt] = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.config["model"]["language_length"],
                return_tensors="pt",
            ).to(self.device)
        text = self.text[prompt]
        stats = self.config["statistics"]
        batch = {
            "raw": torch.from_numpy(np.ascontiguousarray(frames))
            .to(self.device)
            .permute(0, 3, 1, 2)[None]
            .float()
            / 255,
            "state": torch.as_tensor(normalize_state(state, stats), device=self.device)[None],
            "tokens": text.input_ids,
            "token_mask": text.attention_mask.bool(),
        }
        cuda = self.device.type == "cuda"
        noise = torch.randn(
            (1, self.config["model"]["action_horizon"], self.config["model"]["action_dim"]),
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=torch.bfloat16 if cuda else torch.float32,
        )
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cuda):
            normalized = self.model.sample(batch, noise)[0].float().cpu().numpy()
        return np.clip(unnormalize_actions(normalized, stats), -1, 1)


def observation(episode, frame):
    with np.load(episode, allow_pickle=False) as data:
        raw = np.stack([data[k][frame] for k in ("agentview_raw_uint8", "wrist_raw_uint8")])
        state = data["states"][frame, 1:10]
        prompt = json.loads(str(data["metadata_json"].item()))["task_description"]
    return raw, state, prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, help="a public LIBERO RAW demonstration NPZ")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional NumPy action array")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    policy = LiberoPolicy(args.checkpoint, args.device)
    if args.episode:
        raw, state, prompt = observation(args.episode, args.frame)
    else:
        raw = np.random.default_rng(0).integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        state = np.zeros(9, dtype=np.float32)
        prompt = "put the cream cheese in the bowl"
    actions = policy.predict(raw, state, prompt, seed=args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions)
    print(
        json.dumps(
            {
                "input": str(args.episode or "synthetic"),
                "step": policy.step,
                "prompt": prompt,
                "shape": list(actions.shape),
                "actions": actions.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
