"""Serve the RAW-native compact policy to the existing native simulator."""

import argparse
from pathlib import Path

import numpy as np
import torch
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from transformers import T5TokenizerFast

from owac.paper.model import CompactPolicy
from owac.training.compact_data import normalize_state, unnormalize_actions


class CompactActionPolicy:
    def __init__(self, checkpoint, device):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.cfg = payload["config"]
        self.device = torch.device(device)
        self.model = CompactPolicy(self.cfg).to(self.device)
        self.model.load_state_dict(payload.get("ema", payload["model"]))
        self.model.eval()
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.cfg["assets"]["tokenizer"], local_files_only=True
        )
        self.prompts = {}

    @torch.inference_mode()
    def infer(self, request):
        prompt = request["prompt"]
        if prompt not in self.prompts:
            encoded = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.cfg["model"]["language_length"],
                return_tensors="pt",
            )
            self.prompts[prompt] = (
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device, dtype=torch.bool),
            )
        tokens, mask = self.prompts[prompt]
        batch = {
            "raw": torch.tensor(np.asarray(request["raw"]), device=self.device)
            .permute(0, 3, 1, 2)
            .float()[None]
            / 255,
            "state": torch.as_tensor(
                normalize_state(np.asarray(request["proprio"], np.float32), self.cfg["statistics"]),
                device=self.device,
            )[None],
            "tokens": tokens,
            "token_mask": mask,
        }
        noise = torch.tensor(np.asarray(request["noise"]), device=self.device).float()[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = self.model.sample(batch, noise)[0].float().cpu().numpy()
        return {
            "actions": unnormalize_actions(normalized, self.cfg["statistics"]).astype(np.float32),
            "normalized_actions": normalized,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=9036)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy = CompactActionPolicy(args.checkpoint, args.device)
    WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata={"role": "RAW-compact"}
    ).serve_forever()


if __name__ == "__main__":
    main()
