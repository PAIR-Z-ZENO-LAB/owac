"""Serve the exported RAW policy or official RGB pi05 baseline to the LIBERO runtime."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece
import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from owac.training.j5_pi05 import J5Policy, observation


class ActionPolicy:
    def __init__(self, cfg, checkpoint, baseline, device):
        self.proprio = cfg.get("proprioception")
        self.device, self.baseline = torch.device(device), baseline
        if baseline:
            from safetensors.torch import load_model

            self.model = PI0Pytorch(
                Pi0Config(
                    pi05=True,
                    action_horizon=10,
                    discrete_state_input=False,
                    pytorch_compile_mode=None,
                )
            )
            load_model(self.model, str(checkpoint))
        else:
            payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            self.model = J5Policy(cfg, initialize=False)
            if cfg.get("adaptation", {}).get("kind") == "lora":
                from owac.training.j5_lora import configure_lora

                configure_lora(self.model, cfg)
            self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg["assets"]["tokenizer"])
        self.stats = json.loads(Path(cfg["assets"]["statistics"]).read_text())["norm_stats"][
            "actions"
        ]

    @torch.inference_mode()
    def infer(self, request):
        prompt = request["prompt"].strip().replace("_", " ").replace("\n", " ")
        ids = (self.tokenizer.encode(prompt, add_bos=True) + self.tokenizer.encode("\n"))[:200]
        state = None
        if self.proprio and not self.baseline:
            from owac.training.j5_proprio import normalize_proprio, tokenize_proprio

            state = normalize_proprio(request["proprio"], self.proprio["statistics"])
            ids, valid = tokenize_proprio(self.tokenizer, prompt, state)
            ids = ids[valid].tolist()
        tokens = torch.tensor([ids], device=self.device)
        masks = torch.ones_like(tokens, dtype=torch.bool)
        yes = torch.ones(1, dtype=torch.bool, device=self.device)
        if self.baseline:
            from openpi.shared.image_tools import resize_with_pad_torch

            images = (
                torch.tensor(request["rgb"], device=self.device).permute(0, 3, 1, 2).float() / 127.5
                - 1
            )
            # Resize the two-view batch before splitting: the pinned helper squeezes
            # a size-one channels-last batch in its internal resize path.
            images = resize_with_pad_torch(images, 224, 224)
            obs = SimpleNamespace(
                images={
                    "base_0_rgb": images[0:1],
                    "left_wrist_0_rgb": images[1:2],
                    "right_wrist_0_rgb": torch.zeros_like(images[0:1]),
                },
                image_masks={"base_0_rgb": yes, "left_wrist_0_rgb": yes, "right_wrist_0_rgb": ~yes},
                state=torch.zeros(1, 32, device=self.device),
                tokenized_prompt=tokens,
                tokenized_prompt_mask=masks,
                token_ar_mask=None,
                token_loss_mask=None,
            )
        else:
            raw = torch.tensor(request["raw"], device=self.device).permute(0, 3, 1, 2).float() / 255
            batch = {"raw": raw[None], "tokens": tokens, "token_mask": masks}
            if state is not None:
                batch["state"] = torch.tensor(state[None], device=self.device)
            obs = observation(batch)
        noise = torch.tensor(request["noise"], device=self.device).float()[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = self.model.sample_actions(self.device, obs, noise=noise, num_steps=10)[
                0, :, :7
            ]
        normalized = normalized.float().cpu().numpy()
        low, high = np.asarray(self.stats["q01"])[:7], np.asarray(self.stats["q99"])[:7]
        actions = (normalized + 1) / 2 * (high - low + 1e-6) + low
        return {"actions": actions.astype(np.float32), "normalized_actions": normalized}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=9015)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = json.loads(args.config.read_text())
    policy = ActionPolicy(cfg, args.checkpoint, args.baseline, args.device)
    WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=args.port,
        metadata={"role": "official-rgb" if args.baseline else "J5-RAW"},
    ).serve_forever()


if __name__ == "__main__":
    main()
