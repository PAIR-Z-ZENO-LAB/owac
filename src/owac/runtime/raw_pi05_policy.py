"""RAW-only inference from an offline-selected pi05 adaptation checkpoint."""

import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from transformers.models.gemma import modeling_gemma

from owac.runtime.raw_pi05_attention import dense_eager_attention
from owac.training.j5_pi05 import observation
from owac.training.raw_pi05_model import build_model


class RawPi05Policy:
    def __init__(self, item):
        # Scope the layout repair to this policy process; do not modify installed libraries.
        modeling_gemma.eager_attention_forward = dense_eager_attention
        with Path(item["checkpoint"]).open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = item.get("checkpoint_sha256")
        if expected and self.checkpoint_sha256 != expected:
            raise ValueError("Selected policy checkpoint differs from the evaluation config")
        payload = torch.load(item["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        self.checkpoint_step = payload.get("step")
        cfg = payload["config"]
        del payload
        self.model = build_model(cfg, item["checkpoint"]).cuda().eval()
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg["assets"]["tokenizer"])
        self.stats = json.loads(Path(cfg["assets"]["statistics"]).read_text())["norm_stats"][
            "actions"
        ]
        self.prompts = {}

    @torch.inference_mode()
    def predict(self, request):
        prompt = request["prompt"].strip().replace("_", " ").replace("\n", " ")
        if prompt not in self.prompts:
            ids = (self.tokenizer.encode(prompt, add_bos=True) + self.tokenizer.encode("\n"))[:200]
            self.prompts[prompt] = torch.tensor([ids], device="cuda")
        tokens = self.prompts[prompt]
        batch = {
            "raw": torch.as_tensor(request["raw"], device="cuda").permute(0, 3, 1, 2)[None].float()
            / 255,
            "tokens": tokens,
            "token_mask": torch.ones_like(tokens, dtype=torch.bool),
        }
        generator = torch.Generator(device="cuda").manual_seed(request["seed"])
        noise = torch.randn((1, 10, 32), generator=generator, device="cuda", dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = (
                self.model.sample_actions(
                    torch.device("cuda"), observation(batch), noise=noise, num_steps=10
                )[0, :, :7]
                .float()
                .cpu()
                .numpy()
            )
        low, high = np.asarray(self.stats["q01"])[:7], np.asarray(self.stats["q99"])[:7]
        return ((normalized + 1) / 2 * (high - low + 1e-6) + low).astype(np.float32)
