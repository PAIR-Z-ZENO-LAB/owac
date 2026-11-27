"""Serve fixed RAW checkpoints using their archived model implementation."""

import argparse
import json
import queue
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch


class CompactRawPolicy:
    def __init__(self, item):
        from transformers import T5TokenizerFast

        from owac.paper.model import CompactPolicy

        payload = torch.load(item["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        self.cfg = payload["config"]
        self.model = CompactPolicy(self.cfg, initialize=False).cuda().eval()
        self.model.load_state_dict(payload.get("ema", payload.get("model")), strict=True)
        self.tokenizer = T5TokenizerFast.from_pretrained(item["tokenizer"], local_files_only=True)
        self.prompts = {}

    @torch.inference_mode()
    def predict(self, request):
        from owac.training.compact_data import normalize_state, unnormalize_actions

        prompt = request["prompt"]
        if prompt not in self.prompts:
            text = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.cfg["model"]["language_length"],
                return_tensors="pt",
            )
            self.prompts[prompt] = (text["input_ids"].cuda(), text["attention_mask"].cuda().bool())
        tokens, mask = self.prompts[prompt]
        batch = {
            "raw": torch.as_tensor(request["raw"], device="cuda").permute(0, 3, 1, 2)[None].float()
            / 255,
            "state": torch.as_tensor(
                normalize_state(np.asarray(request["proprio"], np.float32), self.cfg["statistics"]),
                device="cuda",
                dtype=torch.float32,
            )[None],
            "tokens": tokens,
            "token_mask": mask,
        }
        generator = torch.Generator(device="cuda").manual_seed(request["seed"])
        noise = torch.randn(
            (1, self.cfg["model"]["action_horizon"], 7),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = self.model.sample(batch, noise)[0].float().cpu().numpy()
        return np.clip(unnormalize_actions(normalized, self.cfg["statistics"]), -1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads(args.config.read_text())
    item = cfg["policies"][args.policy]
    if item["kind"] == "compact":
        policy = CompactRawPolicy(item)
    else:
        from owac.runtime.raw_pi05_policy import RawPi05Policy

        policy = RawPi05Policy(item)
    listener = Listener(("127.0.0.1", args.port), authkey=b"owac-native-rgb-benchmark")
    pending = queue.Queue(maxsize=32)

    def connection_loop(connection):
        try:
            while True:
                request = connection.recv()
                answer = queue.Queue(maxsize=1)
                pending.put((request, answer))
                connection.send(answer.get())
        except (EOFError, ConnectionError):
            pass
        finally:
            connection.close()

    def accept_loop():
        while True:
            connection = listener.accept()
            threading.Thread(target=connection_loop, args=(connection,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    identity = {
        "checkpoint_sha256": getattr(policy, "checkpoint_sha256", None),
        "checkpoint_step": getattr(policy, "checkpoint_step", None),
    }
    args.ready.write_text(json.dumps({"port": args.port, "policy": args.policy, **identity}))
    started = time.monotonic()
    count = 0
    while True:
        request, answer = pending.get()
        try:
            actions = policy.predict(request)
            answer.put({"actions": actions.tolist(), "dtype": actions.dtype.str, **identity})
        except Exception:
            answer.put({"error": traceback.format_exc()})
            raise
        count += 1
        if count % 100 == 0:
            print(
                json.dumps(
                    {
                        "requests": count,
                        "requests_per_second": count / (time.monotonic() - started),
                        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
