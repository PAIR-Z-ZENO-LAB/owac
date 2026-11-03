"""Serve explicit checkpoint-module swaps for causal native rollouts."""

import argparse
from pathlib import Path

import torch
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from owac.evaluation.j5_regression_probe import compact_policy
from owac.evaluation.j5_weight_audit import group
from owac.training.j5_lora import copy_trainable

MODES = {
    "before": set(),
    "after": {"vision", "bridge", "language", "expert"},
    "vision": {"vision"},
    "bridge": {"bridge"},
    "language": {"language"},
    "expert": {"expert"},
    "without_expert": {"vision", "bridge", "language"},
    "without_language": {"vision", "bridge", "expert"},
    "without_vision": {"bridge", "language", "expert"},
    "without_bridge": {"vision", "language", "expert"},
}


class SwapPolicy:
    def __init__(self, before, after, device):
        self.policy = compact_policy(after, device)
        self.old = torch.load(before, map_location="cpu", mmap=True, weights_only=False)["ema"]
        self.new = torch.load(after, map_location="cpu", mmap=True, weights_only=False)["ema"]
        self.variant = "after"

    def infer(self, observation):
        variant = observation["variant"]
        if variant != self.variant:
            if variant == "half":
                state = {n: (self.old[n] + self.new[n]) * 0.5 for n in self.new}
            else:
                state = {
                    n: self.new[n] if group(n) in MODES[variant] else self.old[n] for n in self.new
                }
            copy_trainable(
                self.policy.model,
                state,
            )
            self.variant = variant
        return self.policy.infer(observation)

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--port", type=int, default=9026)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = SwapPolicy(args.before, args.after, args.device)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
