"""Export and load the selected OWAC EMA weights without the training environment."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from owac.paper.model import CompactPolicy

TOKENIZER = Path(__file__).with_name("tokenizer")
FORMAT = "owac-libero-policy-v1"


def model_config(config):
    return {key: config[key] for key in ("model", "statistics", "task_names", "training")}


def export(checkpoint, output, *, trusted_training_state=False):
    """Convert an OWAC training checkpoint or a selected EMA export."""
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=not trusted_training_state, mmap=True
    )
    if "ema" in payload:
        weights = payload["ema"]
    elif set(payload) == {"model", "step", "config"}:
        # compact_train writes its EMA as model in the standalone policy.pt.
        weights = payload["model"]
    else:
        raise ValueError("Expected a training state or a standalone EMA policy")
    config = model_config(payload["config"])
    metadata = {
        "format": FORMAT,
        "config": json.dumps(config),
        "step": str(payload["step"]),
        "license": "CC-BY-NC-4.0",
    }
    # Safetensors stores independent contiguous tensors, including tied T5 embeddings.
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in weights.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=metadata)


def load_policy(path, device="cpu"):
    with safe_open(path, framework="pt", device="cpu") as file:
        meta = file.metadata()
        if not meta or meta.get("format") != FORMAT:
            raise ValueError("Expected an OWAC LIBERO policy exported with owac.paper.checkpoint")
        cfg = json.loads(meta["config"])
        model = CompactPolicy(cfg)
        model.load_state_dict({key: file.get_tensor(key) for key in file.keys()}, strict=True)
    return model.to(device).eval(), cfg, int(meta["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--trusted-training-state",
        action="store_true",
        help="enable pickle only for a full training state you created",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, trusted_training_state=args.trusted_training_state)


if __name__ == "__main__":
    main()
