"""Explicit owned RAW encoder loading shared by visual learning and policies."""

import torch

from owac.training.compact_observation import observation_spec


def initialize_visual(encoder, cfg):
    """Load J5, an owned EMA policy encoder, or an owned standalone encoder."""
    kind = cfg["model"].get("visual_initialization", "j5")
    assets = {
        "j5": "j5",
        "owned_policy_encoder": "visual_policy",
        "owned_raw_encoder": "visual_encoder",
    }
    if kind not in assets:
        raise ValueError(f"Unknown RAW visual initialization: {kind}")
    path = cfg["assets"][assets[kind]]
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if kind == "j5":
        weights = payload["encoder"]
    else:
        saved = payload["config"]
        domains = [observation_spec(saved)["domain"]]
        if kind == "owned_raw_encoder" and "observation" not in saved:
            domains = saved.get("data", {}).get("domains", domains)
        if observation_spec(cfg)["domain"] not in domains:
            raise ValueError("Owned encoder must come from the same platform")
        if kind == "owned_raw_encoder":
            weights = payload["encoder"]
        else:
            weights = {
                key.removeprefix("raw_encoder."): value
                for key, value in payload["ema"].items()
                if key.startswith("raw_encoder.")
            }
    from owac.training.raw_sensel import migrate_owned_rgb_projection

    weights = dict(weights)
    migration = migrate_owned_rgb_projection(encoder, weights)
    if migration is not None:
        initialized = encoder.state_dict()
        initialized.update(weights)
        weights = initialized
    encoder.load_state_dict(weights)
    return {
        "visual_checkpoint": str(path),
        "visual_source_step": payload.get("step"),
        "measurement_projection": migration,
    }
