"""Small native-RAW model configurations for offline test fixtures."""

import torch


def native_sensor(layout="mono", **extra):
    return dict(
        layout=layout,
        units="dn",
        bit_depth=10,
        black_level=0,
        white_level=1023,
        shot=0,
        read_var=0,
        **extra,
    )


def measurement(sensor=None, **extra):
    return {
        "sensor": native_sensor() if sensor is None else sensor,
        "scales": [4, 8, 16],
        "response_pivot": 0.02,
        **extra,
    }


def example(domain):
    views, state_dim, action_dim, height, width = (
        (["agentview", "eye_in_hand"], 9, 7, 32, 48)
        if domain == "libero"
        else (["head", "left", "right"], 34, 14, 48, 64)
    )
    cfg = {
        "observation": {"domain": domain, "views": dict.fromkeys(views, "unused")},
        "model": {
            "kind": "corp",
            "visual_initialization": "random",
            "raw_encoder": dict(
                width=32,
                depth=2,
                heads=4,
                patch_size=16,
                image_size=32,
                drop_path=0.0,
                response_pivot=0.02,
            ),
            "language": dict(
                vocab_size=24,
                d_model=32,
                d_ff=64,
                d_kv=8,
                num_heads=4,
                num_layers=1,
                dropout_rate=0.0,
                pad_token_id=0,
                eos_token_id=1,
            ),
            "hidden_dim": 48,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_horizon": 3,
            "tokens_per_view": 2,
            "resampler_heads": 4,
            "film_rank": 8,
            "spatial_coordinates": True,
            "action_generator": dict(
                num_layers=1,
                num_heads=4,
                mlp_ratio=2,
                max_horizon=8,
                adaln_rank=8,
                num_sample_steps=2,
            ),
        },
        "training": {"task_loss_weight": 0.01},
        "task_names": ["move object", "open drawer"],
    }
    batch = {
        "raw": torch.rand(2, len(views), 3, height, width),
        "state": torch.randn(2, state_dim),
        "tokens": torch.tensor([[2, 3, 1, 0], [3, 2, 1, 0]]),
        "token_mask": torch.tensor([[True, True, True, False]]).expand(2, -1),
        "actions": torch.randn(2, 3, action_dim),
        "action_valid": torch.tensor([[True, True, True], [True, True, False]]),
        "task_id": torch.tensor([0, 1]),
    }
    return cfg, batch
