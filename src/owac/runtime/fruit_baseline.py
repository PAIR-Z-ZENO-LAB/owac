"""Apply the trained CoRP and π0.5 fruit policies to recorded or live U2/U3 measurements."""

from pathlib import Path

import numpy as np
import torch

from owac.raw.fruit_isp import IspDino, corp_rgb, linear_planes
from owac.training.fruit_baseline_models import load_export, make_model, rawvla_offsets


class FruitBaseline:
    """Return NERO joint targets in radians and signed gripper aperture in metres."""

    def __init__(self, checkpoint, device="cuda", assets=None, *, method=None, step=None):
        ck = torch.load(checkpoint, weights_only=False, map_location="cpu", mmap=True)
        self.cfg, self.method = ck["config"], ck["method"]
        if method is not None and (method != self.method or step != ck["step"]):
            raise ValueError(f"权重与目录不一致 {self.method} / {ck['step']}")
        if (self.cfg["model"]["action_horizon"], self.cfg["observation"]["frequency_hz"]) != (
            12,
            30,
        ):
            raise ValueError("当前推理台要求 30 Hz、12 步的动作块")
        self.cfg["assets"].update(assets or {})
        self.device = torch.device(device)
        self.policy = make_model(self.cfg, self.method)
        load_export(self.policy, ck["weights"])
        self.policy.to(self.device).eval()
        self.encoder = None
        if self.method == "dino-corp":
            self.encoder = IspDino(Path(self.cfg["assets"]["dino"])).to(self.device).eval()
        self.stats = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in self.cfg["statistics"].items()
        }
        self.relative_mask = torch.tensor(
            self.cfg["observation"]["relative_action_mask"], device=self.device
        )
        self.offsets = rawvla_offsets(self.cfg["pi05"]["recurrent_stride"])
        self.history_size = 1 - int(self.offsets.min()) if self.method == "rawvla" else 1
        self.prepared = {}

    def prepare_pair(self, raw):
        if self.encoder is not None:
            return np.stack([corp_rgb(im, self.cfg["model"]["image_size"]) for im in raw])
        return np.stack([linear_planes(im, self.cfg["pi05"]["linear_shape"]) for im in raw])

    def predict(self, history, state, *, seed):
        """Accept [T,2,H,W] unpacked Bayer pairs in chronological order."""
        raw = np.asarray(history)
        expected = (2, *self.cfg["observation"]["raw_shape"])
        if raw.ndim != 4 or raw.shape[1:] != expected or not len(raw):
            raise ValueError(f"Expected nonempty RAW history [T,{expected}]")
        if not np.issubdtype(raw.dtype, np.integer) or raw.min() < 0 or raw.max() > 1023:
            raise ValueError("Expected unpacked 10-bit Bayer measurements")
        return self.predict_prepared(
            [self.prepare_pair(pair) for pair in raw[-self.history_size :]], state, seed
        )

    def predict_live(self, observations, unpack):
        """Reuse sensor preprocessing for overlapping causal histories across GPU predictions."""
        selected = observations[-self.history_size :]
        prepared = {}
        for obs in selected:
            pair = self.prepared.get(obs.capture)
            if pair is None:
                raw = [unpack(f.data, f.width, f.height, f.stride) for f in obs.frames]
                pair = self.prepare_pair(raw)
            prepared[obs.capture] = pair
        self.prepared = prepared
        current = selected[-1]
        seed = self.cfg["seed"] + int(current.frames[0].sequence)
        return self.predict_prepared(list(prepared.values()), current.feedback.values, seed)

    @torch.inference_mode()
    def predict_prepared(self, pairs, state, seed):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32).reshape(1, -1)
        if state.shape[1] != self.cfg["model"]["state_dim"]:
            raise ValueError("Expected seven joints and a gripper")
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            if self.encoder is not None:
                _, patches = self.encoder.encode_rgb(pairs[-1])
                images = patches.unsqueeze(0).half()
            else:
                if self.method == "rawvla":
                    indices = np.maximum(len(pairs) - 1 + self.offsets, 0)
                    images = torch.as_tensor(np.stack(pairs), device=self.device)[
                        torch.as_tensor(indices, device=self.device)
                    ].permute(2, 0, 1, 3, 4, 5)
                else:
                    images = torch.as_tensor(pairs[-1], device=self.device)
                images = images.unsqueeze(0)
            stats = self.stats
            low, high = stats["state_q01"], stats["state_q99"]
            obs = dict(
                images=images,
                state=(state - stats["state_mean"]) / (stats["state_std"] + 1e-6),
                state_q=(2 * (state - low) / (high - low + 1e-6) - 1).clamp(-1, 1),
            )
            noise = torch.randn(
                (1, self.cfg["model"]["action_horizon"], self.policy.noise_dim),
                device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            norm = self.policy.sample(obs, noise).float()
        rel = (norm + 1) * (stats["action_q99"] - stats["action_q01"] + 1e-6) / 2 + stats[
            "action_q01"
        ]
        absolute = rel + (state * self.relative_mask)[:, None]
        return absolute[0].cpu().numpy()
