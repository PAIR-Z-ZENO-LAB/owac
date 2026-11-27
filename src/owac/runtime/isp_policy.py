"""Released RGB policies with the public differentiable ISP frontends."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ISPPolicy(nn.Module):
    def __init__(self, cfg, policy, frontend, device="cuda:0", checkpoint=None):
        super().__init__()
        from omegaconf import OmegaConf
        from starVLA.model.modules.raw_frontend.registry import build_raw_frontend

        self.cfg, self.kind, self.frontend_name = cfg, policy, frontend
        self.device = torch.device(device)
        self.frontend_cfg = dict(cfg["rawvla_frontend"])
        self.frontend_cfg["name"] = frontend
        if frontend == "rawild":
            self.frontend_cfg = {"name": "rawild", "input_format": "rgb_raw"}
        self.frontend = None if frontend == "default" else build_raw_frontend(self.frontend_cfg)
        if policy == "corp":
            from corp import load_policy

            self.policy = load_policy(cfg["corp_checkpoint"], device=device)
            self.horizon, self.dim = 12, 7
        else:
            from starVLA.model.framework.VLM4A.PI05 import PI05

            model_cfg = OmegaConf.create(
                {
                    "framework": {
                        "name": "PI05",
                        "precision": "bfloat16",
                        "action_horizon": 10,
                        "action_dim": 32,
                        "discrete_state_input": False,
                        "max_token_len": 200,
                        "tokenizer": {"model_path": cfg["tokenizer"], "pad_state_value": -2.0},
                        "raw_frontend": {"name": "none"},
                    }
                }
            )
            self.policy = PI05(model_cfg)
            weights = self.policy._normalize_state_dict(
                self.policy._load_checkpoint_state_dict(cfg["pi05_checkpoint"])
            )
            result = self.policy.load_state_dict(weights, strict=False)
            missing = [
                k for k in result.missing_keys if not self.policy._is_expected_openpi_missing_key(k)
            ]
            if missing or result.unexpected_keys:
                raise ValueError(
                    f"Incomplete public policy load: {missing}, {result.unexpected_keys}"
                )
            del weights
            self.policy.to(device)
            stats = json.loads(Path(cfg["pi05_statistics"]).read_text())
            self.stats = stats.get("norm_stats", stats)
            self.stats = {
                key: {name: np.asarray(value, np.float32) for name, value in item.items()}
                for key, item in self.stats.items()
            }
            self.horizon, self.dim = 10, 32
        self.policy.eval().requires_grad_(False)
        if self.frontend is not None:
            self.frontend.to(device)
            if checkpoint:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved["policy"] != policy or saved["frontend"] != frontend:
                    raise ValueError("ISP checkpoint belongs to another policy/frontend")
                self.frontend.load_state_dict(saved["model"], strict=True)

    def render(self, samples, temporal_index=-1):
        # Input is uint8 [views, recurrent-observations, burst, H, W, RGB].
        # Fixed ISP images are formed by the benchmark directly at the sensor boundary.
        if self.frontend is None:
            images = [s["rgb"] for s in samples]
            if self.kind == "corp":
                from PIL import Image

                images = [
                    [
                        np.asarray(
                            Image.fromarray(view).resize((224, 224), Image.Resampling.BILINEAR)
                        )
                        for view in sample
                    ]
                    for sample in images
                ]
            return (
                torch.as_tensor(np.stack(images), device=self.device).permute(0, 1, 4, 2, 3).float()
                / 255.0
            )
        raw = torch.as_tensor(np.stack([s["raw"] for s in samples]), device=self.device)
        raw = raw.permute(0, 1, 2, 3, 6, 4, 5).float().div_(255.0)
        shape = raw.shape
        if self.frontend_name == "rawvla":
            rendered = self.frontend(raw.flatten(0, 1), input_format="rgb_raw")
        else:
            rendered = self.frontend(
                raw[:, :, temporal_index, -1].flatten(0, 1), input_format="rgb_raw"
            )
        return rendered.reshape(shape[0], shape[1], *rendered.shape[1:])

    def _corp_batch(self, rgb, states, prompts):
        from corp.preprocessing import prepare_observation

        # Float bilinear resizing preserves gradients; quantization is only native RGB input.
        rgb = F.interpolate(
            rgb.flatten(0, 1), (224, 224), mode="bilinear", align_corners=False, antialias=True
        ).reshape(len(states), 2, 3, 224, 224)
        states = np.asarray(states, np.float32).copy()
        centers = self.policy.normalizer.state_angle_center.numpy()
        for index in range(3, 6):
            states[:, index] = (
                centers[index] + (states[:, index] - centers[index] + np.pi) % (2 * np.pi) - np.pi
            )
        return prepare_observation(
            {
                "images": dict(zip(self.policy.config["view_names"], rgb.unbind(1), strict=True)),
                "state": states,
                "instruction": prompts,
            },
            config=self.policy.config,
            normalizer=self.policy.normalizer,
            device=self.device,
        )

    def _pi_observation(self, rgb, states, prompts):
        from pi_libero_common import normalize_openpi_value

        examples = [
            {
                "image": list(images),
                "lang": prompt,
                "state": normalize_openpi_value(np.asarray(state), self.stats["state"], True),
            }
            for images, state, prompt in zip(rgb, states, prompts, strict=True)
        ]
        return self.policy._build_observation_from_batch(
            self.policy._prepare_examples(examples, include_actions=False)
        )

    def loss(self, samples, temporal_index=0):
        rgb = self.render(samples, temporal_index)
        states = np.stack([s["state"][temporal_index] for s in samples])
        actions = np.stack([s["actions"][temporal_index] for s in samples])
        prompts = [s["prompt"] for s in samples]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "corp":
                representation = self.policy.representation_extractor(
                    **self._corp_batch(rgb, states, prompts)
                )
                target = torch.as_tensor(actions, device=self.device).clone()
                low, high = (
                    self.policy.normalizer.action_min.to(self.device),
                    self.policy.normalizer.action_max.to(self.device),
                )
                target[..., :6] = 2 * (target[..., :6] - low[:6]) / (high[:6] - low[:6]) - 1
                noise = torch.randn_like(target)
                tau = torch.rand(len(samples), device=self.device)
                prediction = self.policy.action_generator(
                    noise * (1 - tau[:, None, None]) + target * tau[:, None, None],
                    tau,
                    self.policy.action_generator.prepare_prefix(representation),
                )
                action_loss = F.mse_loss(prediction.float(), target - noise)
            else:
                low, high = [np.asarray(self.stats["actions"][k][:7]) for k in ("q01", "q99")]
                target = np.clip(2 * (actions - low) / np.maximum(high - low, 1e-6) - 1, -1, 1)
                target = torch.as_tensor(
                    np.pad(target, ((0, 0), (0, 0), (0, 25))),
                    device=self.device,
                    dtype=torch.float32,
                )
                action_loss = self.policy.forward_from_observation(
                    self._pi_observation(rgb, states, prompts), target
                )["action_loss"]
        losses = {"action": action_loss}
        total = action_loss
        if self.frontend_name == "rawvla":
            auxiliary = self.frontend.last_aux_losses
            losses.update(
                exposure=auxiliary["exposure_prior_loss"], chroma=auxiliary["chroma_prior_loss"]
            )
            total = 0.15 * action_loss + 0.01 * losses["exposure"] + 0.001 * losses["chroma"]
        return total, losses

    @torch.inference_mode()
    def predict(self, samples):
        rgb = self.render(samples)
        states, prompts = [s["state"] for s in samples], [s["prompt"] for s in samples]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "corp":
                representation = self.policy.representation_extractor(
                    **self._corp_batch(rgb, states, prompts)
                )
                action = torch.stack(
                    [
                        torch.randn(
                            (self.horizon, self.dim),
                            generator=torch.Generator(device=self.device).manual_seed(
                                int(s["seed"])
                            ),
                            device=self.device,
                            dtype=representation.action_dtype,
                        )
                        for s in samples
                    ]
                )
                prefix = self.policy.action_generator.prepare_prefix(representation)
                for step in range(10):
                    tau = torch.full(
                        (len(samples),), step / 10, device=self.device, dtype=action.dtype
                    )
                    action = action + self.policy.action_generator(action, tau, prefix) / 10
                action = self.policy.normalizer.unnormalize_action(action).float().cpu().numpy()
            else:
                from pi_libero_common import postprocess_openpi_actions

                noise = torch.stack(
                    [
                        torch.randn(
                            (self.horizon, self.dim),
                            generator=torch.Generator(device=self.device).manual_seed(
                                int(s["seed"])
                            ),
                            device=self.device,
                        )
                        for s in samples
                    ]
                )
                normalized = (
                    self.policy.sample_actions(
                        self._pi_observation(rgb, states, prompts), noise=noise, num_steps=10
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                action = np.stack(
                    [
                        postprocess_openpi_actions(
                            normalized_actions=a,
                            raw_state=np.asarray(s),
                            norm_stats=self.stats,
                            model_name="PI05",
                            model_source="openpi",
                        )
                        for a, s in zip(normalized, states, strict=True)
                    ]
                )
        return np.clip(action, -1, 1)
