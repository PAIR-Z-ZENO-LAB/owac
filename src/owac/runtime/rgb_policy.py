"""Transport adapters around the four released, unchanged RGB inference APIs."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NativeRGBPolicy:
    def __init__(self, cfg, kind):
        self.kind = kind
        item = cfg["policies"][kind]
        if kind == "corp":
            from corp import load_policy

            self.policy = load_policy(item["checkpoint"], device="cuda:0")
            self.native = source_module(
                "native_corp_eval", Path(item["source"]) / "scripts/evaluate_libero.py"
            )
        elif kind == "pi05":
            from openpi.policies import policy_config
            from openpi.training import config

            self.native = source_module(
                "native_openpi_eval", Path(item["source"]) / "examples/libero/main.py"
            )
            train_config = config.get_config("pi05_libero")
            # Official execution option: avoid Torch 2.12's invalid SDPA rewrite
            # of the author's eager BF16 attention with its FP32 attention mask.
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(
                    train_config.model, pytorch_compile_mode=item["pytorch_compile_mode"]
                ),
            )
            self.policy = policy_config.create_trained_policy(
                train_config, item["checkpoint"], pytorch_device="cuda:0"
            )
        elif kind == "openvla":
            import tensorflow as tf
            from experiments.robot.libero import run_libero_eval

            tf.config.set_visible_devices([], "GPU")
            self.native = run_libero_eval
            self.options = self.native.GenerateConfig(
                pretrained_checkpoint=item["checkpoint"], task_suite_name="libero_spatial"
            )
            (
                self.policy,
                self.head,
                self.proprio,
                self.noise_projector,
                self.processor,
            ) = self.native.initialize_model(self.options)
        elif kind == "xvla":
            from models.modeling_xvla import XVLA
            from models.processing_xvla import XVLAProcessor

            processor = XVLAProcessor.from_pretrained(item["checkpoint"])
            self.policy = XVLA.from_pretrained(
                item["checkpoint"], trust_remote_code=True, torch_dtype=torch.float32
            ).to("cuda:0", dtype=torch.float32)
            self.policy._build_app(processor)
            self.endpoint = next(r.endpoint for r in self.policy.app.routes if r.path == "/act")
        else:
            raise ValueError(kind)

    @torch.inference_mode()
    def predict(self, request):
        # A request-local generator makes replicas/resumption independent of queue ordering.
        torch.manual_seed(request["seed"])
        if self.kind == "xvla":
            response = self.endpoint(request["payload"])
            if response.status_code != 200:
                raise RuntimeError(response.body.decode())
            # The official HTTP client reconstructs JSON numbers as float64
            # before converting the 6D rotations. Preserve that boundary.
            return np.asarray(json.loads(response.body)["action"])
        obs, prompt = request["observation"], request["prompt"]
        if self.kind == "corp":
            batch = self.native.policy_observation(obs, prompt, self.policy)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actions = self.policy.predict(batch)[0].float().cpu().numpy()
            return np.clip(actions, -1.0, 1.0)
        if self.kind == "pi05":
            from openpi_client import image_tools

            images = [
                image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(np.ascontiguousarray(obs[k][::-1, ::-1]), 224, 224)
                )
                for k in ("agentview_image", "robot0_eye_in_hand_image")
            ]
            element = {
                "observation/image": images[0],
                "observation/wrist_image": images[1],
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        self.native._quat2axisangle(obs["robot0_eef_quat"].copy()),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": str(prompt),
            }
            return self.policy.infer(element)["actions"][:5]
        self.options.task_suite_name = request["suite"]
        self.native.check_unnorm_key(self.options, self.policy)
        prepared, _ = self.native.prepare_observation(obs, 224)
        actions = self.native.get_action(
            self.options,
            self.policy,
            prepared,
            prompt,
            processor=self.processor,
            action_head=self.head,
            proprio_projector=self.proprio,
            noisy_action_projector=self.noise_projector,
            use_film=self.options.use_film,
        )
        return np.asarray([self.native.process_action(a, "openvla") for a in actions])
