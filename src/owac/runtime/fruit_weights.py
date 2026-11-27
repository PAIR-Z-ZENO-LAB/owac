"""Catalog of original and additional evaluated fruit checkpoints on the lab host."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from owac.runtime.fruit_live import HZ, STEPS


@dataclass(frozen=True)
class Weight:
    label: str
    step: int
    checkpoint: Path
    rtc: Path | None
    backend: str = "owac"
    assets: Path | None = None
    execution_horizon: int = 12
    postprocess: Path | None = None


def read_manifest(root):
    file = root / "candidates" / "catalog.json"
    return json.loads(file.read_text()) if file.is_file() else {}


def catalog(root, manifest=None):
    candidates = root / "candidates"
    if manifest is None:
        manifest = read_manifest(root)
    labels = {"best": "原版 · 2,500 步", "final": "原版 · 10,000 步"}
    choices = {
        key: Weight(
            labels[key], step, root / "checkpoints" / f"policy-{key}.pt", root / f"rtc-{key}.json"
        )
        for key, step in STEPS.items()
        if manifest.get("include_originals", True)
    }
    for row in manifest.get("candidates", []):
        key = row["id"]
        if key in choices or Path(key).name != key or key in ("", ".", ".."):
            raise ValueError(f"权重目录中的标识无效 {key}")
        folder = candidates / key
        backend = row.get("backend", "owac")
        if backend not in ("owac", "dino-corp", "rawvla", "rawild"):
            raise ValueError(f"未知的推理后端 {backend}")
        checkpoint = root / row["checkpoint"] if "checkpoint" in row else folder / "policy.pt"
        rtc = folder / "rtc.json" if backend == "owac" and row.get("rtc", True) else None
        assets = root / row["assets"] if backend != "owac" else None
        count = row.get("execution_horizon", 12)
        if type(count) is not int or not 1 <= count <= 12:
            raise ValueError(f"权重 {key} 的连续执行步数应在 1 到 12 之间")
        profile = root / row["postprocess"] if "postprocess" in row else None
        choices[key] = Weight(
            row["label"], row["step"], checkpoint, rtc, backend, assets, count, profile
        )
    return choices


def default_policy(choices, *, check, grasp, place):
    if grasp == "original":
        return "best"
    if len(choices) == 1:
        return next(iter(choices))
    if place:
        return "fruit-final" if "fruit-final" in choices else "basket-place"
    if check and not grasp:
        return None
    if "fruit-final" in choices:
        return "fruit-final"
    if "banana-occlusion-close" in choices:
        return "banana-occlusion-close"
    return next(iter(choices))


class FruitModels(dict):
    """Only the selected additional model shares GPU memory with the two quick choices."""

    def __init__(self, choices, adapters):
        super().__init__()
        self.choices, self.adapters = choices, adapters

    def __missing__(self, key):
        import torch

        from owac.frameworks.raw_compact.factory import make_policy
        from owac.runtime.fruit_rtc import PrefixRegression

        choice = self.choices[key]
        for previous in tuple(self):
            if previous not in STEPS:
                del self[previous]
        if choice.backend != "owac":
            folder = choice.assets
            vendor = folder / "author"
            for path in (
                folder / "deps",
                vendor / "CoRP",
                vendor / "RawVLA",
                vendor / "RawVLA/starVLA",
            ):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            from owac.runtime.fruit_baseline import FruitBaseline

            assets = dict(
                pi05=str(folder / "pi05/model.safetensors"),
                dino=str(folder / "dino"),
                corp=str(vendor / "CoRP/checkpoints/corp_libero.pt"),
                isp_run=str(folder / "isp"),
                tokenizer=str(folder / "isp/checkpoints/paligemma_tokenizer.model"),
            )
            self[key] = FruitBaseline(
                choice.checkpoint, assets=assets, method=choice.backend, step=choice.step
            )
            return self[key]
        prefix = (
            PrefixRegression.load(choice.rtc, choice.checkpoint, choice.step)
            if self.adapters is not None and choice.rtc is not None
            else None
        )
        ck = torch.load(choice.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if ck["step"] != choice.step or ck["config"]["observation"]["frequency_hz"] != HZ:
            raise ValueError(f"权重步数或动作频率异常 {key}")
        model = make_policy(ck["config"])
        model.load_state_dict(ck["model"], strict=True)
        model.return_live_scene = choice.postprocess is not None
        self[key] = model.cuda().eval()
        if prefix is not None:
            self.adapters[key] = prefix
        return self[key]
