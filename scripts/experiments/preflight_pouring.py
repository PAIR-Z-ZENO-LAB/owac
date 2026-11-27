"""Check the scratch action path without an optimizer step or any robot I/O."""

import hashlib
import sys
from pathlib import Path

import torch
from prepare_pouring import read

from owac.training.fruit_baseline_models import make_model, weights


def main(method):
    from prepare_pouring import make_configs, write

    source = Path(__file__).resolve().parents[2]
    templates = source / "configs/experiments"
    spec = read(templates / "real_pouring_20261123.json")
    _, _, cfg = make_configs(spec, templates, "synthetic-preflight-only", "not-used")
    torch.manual_seed(spec["seed"])
    torch.set_num_threads(2)
    model = make_model(cfg, method).cuda().train()
    parameters = weights(model)
    if method == "dino-corp":
        shape = (2, 2, 196, 768)
        probe = model.generator.action_projection.weight
    else:
        if any(name.startswith("policy.vlm.") or "lora" in name.lower() for name in parameters):
            raise ValueError("Expected frozen VLM and fresh full action expert")
        head = model.policy.action_head
        probe = head.action_in_proj.weight
        if (
            not probe.requires_grad
            or not head.gemma_expert.model.layers[0].self_attn.q_proj.weight.requires_grad
        ):
            raise ValueError("Fresh action expert is not trainable")
        shape = (2, 2, 6, 6, 3, 168, 224) if method == "rawvla" else (2, 2, 3, 168, 224)
    probe_hash = hashlib.sha256(probe.detach().float().cpu().numpy().tobytes()).hexdigest()
    batch = dict(
        images=torch.rand(shape, device="cuda"),
        state=torch.randn(2, 8, device="cuda"),
        state_q=torch.zeros(2, 8, device="cuda"),
        actions=torch.randn(2, 12, 8, device="cuda"),
        valid=torch.ones(2, 12, dtype=torch.bool, device="cuda"),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model.loss(batch)
    loss.backward()
    grad = float(probe.grad.detach().float().norm())
    if not torch.isfinite(loss).item() or not 0 < grad < float("inf"):
        raise ValueError("Non-finite or missing scratch policy gradient")
    result = dict(
        method=method,
        optimizer_updates=0,
        input="synthetic",
        loss=float(loss.detach()),
        action_gradient=grad,
        action_probe_sha256=probe_hash,
        trainable=sum(param.numel() for param in parameters.values()),
        memory_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    write(Path(spec["root"]) / f"receipts/preflight-{method}.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
