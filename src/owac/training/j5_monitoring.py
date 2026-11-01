"""One shared action-training metric vocabulary and four non-overlapping sections."""

import json
from pathlib import Path

RENAMES = {
    "train/flow32": "objective/train_flow32",
    "train/flow7": "objective/train_flow7",
    "train/grad_norm": "optimizer/grad_norm",
    "train/validation_lr_factor": "optimizer/lr_factor",
    "train/rollback_to_step": "optimizer/rollback_to_step",
    "train/seconds_per_update": "runtime/update_seconds",
    "train/images_per_second": "runtime/examples_per_second",
    "host/memory_gib": "runtime/host_ram_gib",
    "host/anonymous_gib": "runtime/host_anonymous_gib",
    "host/file_cache_gib": "runtime/host_file_cache_gib",
    "host/shared_gib": "runtime/host_shared_gib",
    "host/oom_kills": "runtime/oom_kills",
    "gpu/peak_allocated_gib": "runtime/gpu_peak_allocated_gib",
    **{
        f"train/lr_{name}": f"optimizer/lr_{name}"
        for name in ["bridge", "language", "expert_content", "expert_qk"]
    },
}


def canonical(metrics):
    # train/lr duplicates lr_bridge in this four-group policy.
    return {RENAMES.get(key, key): value for key, value in metrics.items() if key != "train/lr"}


def setup(swanlab, cfg, root):
    for prefix, section in [
        ("objective", "01 训练目标"),
        ("validation", "02 动作验证"),
        ("optimizer", "03 优化器"),
        ("runtime", "04 运行资源"),
    ]:
        swanlab.define_metric(f"{prefix}/*", section_name=section)
    marker = root / f"monitor-v2-history-{cfg['swanlab']['id']}.json"
    history = cfg.get("monitoring", {}).get("history")
    if history and not marker.exists():
        steps = []
        for line in Path(history).read_text().splitlines():
            record = json.loads(line)
            # Preserve measured training history; never relabel historical MAE as MSE.
            metrics = {
                k: v
                for k, v in record["metrics"].items()
                if k in RENAMES
                or k.startswith(("objective/", "validation/", "optimizer/", "runtime/"))
            }
            if metrics:
                swanlab.log(canonical(metrics), step=record["step"])
                steps.append(record["step"])
        marker.write_text(json.dumps({"replayed_steps": steps, "source": history}))
