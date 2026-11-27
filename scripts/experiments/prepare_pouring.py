"""Bind the four pouring jobs to the verified Hub revision and a common episode split."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_split(spec):
    return {
        f"{subset}/episode_{index:03d}": (
            "validation" if index in spec["validation_indices"][subset] else "train"
        )
        for subset, count in spec["episodes"].items()
        for index in range(1, count + 1)
    }


def align_one(args):
    import numpy as np

    from owac.raw.all_bench_distillation import read_npz_member
    from owac.raw.teleop_mcap import CAMERAS, prepare_group

    episode, output, split = args
    metadata = read(episode / "metadata.json")
    if not metadata.get("ended_at_utc") or not metadata["rosbag_info_ok"]:
        raise ValueError(f"Unfinalized recording: {episode}")
    row = prepare_group(episode, output, {episode.name} if split == "validation" else set())
    size = row["num_frames"]
    path = Path(row["path"])
    with np.load(path) as saved:
        times = saved["observation_ns"]
        if size < 12 or len(times) != size or not (np.diff(times) > 0).all():
            raise ValueError(f"Invalid observation clock: {episode}")
        if not ((saved["state_ns"] <= times) & (saved["state_log_ns"] <= times)).all():
            raise ValueError(f"Noncausal state: {episode}")
        if not (saved["action_ns"] >= times).all():
            raise ValueError(f"Inconsistent next-command labels: {episode}")
        for field in ("state", "action"):
            array = saved[field]
            if array.shape != (size, 8) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field}: {episode}")
    for camera in CAMERAS:
        raw = read_npz_member(path, camera + ".npy")
        if raw.shape != (size, 480, 640) or raw.dtype != np.dtype("<u2") or raw.max() > 1023:
            raise ValueError(f"Unexpected RAW10 source: {episode} {camera}")
    print(row["group_id"], split, size, flush=True)
    return row


def prepare_data(spec):
    from owac.raw.teleop_mcap import write_inventory

    root, raw = Path(spec["root"]), Path(spec["raw"])
    data = root / "aligned"
    data.mkdir(exist_ok=True)
    receipt = read(root / "receipts/hub.json")
    split = episode_split(spec)
    if not (
        receipt["private"] is True
        and receipt["repo"] == spec["repo"]
        and set(receipt["episodes"]) == set(split)
        and receipt["manifest_sha256"] == digest(root / "receipts/lab.sha256")
    ):
        raise ValueError("Private Hub receipt, recorder, and declared protocol must agree")
    jobs = [(raw / episode, data, partition) for episode, partition in split.items()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(align_one, jobs))
    if {row["group_id"]: row["split"] for row in groups} != split:
        raise ValueError("Aligned split changed")
    write_inventory(data, groups)
    counts = Counter()
    for row in groups:
        counts[f"{row['split']}_pairs"] += row["num_frames"]
        counts[f"{row['split']}_episodes"] += 1
    write(
        root / "receipts/split.json",
        dict(
            revision=receipt["revision"],
            counts=dict(counts),
            episodes={row["group_id"]: [row["split"], row["num_frames"]] for row in groups},
            inventory_sha256=digest(data / "inventory.json"),
        ),
    )


def make_configs(spec, templates, revision, inventory_sha):
    root = Path(spec["root"])
    data = dict(
        inventory=str(root / "aligned/inventory.json"),
        dataset_revision=revision,
        inventory_sha256=inventory_sha,
    )
    valid = [name for name, partition in episode_split(spec).items() if partition == "validation"]
    group = spec["swanlab"]
    updates, batch = spec["action_updates"], spec["action_batch"]
    anchors, interval = spec["selection_anchors_per_episode"], spec["eval_every"]
    vision = read(templates / "real_fruit_dark_ssl_20261120.json")
    vision["seed"] = spec["seed"]
    vision["data"].update(data)
    training = vision["training"]
    training.pop("lr_restart")
    training.update(
        updates=spec["visual_updates"],
        batch=spec["visual_batch"],
        warmup=128,
        eval_every=256,
        save_every=256,
        keep_encoder_steps=[spec["visual_updates"]],
        initialize_encoder_from=spec["source_encoder"],
    )
    vision["evaluation"].pop("pca_basis")
    vision["evaluation"].update(
        baseline_step=0,
        panel_steps=[256, 1024, 2048, 4096],
        dataset_label="pouring",
        initialization_label="Fruit encoder (12,288), before pouring",
    )
    vision["tracking"] = dict(
        group,
        run_id="pouring-vision-20261123",
        job_type="raw-visual-adaptation",
        name="视觉续训｜OWAC RAW｜pouring01–03｜fruit 12288 初始化｜4096 步｜20261123",
    )

    owac = read(templates / "real_fruit_action_regression.json")
    owac["seed"] = spec["seed"]
    owac["data"] = dict(
        data, features_in_memory=True, sampling="uniform episode then uniform frame"
    )
    owac["task_names"] = [spec["instruction"]]
    owac["assets"].update(visual_encoder=str(root / "vision/encoder.pt"))
    del owac["assets"]["visual_sha256"]
    owac["training"].update(
        updates=updates,
        batch=batch,
        frozen_steps=spec["frozen_action_updates"],
        eval_every=interval,
        save_every=interval,
    )
    owac["evaluation"] = dict(
        metric="validation/selection_normalized_l1",
        validation_episodes=valid,
        selection_anchors_per_episode=anchors,
        replay_selected=True,
    )
    owac["swanlab"] = dict(
        group,
        id="pouring-owac-20261123",
        job_type="action-regression",
        name="动作训练｜OWAC 回归｜pouring01–03｜动作随机初始化｜10k｜20261123",
    )

    base = read(templates / "real_fruit_baselines_20261120.json")
    base.update(
        seed=spec["seed"],
        root=str(root),
        data=data,
        validation=valid,
        action_initialization="random",
        replay_protocol="commands",
        selection=(
            f"minimum episode-macro normalized L1 on {anchors} fixed anchors per trajectory; "
            "full held-out replay of selected weights"
        ),
    )
    base["training"].update(
        updates=updates,
        batch=batch,
        eval_every=interval,
        balance_pick_and_place=False,
        selection_anchors_per_episode=anchors,
    )
    base["pi05"].update(
        updates=updates,
        batch=batch,
        microbatch=spec["pi_microbatch"],
        eval_every=interval,
        selection_anchors_per_episode=anchors,
        prompt=spec["instruction"],
    )
    base["pi05"].pop("lora_rank")
    methods = {"dino-corp": "DINOv3 + CoRP", "rawild": "RAWild + π0.5", "rawvla": "RawVLA + π0.5"}
    base["swanlab"] = dict(
        group,
        job_type="action-flow-matching",
        run_ids={key: f"pouring-{key}-20261123" for key in methods},
        names={
            key: f"动作训练｜{label}｜pouring01–03｜动作随机初始化｜10k｜20261123"
            for key, label in methods.items()
        },
    )
    return vision, owac, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["align", "configs", "bind-encoder"])
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    root = Path(spec["root"])
    if args.stage == "align":
        prepare_data(spec)
    elif args.stage == "configs":
        if digest(Path(spec["source_encoder"])) != spec["source_encoder_sha256"]:
            raise ValueError("Unexpected fruit visual initialization")
        receipt = read(root / "receipts/split.json")
        signature = digest(root / "aligned/inventory.json")
        if signature != receipt["inventory_sha256"]:
            raise ValueError("Aligned manifest differs from split")
        configs = make_configs(spec, args.spec.parent, receipt["revision"], signature)
        for name, payload in zip(("vision", "owac-template", "baselines"), configs, strict=True):
            write(root / f"configs/{name}.json", payload)
    else:
        status = read(root / "vision/status.json")
        if status != dict(step=spec["visual_updates"], status="completed"):
            raise ValueError("Refusing to start the action head before pouring vision completes")
        owac = read(root / "configs/owac-template.json")
        owac["assets"]["visual_sha256"] = digest(Path(owac["assets"]["visual_encoder"]))
        write(root / "configs/owac.json", owac)


if __name__ == "__main__":
    main()
