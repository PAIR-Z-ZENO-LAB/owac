import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.training.teleop_policy_data import encode_stereo_block
from scripts.experiments.prepare_pouring import episode_split, make_configs


def test_pouring_console_selects_only_the_four_pouring_checkpoints(tmp_path):
    from owac.runtime.fruit_weights import catalog, default_policy

    manifest = Path(__file__).resolve().parents[1] / "configs/runtime/pouring_catalog.json"
    choices = catalog(tmp_path, json.loads(manifest.read_text()))
    assert default_policy(choices, check=False, grasp=None, place=False) == "owac"
    expected = {
        "owac": ("owac", 6000),
        "dino-corp": ("dino-corp", 10000),
        "rawvla-pi05": ("rawvla", 8500),
        "rawild-pi05": ("rawild", 8500),
    }
    assert {key: (choice.backend, choice.step) for key, choice in choices.items()} == expected
    for key, choice in choices.items():
        assert choice.checkpoint == tmp_path / "candidates" / key / "policy.pt"
        assert choice.rtc is None


def test_incomplete_pouring_launcher_does_not_open_fruit(tmp_path, monkeypatch):
    launcher = Path(__file__).resolve().parents[1] / "scripts/pouring-infer"
    monkeypatch.setenv("OWAC_POURING_HOME", str(tmp_path))
    result = subprocess.run(["bash", launcher, "--list"], capture_output=True, text=True)
    assert result.returncode == 1 and "Pouring 推理目录不完整" in result.stderr


def test_common_pouring_command_uses_field_controls_and_keeps_both_other_releases(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    launcher_names = [
        "pouring-infer",
        "pouring-infer-field",
        "pouring-infer-original",
        "pouring-infer-proactive",
    ]
    for name in launcher_names:
        (tmp_path / name).symlink_to(scripts / name)
    spy = tmp_path / "fruit-infer"
    spy.write_text('#!/bin/sh\nprintf "%s\\n" "$OWAC_FRUIT_HOME" "$@"\n')
    spy.chmod(0o700)
    monkeypatch.delenv("OWAC_POURING_HOME", raising=False)
    for kind in ("field", "original", "proactive"):
        root = tmp_path / kind
        (root / "candidates").mkdir(parents=True)
        (root / "candidates/catalog.json").write_text("{}")
        (root / "source/owac/runtime").mkdir(parents=True)
        (root / "source/owac/runtime/fruit_tui.py").touch()
        monkeypatch.setenv(f"OWAC_POURING_{kind.upper()}_HOME", str(root))
    field = tmp_path / "field"
    (field / "FIELD-READY.json").write_text("{}")
    (field / "reset-pose.json").write_text("{}")
    digest = hashlib.sha256((field / "reset-pose.json").read_bytes()).hexdigest()
    (field / "field-config.sha256").write_text(f"{digest}  reset-pose.json\n")
    for name, kind in zip(launcher_names, ("field", "field", "original", "proactive"), strict=True):
        result = subprocess.run(
            ["bash", tmp_path / name, "--check-reset"], capture_output=True, text=True, check=True
        )
        assert result.stdout.splitlines() == [
            str(tmp_path / kind),
            "--require-reset",
            "--check-reset",
        ]


def test_pouring_launcher_loads_the_same_required_pose_for_preview_and_h(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    launcher = tmp_path / "pouring-infer"
    launcher.symlink_to(repo / "scripts/pouring-infer")
    console = tmp_path / "fruit-infer"
    console.write_text('#!/bin/sh\nexec "$TEST_PYTHON" -m owac.runtime.fruit_tui "$@"\n')
    console.chmod(0o700)
    root = tmp_path / "pouring-policy"
    (root / "source/owac/runtime").mkdir(parents=True)
    (root / "source/owac/runtime/fruit_tui.py").touch()
    (root / "candidates").mkdir()
    (root / "candidates/catalog.json").write_bytes(
        (repo / "configs/runtime/pouring_catalog.json").read_bytes()
    )
    urdf = tmp_path / "agx_arm_urdf/nero/urdf"
    urdf.mkdir(parents=True)
    for name, joints, lower, upper in (
        ("nero_description.urdf", [f"joint{i}" for i in range(1, 8)], -3, 3),
        ("nero_with_gripper_description.xacro", ["gripper_joint1", "gripper_joint2"], 0, 0.05),
    ):
        elements = [
            f'<joint name="{j}"><limit lower="{lower}" upper="{upper}" velocity="5"/></joint>'
            for j in joints
        ]
        (urdf / name).write_text("<robot>" + "".join(elements) + "</robot>")
    monkeypatch.setenv("TEST_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    monkeypatch.setenv("OWAC_POURING_HOME", str(root))
    monkeypatch.setenv("OWAC_COLLECTION", str(tmp_path))

    def run(*args):
        return subprocess.run(["bash", launcher, *args], capture_output=True, text=True)

    pose = root / "reset-pose.json"
    for flags in ((), ("--check",), ("--check-reset",)):
        result = run(*flags)
        assert result.returncode == 2 and str(pose) in result.stderr
    pose.write_text(json.dumps(dict(joints_deg=[300] * 7, gripper_mm=20)))
    assert "超出右臂行程" in run("--check").stderr
    pose.write_text(json.dumps(dict(joints_deg=[0] * 7, gripper_mm=20)))
    result = run("--check-reset", "--policy", "owac")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["policy"] == "owac" and report["path"] == str(pose)
    values = json.loads(pose.read_text())
    np.testing.assert_allclose(report["reset_rad_m"][:7], np.deg2rad(values["joints_deg"]))
    assert report["reset_rad_m"][7] == pytest.approx(values["gripper_mm"] / 1000)


def test_pouring_split_budget_and_initialization_are_shared_without_fruit_actions():
    templates = Path(__file__).resolve().parents[1] / "configs/experiments"
    spec = json.loads((templates / "real_pouring_20261123.json").read_text())
    vision, owac, baselines = make_configs(spec, templates, "dataset-revision", "inventory-digest")
    split = episode_split(spec)
    validation = [key for key, part in split.items() if part == "validation"]
    assert len(split) == 29 and len(validation) == 6
    assert validation == [
        "pouring01/episode_005",
        "pouring01/episode_009",
        "pouring02/episode_005",
        "pouring02/episode_010",
        "pouring03/episode_005",
        "pouring03/episode_010",
    ]
    assert owac["evaluation"]["validation_episodes"] == baselines["validation"] == validation
    for recipe in (vision, owac, baselines):
        assert recipe["data"]["dataset_revision"] == "dataset-revision"
        assert recipe["data"]["inventory_sha256"] == "inventory-digest"
    assert vision["training"]["initialize_encoder_from"] == spec["source_encoder"]
    assert vision["training"]["updates"] == 4096
    assert "continuation" not in owac and baselines["action_initialization"] == "random"
    assert baselines["replay_protocol"] == "commands" and owac["evaluation"]["replay_selected"]
    for recipe in (owac["training"], baselines["pi05"], baselines["training"]):
        assert (recipe["updates"], recipe["batch"], recipe["eval_every"]) == (10000, 16, 500)
    assert (
        len(
            {
                vision["tracking"]["run_id"],
                owac["swanlab"]["id"],
                *baselines["swanlab"]["run_ids"].values(),
            }
        )
        == 5
    )
    assert spec["allocation"] == {
        "0": ["vision", "owac"],
        "1": ["input-cache", "dino-corp", "rawild"],
        "2": ["rawvla"],
    }


def test_in_memory_features_preserve_frame_and_camera_order():
    class Encoder(torch.nn.Module):
        def forward(self, raw):
            return None, raw.flatten(1)[:, ::2, None]

    views = [np.arange(48, dtype="<u2").reshape(3, 4, 4) + offset for offset in (600, 900)]
    result = encode_stereo_block(Encoder(), views, 1, 3, torch.device("cpu"))
    expected = np.stack([view[1:].reshape(2, 16)[:, ::2] for view in views], axis=1)
    torch.testing.assert_close(result[..., 0], torch.as_tensor(expected.astype("f4")).bfloat16())
    assert not result.requires_grad


def test_cloud_audit_detects_a_missing_intermediate_point(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    from audit_pouring import compare

    class Cloud:
        def metrics(self, keys, *, all):
            return {"list": [{"key": "train/loss", "metrics": [{"index": 1, "data": 0.2}]}]}

    compare(Cloud(), {"train/loss": {1: 0.2}})
    with pytest.raises(ValueError, match="at 2"):
        compare(Cloud(), {"train/loss": {1: 0.2, 2: 0.1}})


def test_hub_staging_hashes_local_recordings_and_cannot_commit(tmp_path, monkeypatch):
    from scripts.experiments import publish_teleop_subset as publisher

    raw = tmp_path / "raw"
    name = "pouring01/episode_001/episode/episode_0.mcap"
    recording = raw / name
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"verified recording")
    checks = tmp_path / "lab.sha256"
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    checks.write_text(f"{digest}  {name}\n{digest}  pouring02/episode_002/episode/episode_0.mcap\n")
    staged = []

    class Hub:
        def dataset_info(self, repo):
            return SimpleNamespace(private=True, id=repo, sha="unchanged")

        def preupload_lfs_files(self, repo, additions, **kwargs):
            staged.extend(operation.path_in_repo for operation in additions)

    monkeypatch.setattr(publisher, "HfApi", lambda **kwargs: Hub())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish",
            "--raw",
            str(raw),
            "--checks",
            str(checks),
            "--repo",
            "test/dataset",
            "--subsets",
            "pouring01",
            "--preupload-mcap-only",
        ],
    )
    publisher.main()
    assert staged == [name]
    recording.write_bytes(b"partial recording")
    with pytest.raises(ValueError, match="differs from recorder"):
        publisher.main()
    assert staged == [name]


def test_hub_only_accepts_lfs_attributes_for_new_recordings():
    from scripts.experiments.publish_teleop_subset import validate_attributes

    old = "*.bin filter=lfs diff=lfs merge=lfs -text\n*.txt text\n"
    name = "pouring03/episode_010/episode/episode_0.mcap"
    rule = f"{name} filter=lfs diff=lfs merge=lfs -text\n"
    validate_attributes(old, old + rule, [name])
    with pytest.raises(ValueError):
        validate_attributes(old, rule, [name])
    with pytest.raises(ValueError):
        validate_attributes(old, old + "fruit01/** -text\n", [name])


def test_finished_baseline_replays_without_retraining(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    import run_pouring_queue as queue

    config = dict(pi05={"updates": 10000}, action_initialization="random", data={"revision": "abc"})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(config))
    monkeypatch.setattr(queue, "CONFIG", str(recipe))
    commands = []
    monkeypatch.setattr(queue, "run", lambda name, command, env: commands.append(command))
    for filename, step in (("best.pt", 8500), ("final.pt", 10000)):
        torch.save(
            dict(method="rawild", step=step, config=config, weights=torch.ones(1)),
            tmp_path / filename,
        )
    queue.run_or_replay("rawild", tmp_path, {})
    assert len(commands) == 1 and commands[0][-1] == "--eval-only"
    with pytest.raises(ValueError, match="not from this completed training run"):
        queue.finished_action_weights(
            tmp_path, "rawild", dict(config, data={"revision": "another"})
        )


@pytest.mark.parametrize("checkpoint_written", [True, False])
def test_replay_recovery_never_masks_an_incomplete_training(
    tmp_path, monkeypatch, checkpoint_written
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    import run_pouring_queue as queue

    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    monkeypatch.setattr(queue, "CONFIG", str(recipe))
    observed = iter([False, checkpoint_written])
    monkeypatch.setattr(queue, "finished_action_weights", lambda *args: next(observed))
    attempts = []

    def run(name, command, env):
        attempts.append(command)
        if "--eval-only" not in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(queue, "run", run)
    if checkpoint_written:
        queue.run_or_replay("rawild", tmp_path, {})
        assert len(attempts) == 2 and attempts[1] == [*attempts[0], "--eval-only"]
    else:
        with pytest.raises(subprocess.CalledProcessError):
            queue.run_or_replay("rawild", tmp_path, {})
        assert len(attempts) == 1
