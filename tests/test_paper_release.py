import gzip
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_ema_roundtrip_preserves_both_heads_and_bundled_tokenizer(tmp_path, head):
    from transformers import T5TokenizerFast

    from owac.paper.checkpoint import TOKENIZER, export, load_policy
    from owac.paper.infer import LiberoPolicy
    from owac.paper.model import CompactPolicy
    from owac.paper.prepare import ROOT

    torch.set_num_threads(2)
    cfg = json.loads((ROOT / f"configs/paper/{head}.json").read_text())
    model = cfg["model"]
    model["raw_encoder"].update(width=24, depth=1, heads=3)
    model["language"].update(d_model=16, d_ff=32, d_kv=4, num_layers=1, num_heads=4)
    model.update(hidden_dim=24, tokens_per_view=2, resampler_heads=3, film_rank=4)
    model["action_generator"].update(num_heads=3, adaln_rank=4)
    original = CompactPolicy(cfg).eval()
    state = original.state_dict()
    legacy = tmp_path / "policy.pt"
    torch.save({"model": state, "step": 15, "config": cfg}, legacy)
    output = tmp_path / "policy.safetensors"
    export(legacy, output)
    recovered, config, step = load_policy(output)
    assert step == 15 and config["statistics"] == cfg["statistics"]
    for name, value in recovered.state_dict().items():
        assert torch.equal(value, state[name])
    tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER, local_files_only=True)
    assert tokenizer("put the bowl on the plate").input_ids[-1] == 1
    policy = LiberoPolicy(output)
    raw = np.zeros((2, 256, 256, 3), np.uint8)
    command = policy.predict(raw, np.zeros(9, np.float32), cfg["task_names"][0])
    assert command.shape == (12, 7) and np.isfinite(command).all()
    with pytest.raises(ValueError, match="pseudo-RAW"):
        policy.predict(raw.astype(np.float32), np.zeros(9), "test")


def test_manifest_keeps_exact_split_and_reader_paths(tmp_path):
    from owac.paper.prepare import prepare
    from owac.raw.all_bench_distillation import load_inventory

    dataset, work = tmp_path / "RawVLA-Bench", tmp_path / "run"
    assert prepare(dataset, work) == 2403
    sources, _ = load_inventory(work / "inventory.json")
    assert all(s.path.is_relative_to(dataset) for s in sources)
    counts = {
        part: sum(s.domain == "libero" and s.split == part for s in sources)
        for part in ("train", "validation", "test")
    }
    assert counts == {"train": 1421, "validation": 177, "test": 173}
    cfg = json.loads((work / "j5.json").read_text())
    assert cfg["relations"]["anchor"] == str(work / "j3/encoder.pt")


def test_scorer_rejects_partial_duplicate_and_changed_schedule(tmp_path):
    from owac.paper.score import receipts
    from owac.simulation.random_lighting import LIGHTS, PROTOCOL, RENDERER_KEYS, build_manifest
    from owac.simulation.rgb_observation import RAW_PROTOCOL

    src = dict(
        suite="test",
        task_id=0,
        base_seed_id=0,
        init_state_index=0,
        **dict.fromkeys(RENDERER_KEYS, 1.0),
    )
    manifest = {"entries": [dict(src, lighting_domain=name) for name in LIGHTS]}
    entries = build_manifest(manifest)["entries"]
    expected = {e["dynamic_id"]: e for e in entries}
    rows = [
        dict(
            id=e["dynamic_id"],
            suite="test",
            task=0,
            initial_index=0,
            formal=True,
            native_exit=0,
            common_budget_success=True,
            success=True,
            switch_count=3,
            checkpoint_sha256="a" * 64,
            observation_protocol=RAW_PROTOCOL,
            illumination_protocol=PROTOCOL,
            **{k: e[k] for k in ("static_id", "schedule_id", "schedule_sha256")},
        )
        for e in entries
    ]

    def save(values):
        with gzip.open(tmp_path / "episodes.jsonl.gz", "wt") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in values)

    save(rows)
    assert receipts(tmp_path, expected, switching=True)[1] == "a" * 64
    save(rows[:-1])
    with pytest.raises(ValueError, match="Expected"):
        receipts(tmp_path, expected, switching=True)
    save([*rows, rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        receipts(tmp_path, expected, switching=True)
    save([dict(rows[0], schedule_sha256="changed"), *rows[1:]])
    with pytest.raises(ValueError, match="schedule"):
        receipts(tmp_path, expected, switching=True)


def test_encoder_training_runs_without_cgroup_memory_files(tmp_path):
    from owac.training.raw_ssl_train import host_memory_metrics

    assert host_memory_metrics(tmp_path) == {}
    (tmp_path / "memory.stat").write_text("anon 1\nshmem 2\nfile 3\n")
    (tmp_path / "memory.events").write_text("oom_kill 4\n")
    (tmp_path / "memory.current").write_text("1073741824\n")
    metrics = host_memory_metrics(tmp_path)
    assert metrics["host/memory_current_gib"] == 1
    assert metrics["host/oom_kills"] == 4
