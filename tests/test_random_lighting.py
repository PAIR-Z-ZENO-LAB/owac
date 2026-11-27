import copy
import json

import pytest

from owac.runtime.dynamic_results import aggregate, collect
from owac.simulation.random_lighting import LIGHTS, PROTOCOL, build_manifest, entry_schedule


def source():
    return {
        "entries": [
            {
                "suite": "libero_goal",
                "task_id": 6,
                "base_seed_id": 49,
                "init_state_index": 49,
                "base_seed": 49,
                "noise_seed": 100 + i,
                "lighting_domain": light,
                "lighting_ev": i - 2,
                "lighting_scale": 2 ** (i - 2),
                "lighting_rig": "common_agentview_key",
                "agentview_table_flood": i / 10,
                "sensor_saturation_ev": i,
            }
            for i, light in enumerate(LIGHTS)
        ]
    }


def test_random_schedule_preserves_every_non_light_source_field_and_is_replayable(tmp_path):
    original = source()
    before = copy.deepcopy(original)
    generated = build_manifest(original)
    assert generated == build_manifest(original) and original == before
    normal = original["entries"][0]
    seeds = set()
    for entry in generated["entries"]:
        assert all(entry[k] == value for k, value in normal.items())
        seeds.add(entry["schedule_seed"])
        events = entry["lighting_schedule"]["events"]
        assert [e["step"] for e in events] == list(range(10, 901, 10))
        previous = "Normal"
        for event in events:
            assert event["lighting"] in LIGHTS and event["lighting"] != previous
            previous = event["lighting"]
        schedule = entry_schedule(entry, tmp_path)
        # Saturation is a camera/sensor parameter, never part of the renderer mutation.
        assert all("sensor_saturation_ev" not in e["parameters"] for e in schedule.events.values())
    assert len(seeds) == 5


def test_incomplete_five_sequences_cannot_be_counted_as_robust_initial_state():
    rows = [
        {
            "static_id": "goal/6/49/Normal",
            "schedule_id": i,
            "scored_success": i != 4,
            "success": True,
            "switches": 10,
        }
        for i in range(5)
    ]
    reference = {"goal/6/49/Normal": {"success": True}}
    partial = aggregate(rows[:4], reference, 5)
    assert not partial["complete"] and partial["complete_initial_states"] == 0
    assert partial["confidence_interval"] is None
    final = aggregate(rows, reference, 5)
    assert final["complete"] and final["success_pct"] == 80
    assert final["matched_normal_pct"] == 100 and final["drop_pp"] == 20
    assert final["native_successes"] == 5 and final["all_sequences_successes"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(rows + rows[:1], reference, 5)


def test_collector_rejects_static_mixed_schedules_and_unscored_results(tmp_path):
    manifest = build_manifest(source())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    entry = manifest["entries"][0]
    directory = tmp_path / "evaluation/method/shard-00"
    directory.mkdir(parents=True)
    output = directory / "episodes.jsonl"
    cfg = {"manifest": str(path), "methods": {"method": {"root": str(tmp_path)}}}
    row = {
        "id": entry["dynamic_id"],
        "static_id": entry["static_id"],
        "illumination_protocol": PROTOCOL,
        "formal": True,
        "native_exit": 0,
        "schedule_sha256": entry["schedule_sha256"],
        "schedule_id": 0,
        "suite": "libero_goal",
        "task": 6,
        "initial_index": 49,
        "success": True,
        "common_budget_success": False,
        "steps": 350,
        "applied_lighting_events": [{"step": 10}],
    }
    output.write_text(json.dumps(row) + '\n{"id":')
    assert collect(cfg)["method"][0]["scored_success"] is False
    for change in (
        {"illumination_protocol": "static"},
        {"formal": False},
        {"schedule_sha256": "other"},
        {"schedule_id": 4},
        {"native_exit": 1},
    ):
        output.write_text(json.dumps(row | change) + "\n")
        with pytest.raises(ValueError):
            collect(cfg)
