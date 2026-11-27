"""Frozen random renderer-only interventions, indexed by native action count."""

import hashlib
import json
from collections import defaultdict

import numpy as np

from owac.simulation.lighting_schedule import LightingSchedule

LIGHTS = ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
PROTOCOL = "libero-random-renderer-light-k10-v1"
RENDERER_KEYS = ("lighting_ev", "lighting_scale", "lighting_rig", "agentview_table_flood")


def episode_identity(entry):
    return entry.get("dynamic_id") or "/".join(
        str(entry[k]) for k in ("suite", "task_id", "base_seed_id", "lighting_domain")
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_manifest(original, *, seed=20261115, interval=10, sequences=5, max_steps=900):
    """Keep each Normal episode intact; only add independently seeded light schedules.

    Illumination amplitudes come from the original paired five-domain manifest.
    Camera saturation/exposure and sensor RNG remain those of the Normal entry.
    """
    if interval != 10 or sequences != 5:
        raise ValueError("This protocol version fixes interval=10 and five sequences")
    grouped = defaultdict(dict)
    for entry in original["entries"]:
        key = (entry["suite"], entry["task_id"], entry["base_seed_id"])
        if entry["lighting_domain"] in grouped[key]:
            raise ValueError(f"Duplicate source identity: {key}")
        grouped[key][entry["lighting_domain"]] = entry
    entries = []
    for key, domains in grouped.items():
        if set(domains) != set(LIGHTS):
            raise ValueError(f"Incomplete source lighting pair: {key}")
        normal = domains["Normal"]
        renderer = {light: {k: domains[light][k] for k in RENDERER_KEYS} for light in LIGHTS}
        for sequence in range(sequences):
            identity = "/".join(map(str, (*key, f"random-{sequence}")))
            schedule_seed = int(digest([PROTOCOL, seed, identity])[:16], 16)
            rng = np.random.default_rng(schedule_seed)
            current, events = "Normal", []
            for step in range(interval, max_steps + 1, interval):
                choices = [light for light in LIGHTS if light != current]
                current = choices[int(rng.integers(len(choices)))]
                events.append({"step": step, "lighting": current})
            schedule = {"name": PROTOCOL, "events": events}
            entries.append(
                {
                    **normal,
                    "dynamic_id": identity,
                    "static_id": episode_identity(normal),
                    "schedule_id": sequence,
                    "schedule_seed": schedule_seed,
                    "schedule_sha256": digest([schedule, renderer]),
                    "lighting_schedule": schedule,
                    "renderer_by_domain": renderer,
                }
            )
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "interval": interval,
        "sequences_per_initial_state": sequences,
        "initial_lighting": "Normal",
        "source_manifest_sha256": digest(original),
        "sensor_parameters": "unchanged Normal-entry sensor model, exposure and noise seed",
        "physical_initial_states": len(grouped),
        "entries": entries,
    }


def entry_schedule(entry, directory):
    specification = entry.get("lighting_schedule")
    if specification is None:
        return LightingSchedule()
    specification = {
        **specification,
        "events": [
            {**event, "parameters": entry["renderer_by_domain"][event["lighting"]]}
            for event in specification["events"]
        ],
    }
    return LightingSchedule(specification, trace_path=directory / "lighting-events.jsonl")


def dynamic_result(entry, schedule, observation_steps):
    if "dynamic_id" not in entry:
        return {}
    return {
        "illumination_protocol": PROTOCOL,
        "lighting": "random-k10",
        "initial_lighting": "Normal",
        "static_id": entry["static_id"],
        "schedule_id": entry["schedule_id"],
        "schedule_seed": entry["schedule_seed"],
        "schedule_sha256": entry["schedule_sha256"],
        "applied_lighting_events": schedule.applied,
        "policy_observation_steps": observation_steps,
        "fixed_sensor_saturation_ev": entry["sensor_saturation_ev"],
    }
