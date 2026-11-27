"""Optional, action-count-indexed illumination interventions for native evaluation.

These are separate diagnostic conditions, not the original static five-light
benchmark. The model never receives the schedule. Changes use the renderer's
existing calibrated domains and do not reset physics, policy history, or noise.
"""

import json

import numpy as np


class LightingSchedule:
    def __init__(self, specification=None, *, trace_path=None):
        self.specification = specification
        self.trace_path = trace_path
        self.events, self.applied = {}, []
        if specification is not None:
            if not specification["name"] or not specification["events"]:
                raise ValueError("A lighting schedule needs a name and explicit events")
            previous = -1
            for event in specification["events"]:
                step = event["step"]
                if not isinstance(step, int) or step <= previous:
                    raise ValueError(
                        "Lighting events must have increasing nonnegative action counts"
                    )
                self.events[step] = dict(event)
                previous = step

    def condition(self, initial_lighting):
        if self.specification is None:
            return initial_lighting
        return f"scheduled:{self.specification['name']}:{initial_lighting}"

    def record(self, step, parameters, **metadata):
        self.applied.append(
            {
                "step": step,
                "lighting": self.events[step]["lighting"],
                "parameters": dict(parameters),
                **metadata,
            }
        )
        if self.trace_path is not None:
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(self.applied[-1]) + "\n")

    def result(self, initial_lighting):
        if self.specification is None:
            return {}
        return {
            "initial_lighting": initial_lighting,
            "lighting_schedule": self.specification,
            "applied_lighting_events": self.applied,
            "lighting_schedule_scope": (
                "Exogenous action counts; original calibrated renderer domains; "
                "not a static five-light score. Unreached events remain in the declaration."
            ),
        }


def apply_libero_event(env, event, seed):
    """Change rendered light without advancing physics or observable sampling clocks."""
    from rawvla_bench.lighting import (
        COMMON_AGENTVIEW_KEY_RIG,
        DEFAULT_EV_RANGES,
        _assign_scaled,
        _set_agentview_table_flood,
        lighting_parameters_for_sample,
    )

    domain = event["lighting"]
    if "parameters" in event:
        parameters = dict(event["parameters"])
    else:
        rng = np.random.default_rng(seed)
        ev = float(rng.uniform(*DEFAULT_EV_RANGES[domain]))
        parameters = dict(
            lighting_parameters_for_sample(domain, ev, sample_unit=float(rng.random()))
        )
    # Reuse the author's calibrated intensity updates, excluding its static
    # setup-only sim.forward(). Recomputing dynamics here would also change
    # derived geometry between the original observation and this render.
    baseline = env._rawvla_bench_lighting_baseline
    model = env.sim.model
    scale = float(2.0 ** parameters["lighting_ev"])
    for name in ("light_ambient", "light_diffuse", "light_specular"):
        _assign_scaled(model, name, getattr(baseline, name), scale)
    headlight = getattr(getattr(model, "vis", None), "headlight", None)
    if headlight is not None:
        for name in ("ambient", "diffuse", "specular"):
            _assign_scaled(headlight, name, getattr(baseline, "headlight_" + name), scale)
    _set_agentview_table_flood(
        env,
        parameters.get("agentview_table_flood", 0.0),
        require=parameters.get("lighting_rig", "default") == COMMON_AGENTVIEW_KEY_RIG,
    )
    return parameters


def step_libero_observation(env, action, *, observation_step, schedule, lighting, seed):
    """Produce the scheduled frame through the normal native sensor sampling.

    The action is already selected from the previous observation. Set the
    exogenous light before that action's native step captures the next frame;
    forcing a fresh observation afterward changes sensor timing and proprio.
    """
    if observation_step in schedule.events:
        lighting = apply_libero_event(env, schedule.events[observation_step], seed)
        schedule.record(
            observation_step,
            lighting,
            noise_frame_index=observation_step + 10,
            applied_before_action_index=observation_step - 1,
            sampling_phase="normal native step producing the declared observation",
        )
    return env.step(action), lighting


def apply_robotwin_event(task, event, case):
    """Change illumination while preserving the sensor's ongoing noise stream."""
    from envs.lighting import apply_lighting

    noise_index = task.cameras.raw_sensor_noise_frame_index
    parameters = apply_lighting(
        task,
        event["lighting"],
        case["lighting_strategy"],
        sample_unit=case["lighting_sample_unit"],
    )
    # The author's static capture helper resets this index. A live transition
    # must not replay earlier Gaussian draws as if they were new observations.
    task.cameras.raw_sensor_noise_frame_index = noise_index
    return parameters, noise_index
