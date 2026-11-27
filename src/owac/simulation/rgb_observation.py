"""Sensor-matched RGB observations without changing native policy preprocessing."""

import numpy as np

from owac.simulation.rawvla_teacher_controller import raw_observation_frames

CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
SENSOR_PROTOCOL = "rawvla-noisy-raw10-fixed-isp-v1"
CLEAN_PROTOCOL = "rendered-rgb-v1"
RAW_PROTOCOL = "rawvla-noisy-raw-view8-v1"


def policy_observation(observation, entry, *, frame_index, protocol):
    """Apply the benchmark sensor once, then restore the simulator camera convention.

    Noise is sampled in the same upright coordinates as RAW/ISP evaluations.
    The native policies retain their own camera flips, crops and normalization.
    """
    if protocol == CLEAN_PROTOCOL:
        return observation
    if protocol not in (SENSOR_PROTOCOL, RAW_PROTOCOL):
        raise ValueError(f"Unknown RGB observation protocol: {protocol}")
    from rawvla_bench.raw import make_rawvla_observation

    result = dict(observation)
    for view, (key, frame) in enumerate(
        zip(CAMERA_KEYS, raw_observation_frames(observation).values(), strict=True)
    ):
        rgb = make_rawvla_observation(
            frame,
            noise_seed=int(entry["noise_seed"]),
            frame_index=int(frame_index),
            view_index=view,
            representation="raw" if protocol == RAW_PROTOCOL else "default_isp",
            sensor_saturation_ev=float(entry["sensor_saturation_ev"]),
        )
        result[key] = np.ascontiguousarray(rgb[::-1, ::-1])
    return result
