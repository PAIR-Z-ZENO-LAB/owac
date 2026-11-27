"""Prepare paired native Bayer RAW and causal NERO state/command rows from a ROS 2 MCAP."""

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

CAMERAS = ("u2_raw", "u3_raw")
CAMERA_TOPICS = ("/teleop_camera/u2/raw10", "/teleop_camera/u3/raw10")
STATE = "/slave_right/joint_states_single"
ACTION = "/slave_right/joint_cmd_mux/output"
JOINTS = tuple(f"joint{i}" for i in range(1, 8)) + ("gripper",)


def unpack_bg10(data, width, height, stride):
    """rkcif BG10: little-endian contiguous 10-bit samples, with per-row padding."""
    active = (width * 10 + 7) // 8
    rows = np.frombuffer(data, np.uint8).reshape(height, stride)
    if active > stride:
        raise ValueError("Active BG10 row exceeds wire stride")
    # Up to 10 bits can straddle two bytes (sample starts at bit 0, 2, 4 or 6).
    bit = np.arange(width) * 10
    byte = bit // 8
    words = rows[:, byte].astype(np.uint16) | (rows[:, byte + 1].astype(np.uint16) << 8)
    return ((words >> (bit % 8).astype(np.uint16)) & 1023).astype("<u2", copy=False)


def joint_positions(msg):
    if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
        raise ValueError("JointState has inconsistent names and positions")
    positions = dict(zip(msg.name, msg.position, strict=True))
    ordered = np.asarray([positions[name] for name in JOINTS], dtype=np.float32)
    if not np.isfinite(ordered).all():
        raise ValueError("Missing or nonfinite joint/gripper position")
    return ordered


def causal_state_indices(header_ns, log_ns, observation_ns):
    """Return the freshest stamped state actually available by observation time, or -1."""
    order = np.lexsort((log_ns, header_ns))
    available = np.maximum(header_ns, log_ns)[order]
    arrival = np.argsort(available, kind="stable")
    latest = np.maximum.accumulate(arrival)
    idx = np.searchsorted(available[arrival], observation_ns, side="right") - 1
    return np.where(idx >= 0, order[latest[idx]], -1)


def _stamps(rows):
    return np.array([r[:2] for r in rows], dtype=np.int64).T


def _quantiles(ns):
    ms = np.asarray(ns) * 1e-6
    return dict(zip(("p50", "p95", "max"), np.quantile(ms, [0.5, 0.95, 1]).tolist(), strict=True))


def prepare_episode(files, destination):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    camera = [{}, {}]
    signals = {STATE: [], ACTION: []}
    layout = [None, None]
    counts = Counter()
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        tmp = Path(tmp)
        # Stream the packed payloads to disk. Only per-frame metadata stays resident.
        spool = [(tmp / f"{name}.bin").open("w+b") for name in CAMERAS]
        try:
            for path in files:
                with path.open("rb") as f:
                    reader = make_reader(f, decoder_factories=[DecoderFactory()])
                    topics = [*CAMERA_TOPICS, *signals]
                    for _, channel, record, msg in reader.iter_decoded_messages(
                        topics=topics, log_time_order=False
                    ):
                        topic = channel.topic
                        counts[topic] += 1
                        ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                        if topic in signals:
                            signals[topic].append((ns, record.log_time, joint_positions(msg)))
                            continue
                        view = CAMERA_TOPICS.index(topic)
                        size = (msg.width, msg.height, msg.stride, msg.wire_width)
                        if msg.fourcc != "BG10" or msg.camera_id != view:
                            raise ValueError(
                                f"Unexpected sensor format or camera identity at {topic}"
                            )
                        if layout[view] is not None and size != layout[view]:
                            raise ValueError(f"Sensor dimensions changed within {path} at {topic}")
                        layout[view] = size
                        request = msg.request_sent_monotonic_ns
                        if request in camera[view]:
                            raise ValueError(f"Duplicate {topic} frame for capture {request}")
                        if len(msg.data) != msg.height * msg.stride:
                            raise ValueError(f"Invalid BG10 payload length at {topic}")
                        camera[view][request] = (
                            ns,
                            record.log_time,
                            msg.board_timestamp_us,
                            msg.sequence,
                            msg.received_monotonic_ns,
                            msg.capture_done_monotonic_ns,
                            spool[view].tell(),
                        )
                        spool[view].write(msg.data)
            requests = sorted(camera[0].keys() & camera[1].keys())
            if not requests or not all(signals.values()):
                raise ValueError(f"Missing camera pairs, state or mux commands in {files}")
            meta = np.asarray([[view[k] for view in camera] for k in requests], dtype=np.int64)
            if np.any(meta[:, 0, 5] != meta[:, 1, 5]):
                raise ValueError("Matched request IDs have inconsistent capture completion times")
            obs = meta[:, :, 0].max(axis=1)
            if np.any(np.diff(obs) <= 0):
                raise ValueError(
                    "Host camera clock is not monotonic across captures in this episode"
                )
            state_hdr, state_log = _stamps(signals[STATE])
            state_index = causal_state_indices(state_hdr, state_log, obs)
            commands = sorted(signals[ACTION], key=lambda row: row[0])
            cmd_hdr, cmd_log = _stamps(commands)
            act_index = np.searchsorted(cmd_hdr, obs)
            valid = (state_index >= 0) & (act_index < len(commands))
            if not valid.any():
                raise ValueError(f"No bounded observation/action pairs in {files}")
            meta, obs = meta[valid], obs[valid]
            si, ai = state_index[valid], act_index[valid]
            action_series = np.stack([entry[2] for entry in commands])
            arrays = {
                "observation_ns": obs,
                "request_monotonic_ns": np.asarray(requests)[valid],
                "camera_header_ns": meta[:, :, 0],
                "camera_log_ns": meta[:, :, 1],
                "board_timestamp_us": meta[:, :, 2],
                "sequence": meta[:, :, 3],
                "received_monotonic_ns": meta[:, :, 4],
                "capture_done_monotonic_ns": meta[:, 0, 5],
                "state_ns": state_hdr[si],
                "state_log_ns": state_log[si],
                "state": np.stack([signals[STATE][i][2] for i in si]),
                "action_ns": cmd_hdr[ai],
                "action": action_series[ai],
                "command_ns": cmd_hdr,
                "command_log_ns": cmd_log,
                "commands": action_series,
            }
            for i, name in enumerate(CAMERAS):
                width, height, stride, _ = layout[i]
                images = np.lib.format.open_memmap(
                    tmp / f"{name}.npy", mode="w+", dtype="<u2", shape=(len(obs), height, width)
                )
                spool[i].flush()
                for frame, offset in enumerate(meta[:, i, 6]):
                    spool[i].seek(offset)
                    images[frame] = unpack_bg10(
                        spool[i].read(height * stride), width, height, stride
                    )
                arrays[name] = images
            np.savez(tmp / "episode.npz", **arrays)
            os.replace(tmp / "episode.npz", destination)
        finally:
            for file in spool:
                file.close()
    return {
        "num_frames": len(obs),
        "complete_captures": len(requests),
        "counts": dict(counts),
        "unpaired_frames_by_view": [len(view) - len(requests) for view in camera],
        "unbounded_rows": int((~valid).sum()),
        "width_height_stride_wire_width": layout,
        "state_age_ms": _quantiles(obs - state_hdr[si]),
        "action_wait_ms": _quantiles(cmd_hdr[ai] - obs),
        "board_pair_skew_ms": _quantiles(abs(meta[:, 0, 2] - meta[:, 1, 2]) * 1000),
        "host_pair_receive_skew_ms": _quantiles(abs(meta[:, 0, 0] - meta[:, 1, 0])),
        "pair_interval_ms": _quantiles(np.diff(obs)) if len(obs) > 1 else None,
    }


def prepare_group(ep, output, validation):
    files = sorted((ep / "episode").glob("*.mcap"))
    if not files:
        raise ValueError(f"No MCAP in {ep}")
    subset = ep.parent.name
    folder = output / subset
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (ep.name + ".npz")
    return dict(
        group_id=f"{subset}/{ep.name}",
        domain="real_teleop",
        suite=subset,
        task_name=subset,
        lighting="recorded",
        split="validation"
        if (ep.name in validation or f"{subset}/{ep.name}" in validation)
        else "train",
        relpath=str(dest.relative_to(output)),
        path=str(dest),
        cameras=list(CAMERAS),
        **prepare_episode(files, dest),
    )


def write_inventory(output, groups):
    if not groups or {r["split"] for r in groups} != {"train", "validation"}:
        raise ValueError("Need both training and held-out episodes")
    payload = dict(
        groups=groups,
        joint_names=JOINTS,
        units=["rad"] * 7 + ["m"],
        camera_topics=CAMERA_TOPICS,
        state_topic=STATE,
        action_topic=ACTION,
    )
    manifest = output / "inventory.json"
    temp = manifest.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(manifest)


def prepare(root, output, validation):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for ep in sorted(root.glob("*/episode_*")):
        if ep.is_dir():
            row = prepare_group(ep, output, validation)
            groups.append(row)
            print(row["group_id"], row["num_frames"], row["split"], flush=True)
    write_inventory(output, groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation",
        nargs="+",
        required=True,
        help="Held-out subset/episode IDs, or episode names shared across subsets",
    )
    args = parser.parse_args()
    prepare(args.data, args.output, set(args.validation))
