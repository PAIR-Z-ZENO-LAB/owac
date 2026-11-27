from types import SimpleNamespace

import numpy as np
import pytest

from owac.raw.teleop_mcap import JOINTS, causal_state_indices, joint_positions, unpack_bg10


@pytest.mark.parametrize("width", [1, 4, 7, 640])
def test_padded_vendor_bitstream(width):
    rng = np.random.default_rng(29)
    values = rng.integers(0, 1024, (3, width), dtype=np.uint16)
    values[0, 0] = 1023
    values[1, -1] = 0
    active = (width * 10 + 7) // 8
    stride = active + 5
    encoded = b"".join(
        sum(int(x) << (10 * i) for i, x in enumerate(row)).to_bytes(active, "little")
        + b"\xff" * (stride - active)
        for row in values
    )
    out = unpack_bg10(encoded, width, len(values), stride)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, values)


def test_state_requires_available_telemetry_and_uses_freshest_header():
    hdr = np.array([110, 103, 120, 130])
    log = np.array([115, 128, 150, 132])
    ts = np.array([100, 114, 116, 129, 133, 151])
    np.testing.assert_array_equal(causal_state_indices(hdr, log, ts), [-1, -1, 0, 0, 3, 3])
    msg = SimpleNamespace(name=JOINTS[::-1], position=np.arange(8)[::-1])
    np.testing.assert_array_equal(joint_positions(msg), np.arange(8))


def test_causal_reader_does_not_cross_episode_or_feed_future_state(tmp_path):
    pytest.importorskip("torch")
    from owac.training.teleop_data import TeleopFrames

    sources = []
    for sid in (1, 2):
        file = tmp_path / f"{sid}.npz"
        t = 1_000_000_000 * np.arange(1, 4, dtype=np.int64)
        a = np.full((3, 8), sid, np.float32)
        cmd = np.stack([np.full(8, sid * 10 + i) for i in range(3)])
        raw = np.arange(12, dtype=np.uint16).reshape(3, 2, 2) + sid * 10
        np.savez(
            file,
            observation_ns=t,
            state_ns=t - 4,
            state_log_ns=t - 1,
            action_ns=t + 1,
            state=a,
            action=cmd,
            commands=cmd,
            command_ns=t + 1,
            camera_header_ns=np.stack((t - 1, t), axis=1),
            u2_raw=raw,
            u3_raw=raw + 20,
        )
        sources.append(SimpleNamespace(path=file, group_id=str(sid), suite="fruit"))
    dataset = TeleopFrames(sources, future_s=0.4)
    assert len(dataset) == 4
    a = dataset[1]
    b = dataset[2]
    assert (a["episode"], b["episode"]) == ("1", "2")
    assert a["raw"].shape == (2, 2, 2) and a["raw"].dtype == np.uint16
    assert np.all(a["raw"][1] - a["raw"][0] == 20)
    assert a["state"][0] == 1 and a["action"][0] == 11 and a["future_action"][0] == 12
    assert a["state_log_ns"] <= a["observation_ns"] <= a["action_ns"] < a["future_action_ns"]


def test_shifted_patch_coordinates_recover_physical_locations():
    torch = pytest.importorskip("torch")
    from owac.training.teleop_probe import match_patches

    full = torch.eye(20).reshape(4, 5, 20)
    ref = full[:-1, :-1].reshape(1, 12, 20)
    query = full[1:, 1:].reshape(1, 12, 20)
    assert match_patches(ref, query, (3, 4), 16).tolist() == [6.0, 6.0, 0.0]
    assert match_patches(ref, query, (3, 4), 16, [(16, 16, 32, 32)]).tolist() == [1.0, 1.0, 0.0]
