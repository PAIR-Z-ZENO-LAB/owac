"""OWSF v1 capture transactions and the OWAC board's packed Bayer RAW10 layout."""

import json
import struct
import time
from dataclasses import dataclass

CHANNEL_TOPICS = {"U2": "/teleop_camera/u2/raw10", "U3": "/teleop_camera/u3/raw10"}
TOPICS = tuple(CHANNEL_TOPICS.values())
DEFAULT_PORT, DEFAULT_FPS = 5560, 30.0
MSG_TYPE = "agx_arm_msgs/msg/OwacRawFrame"
HEADER = struct.Struct("<4B4I4sQI")  # after the four-byte OWSF magic
WIDTH, WIRE_WIDTH, HEIGHT, STRIDE = 640, 512, 480, 1024


@dataclass
class RawFrame:
    camera_id: int
    sequence: int
    board_timestamp_us: int
    received_monotonic_ns: int
    received_unix_ns: int
    data: bytes


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ConnectionError("OWAC connection closed during capture")
    return data


def capture_pair(sock, stream, required_ids=(0, 1)):
    """Read one complete reply; legacy callers require both RAW channels."""
    sock.sendall(b'{"cmd":"capture_frame"}\n')
    sent_ns = time.monotonic_ns()
    frames = {}
    captured_packets = 0
    # A normal response has two RAW, two YUV and one JSON acknowledgement.
    for _ in range(16):
        prefix = read_exact(stream, 4)
        if prefix != b"OWSF":
            if not prefix.startswith(b"{"):
                raise ValueError("unexpected OWAC packet prefix")
            line = prefix + stream.readline(4096)
            if not line.endswith(b"\n"):
                raise ValueError("unterminated OWAC capture acknowledgement")
            reply = json.loads(line)
            if (
                not isinstance(reply, dict)
                or reply.get("ok") is not True
                or reply.get("frames") != captured_packets
            ):
                raise ValueError(f"OWAC capture acknowledgement mismatch: {reply!r}")
            if any(camera_id not in frames for camera_id in required_ids):
                raise ValueError(f"missing RAW camera in capture: got IDs {sorted(frames)}")
            return sent_ns, time.monotonic_ns(), tuple(frames.values())

        version, cam, kind, flags, seq, width, height, stride, fmt, ts, size = HEADER.unpack(
            read_exact(stream, HEADER.size)
        )
        if version != 1 or size > 4 * 1024 * 1024:
            raise ValueError(f"unsupported OWSF header: version={version}, bytes={size}")
        payload = read_exact(stream, size)
        if not flags & 1:
            continue
        captured_packets += 1
        if kind != 0:  # YUV companion packets are drained from the shared socket.
            continue
        received_ns = time.monotonic_ns()
        unix_ns = time.time_ns()
        if cam not in (0, 1) or cam in frames:
            raise ValueError(f"unexpected or duplicate OWAC RAW camera ID {cam}")
        if (fmt, width, height, stride, size) != (
            b"BG10",
            WIRE_WIDTH,
            HEIGHT,
            STRIDE,
            HEIGHT * STRIDE,
        ):
            raise ValueError(
                f"unsupported OWAC RAW layout: {fmt!r} {width}x{height}, "
                f"stride={stride}, bytes={size}"
            )
        frames[cam] = RawFrame(cam, seq, ts, received_ns, unix_ns, payload)
    raise ValueError("OWAC capture acknowledgement missing")


def unpack_raw10(data, width, height, stride):
    """Return the active Bayer samples as a (height, width) little-endian uint16 array."""
    import numpy as np

    if (
        width <= 0
        or height <= 0
        or width % 4
        or stride < width * 5 // 4
        or len(data) != height * stride
    ):
        raise ValueError(
            f"invalid packed RAW10 geometry: {width}x{height}, stride={stride}, bytes={len(data)}"
        )
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, stride)
    groups = rows[:, : width * 5 // 4].reshape(height, width // 4, 5).astype(np.uint16)
    out = np.empty((height, width), dtype="<u2")
    out[:, 0::4] = groups[:, :, 0] | ((groups[:, :, 1] & 3) << 8)
    out[:, 1::4] = (groups[:, :, 1] >> 2) | ((groups[:, :, 2] & 15) << 6)
    out[:, 2::4] = (groups[:, :, 2] >> 4) | ((groups[:, :, 3] & 63) << 4)
    out[:, 3::4] = (groups[:, :, 3] >> 6) | (groups[:, :, 4] << 2)
    return out
