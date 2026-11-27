#!/usr/bin/env python3
"""Publish the board's two native RAW10 streams for the keyboard recorder."""

import argparse
import math
import os
import socket
import threading
import time
from array import array

import rclpy
from agx_arm_msgs.msg import OwacRawFrame
from owac_raw import (
    CHANNEL_TOPICS,
    DEFAULT_FPS,
    DEFAULT_PORT,
    HEIGHT,
    STRIDE,
    TOPICS,
    WIDTH,
    WIRE_WIDTH,
    capture_pair,
)


def publish_captures(node, host, port, fps, required=tuple(CHANNEL_TOPICS), optional=()):
    selected = set(required) | set(optional)
    publishers = {
        i: node.create_publisher(OwacRawFrame, topic, 5)
        for i, (channel, topic) in enumerate(CHANNEL_TOPICS.items())
        if channel in selected
    }
    required_ids = tuple(TOPICS.index(CHANNEL_TOPICS[channel]) for channel in required)
    previous_ts = [None] * len(TOPICS)
    stopped = threading.Event()
    node.context.on_shutdown(stopped.set)
    while rclpy.ok():
        try:
            with (
                socket.create_connection((host, port), timeout=2.0) as sock,
                sock.makefile("rb") as stream,
            ):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                node.get_logger().info(f"Connected to OWAC at {host}:{port}, requesting {fps:g} Hz")
                while rclpy.ok():
                    start = time.monotonic()
                    sent_ns, done_ns, frames = capture_pair(sock, stream, required_ids)
                    if not rclpy.ok():
                        return
                    for frame in frames:
                        cam = frame.camera_id
                        if cam not in publishers or previous_ts[cam] == frame.board_timestamp_us:
                            continue
                        msg = OwacRawFrame(
                            camera_id=cam,
                            sequence=frame.sequence,
                            board_timestamp_us=frame.board_timestamp_us,
                            request_sent_monotonic_ns=sent_ns,
                            received_monotonic_ns=frame.received_monotonic_ns,
                            capture_done_monotonic_ns=done_ns,
                            width=WIDTH,
                            wire_width=WIRE_WIDTH,
                            height=HEIGHT,
                            stride=STRIDE,
                            fourcc="BG10",
                            data=array("B", frame.data),
                        )
                        msg.header.frame_id = f"owac_u{cam + 2}"
                        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
                            frame.received_unix_ns, 10**9
                        )
                        publishers[cam].publish(msg)
                        previous_ts[cam] = frame.board_timestamp_us
                    stopped.wait(max(0, start + 1 / fps - time.monotonic()))
        except (OSError, ValueError) as exc:
            if rclpy.ok():
                node.get_logger().warning(
                    f"OWAC capture interrupted ({exc}), reconnecting in one second"
                )
                stopped.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("OWAC_HOST"), required=not os.environ.get("OWAC_HOST")
    )
    parser.add_argument("--port", type=int, default=os.environ.get("OWAC_PORT", DEFAULT_PORT))
    parser.add_argument("--fps", type=float, default=os.environ.get("OWAC_FPS", DEFAULT_FPS))
    parser.add_argument("--channel", action="append", choices=CHANNEL_TOPICS)
    parser.add_argument("--optional-channel", action="append", choices=CHANNEL_TOPICS, default=[])
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0 or not 0 < args.port <= 65535:
        parser.error("expected a positive finite fps and a TCP port in 1..65535")
    rclpy.init(args=[])
    node = rclpy.create_node("owac_raw_camera")
    try:
        required = (
            args.channel
            if args.channel is not None
            else (() if args.optional_channel else tuple(CHANNEL_TOPICS))
        )
        publish_captures(node, args.host, args.port, args.fps, required, args.optional_channel)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
