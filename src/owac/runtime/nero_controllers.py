"""Identify local NERO drivers by ROS namespace and CAN interface."""

import os
from pathlib import Path


def local_controllers():
    """Also find drivers that own a CAN device but are still starting or use another ROS domain."""
    import yaml

    controllers = set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip() != "nero_ctrl_node":
                continue
            args = os.fsdecode((proc / "cmdline").read_bytes()).split("\0")
            ns = next(
                (word.removeprefix("__ns:=") for word in args if word.startswith("__ns:=")), ""
            )
            for key, value in zip(args, args[1:], strict=False):
                if key == "--params-file":
                    # ROS launch can emit tagged tuples. BaseLoader reads values as plain strings.
                    params = yaml.load(Path(value).read_bytes(), Loader=yaml.BaseLoader)
                    for entry in params.values():
                        channel = entry.get("ros__parameters", {}).get("can_port")
                        if channel:
                            controllers.add((ns, channel))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return controllers
