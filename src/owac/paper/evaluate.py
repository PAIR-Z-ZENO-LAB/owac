"""Run the native LIBERO evaluator with the selected OWAC policy and published lighting schedule."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pipe
from pathlib import Path

from owac.paper.infer import LiberoPolicy
from owac.simulation.random_lighting import build_manifest
from owac.simulation.rgb_observation import RAW_PROTOCOL

MANIFEST = (
    "experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rawvla", type=Path, required=True, help="pinned RawVLA checkout")
    parser.add_argument("--sim-python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["fixed", "switching"], default="fixed")
    parser.add_argument("--limit", type=int, help="run a smoke subset of the formal manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--initial-index", type=int, help="run the paired 40-task subset at one initial state"
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        parser.error("Expected 0 <= shard-index < shards")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    rawvla = args.rawvla.resolve()
    manifest_path = rawvla / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    entries = (build_manifest(manifest) if args.mode == "switching" else manifest)["entries"]
    if args.initial_index is not None:
        entries = [entry for entry in entries if entry["init_state_index"] == args.initial_index]
        if not entries:
            parser.error("No such initial state in the manifest")
    policy = LiberoPolicy(args.checkpoint, args.device)
    if policy.device.type == "cpu":
        import torch

        torch.set_num_threads(min(8, torch.get_num_threads()))
    name = policy.config["model"].get("action_head", "flow_matching")
    with args.checkpoint.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    config = {
        "author_root": str(rawvla),
        "observation_protocol": RAW_PROTOCOL,
        "policies": {
            name: {
                "execute_horizon": 12,
                "checkpoint_step": policy.step,
                "checkpoint_sha256": checksum,
            }
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluator is writing to this output directory")
        run = {
            "checkpoint": checksum,
            "mode": args.mode,
            "limit": args.limit,
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "device": args.device,
        }
        if args.initial_index is not None:
            run["initial_index"] = args.initial_index
        if args.shards != 1:
            run.update(shards=args.shards, shard_index=args.shard_index)
        identity = args.output / "protocol.json"
        if identity.exists():
            if json.loads(identity.read_text()) != run:
                parser.error(
                    "This output belongs to a different protocol. Choose a new output directory."
                )
        elif (args.output / "episodes.jsonl").exists():
            parser.error("This output contains episodes without a matching protocol.")
        else:
            with identity.open("x") as f:
                json.dump(run, f)
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            entries_path = Path(folder) / "entries.json"
            config_path.write_text(json.dumps(config))
            entries_path.write_text(
                json.dumps(entries[: args.limit][args.shard_index :: args.shards])
            )
            parent, child = Pipe()
            command = [
                args.sim_python,
                "-m",
                "owac.simulation.rgb_eval",
                "--config",
                str(config_path),
                "--entries",
                str(entries_path),
                "--output",
                str(args.output.resolve()),
                "--policy",
                name,
                "--pipe-fd",
                str(child.fileno()),
            ]
            if args.limit or args.initial_index is not None:
                command.append("--engineering")
            env = os.environ.copy()
            source = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
            with (
                parent,
                child,
                subprocess.Popen(command, env=env, pass_fds=(child.fileno(),)) as sim,
            ):
                child.close()
                try:
                    while True:
                        try:
                            req = parent.recv()
                        except EOFError:
                            break
                        actions = policy.predict(
                            req["raw"], req["proprio"], req["prompt"], seed=req["seed"]
                        )
                        parent.send(
                            {
                                "actions": actions.tolist(),
                                "dtype": actions.dtype.str,
                                "checkpoint_sha256": checksum,
                                "checkpoint_step": policy.step,
                            }
                        )
                except BaseException:
                    sim.terminate()
                    raise
                if code := sim.wait():
                    raise SystemExit(code)


if __name__ == "__main__":
    main()
