"""Run the fixed four-policy queue on physical GPUs 0, 1, and 2 only."""

import fcntl
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from prepare_pouring import read, write

SOURCE = Path(__file__).resolve().parents[2]
SPEC_FILE = SOURCE / "configs/experiments/real_pouring_20261123.json"
SPEC = read(SPEC_FILE)
ROOT = Path(SPEC["root"])
PYTHON = sys.executable
PREPARE = [PYTHON, str(SOURCE / "scripts/experiments/prepare_pouring.py")]
CONFIG = str(ROOT / "configs/baselines.json")


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def phase(name, state, **extra):
    write(ROOT / f"queue/{name}.json", dict(stage=name, state=state, time=stamp(), **extra))
    print(stamp(), name, state, extra, flush=True)


def environment(uuid=""):
    return dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="3",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        SWANLAB_MODE=os.environ.get("SWANLAB_MODE", "local"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TMPDIR=str(ROOT / "tmp"),
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(SOURCE / "src"), os.environ.get("PYTHONPATH")))
        ),
    )


def run(name, command, env, cwd=SOURCE):
    with (ROOT / f"logs/{name}.log").open("a") as output:
        subprocess.run(
            command, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True
        )


def train_ddp(name, module, config, folder, env):
    completed = folder / "status.json"
    expected = SPEC["visual_updates" if name == "vision" else "action_updates"]
    if completed.exists() and read(completed).get("status") == "completed":
        if read(completed)["step"] != expected:
            raise ValueError(f"Unexpected terminal step: {name}")
        return
    command = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "--module",
        module,
        "--config",
        str(config),
        "--output",
        str(folder),
    ]
    if (folder / "latest.pt").is_file():
        command.extend(["--resume", str(folder / "latest.pt")])
    run(name, command, env)
    status = read(completed)
    if status["status"] != "completed" or status["step"] != expected:
        raise RuntimeError(f"Training did not reach the declared budget: {name}")


def finished_action_weights(folder, method, config):
    import torch

    if not (folder / "final.pt").exists():
        return False
    for filename in ("final.pt", "best.pt"):
        saved = torch.load(folder / filename, map_location="meta", weights_only=False, mmap=True)
        source = saved["config"]
        updates = config["training" if method == "dino-corp" else "pi05"]["updates"]
        if not (
            saved["method"] == method
            and 0 < saved["step"] <= updates
            and (filename != "final.pt" or saved["step"] == updates)
            and all(source.get(key) == value for key, value in config.items())
            and source["action_initialization"] == "random"
        ):
            raise ValueError(f"Checkpoint is not from this completed training run: {filename}")
    return True


def run_or_replay(method, folder, env):
    command = ["bash", "scripts/experiments/run_real_fruit_baseline.sh", method, CONFIG]
    config = read(Path(CONFIG))
    if not finished_action_weights(folder, method, config):
        try:
            run(method, command, env)
            return
        except subprocess.CalledProcessError:
            if not finished_action_weights(folder, method, config):
                raise
    run(f"{method}-evaluation-only", [*command, "--eval-only"], env)


def baseline(method, env):
    folder = ROOT / "runs" / method
    folder.mkdir(exist_ok=True)
    track = [
        PYTHON,
        str(SOURCE / "scripts/experiments/track_real_fruit_baseline.py"),
        "--config",
        CONFIG,
        "--method",
        method,
    ]
    with (ROOT / f"logs/cloud-{method}.log").open("a") as output:
        proc = subprocess.Popen(
            track, cwd=SOURCE, env=environment(), stdout=output, stderr=subprocess.STDOUT
        )
        try:
            if not (folder / "replay/results.json").exists():
                run_or_replay(method, folder, env)
            if not all(
                (folder / item).is_file() for item in ("best.pt", "final.pt", "replay/results.json")
            ):
                raise FileNotFoundError(
                    f"Missing selected/final policy or full replay for {method}"
                )
            if proc.wait(timeout=600):
                raise RuntimeError(f"SwanLab tracker failed for {method}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def stage(name, worker, **identity):
    path = ROOT / f"queue/{name}.json"
    if path.exists() and read(path)["state"] == "completed":
        return True
    phase(name, "running", **identity)
    try:
        worker()
    except Exception as exc:
        phase(name, "failed", error=str(exc), **identity)
        return False
    phase(name, "completed", **identity)
    return True


def main():
    os.umask(0o077)
    for folder in ("receipts", "queue", "logs", "configs", "runs", "tmp"):
        (ROOT / folder).mkdir(exist_ok=True)
    lock = (ROOT / "queue/runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected_lanes = {
        "0": ["vision", "owac"],
        "1": ["input-cache", "dino-corp", "rawild"],
        "2": ["rawvla"],
    }
    if SPEC["allocation"] != expected_lanes:
        raise ValueError("Unexpected GPU allocation")
    for index, stages in expected_lanes.items():
        for name in stages:
            if not (ROOT / f"queue/{name}.json").exists():
                phase(name, "queued", gpu=int(index))
    phase("pipeline", "waiting_for_verified_hub_and_server_copy", pid=os.getpid())
    while not all((ROOT / f"receipts/{item}").is_file() for item in ("lab.sha256", "hub.json")):
        time.sleep(8)
    env = environment()

    def prepare():
        run(
            "server-verify",
            ["sha256sum", "--quiet", "-c", str(ROOT / "receipts/lab.sha256")],
            env,
            cwd=Path(SPEC["raw"]),
        )
        run("align", [*PREPARE, "align", "--spec", str(SPEC_FILE)], env)
        run("configs", [*PREPARE, "configs", "--spec", str(SPEC_FILE)], env)

    if not stage("prepare", prepare):
        phase("pipeline", "failed", dependency="prepare")
        return 1
    phase("pipeline", "training", pid=os.getpid())
    cache_ready = threading.Event()

    def lane(gpu):
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        if not uuid.startswith("GPU-") or "\n" in uuid:
            raise ValueError(f"Invalid physical GPU {gpu}")
        gpu_env = environment(uuid)
        identity = dict(gpu=gpu, gpu_uuid=uuid)
        success = True
        for name in expected_lanes[str(gpu)]:
            if name == "rawvla":
                cache_ready.wait()
            if name in ("dino-corp", "rawild", "rawvla"):
                if read(ROOT / "queue/input-cache.json")["state"] != "completed":
                    phase(name, "blocked", dependency="input-cache", **identity)
                    success = False
                    continue
                ok = stage(name, lambda method=name: baseline(method, gpu_env), **identity)
            elif name == "input-cache":
                try:
                    ok = stage(
                        name,
                        lambda: run(
                            "input-cache",
                            [
                                PYTHON,
                                "-m",
                                "owac.training.fruit_baseline_cache",
                                "--config",
                                CONFIG,
                            ],
                            gpu_env,
                        ),
                        **identity,
                    )
                finally:
                    cache_ready.set()
            elif name == "vision":
                ok = stage(
                    name,
                    lambda: train_ddp(
                        "vision",
                        "owac.training.raw_ssl_train",
                        ROOT / "configs/vision.json",
                        ROOT / "vision",
                        gpu_env,
                    ),
                    **identity,
                )
            else:
                if read(ROOT / "queue/vision.json")["state"] != "completed":
                    phase(name, "blocked", dependency="vision", **identity)
                    success = False
                    continue

                def owac():
                    run("bind-encoder", [*PREPARE, "bind-encoder", "--spec", str(SPEC_FILE)], env)
                    folder = ROOT / "runs/owac"
                    train_ddp(
                        "owac",
                        "owac.training.teleop_policy_train",
                        ROOT / "configs/owac.json",
                        folder,
                        gpu_env,
                    )
                    if not all(
                        (folder / key).is_file()
                        for key in ("policy-best.pt", "policy-final.pt", "replay/results.json")
                    ):
                        raise ValueError("Missing OWAC selected/final policy or full replay")

                ok = stage(name, owac, **identity)
            success = success and ok
        return success

    with ThreadPoolExecutor(max_workers=3) as lanes:
        results = list(lanes.map(lane, (0, 1, 2)))
    if all(results):
        ok = stage(
            "cloud-audit",
            lambda: run(
                "cloud-audit",
                [PYTHON, str(SOURCE / "scripts/experiments/audit_pouring.py"), str(ROOT)],
                env,
            ),
        )
    else:
        ok = False
    phase("pipeline", "completed" if ok else "failed", lanes=results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
