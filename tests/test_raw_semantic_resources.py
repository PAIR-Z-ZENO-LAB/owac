"""Resource chart distinguishes file cache from anonymous memory and worker RSS."""

import json

from owac.runtime.raw_action_swanlab import host_memory_metrics


def test_memory_layers_and_actual_worker_fds(tmp_path):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.stat").write_text(f"anon {2**30}\nfile {8 * 2**30}\nfile_mapped {2**29}\n")
    (cg / "memory.current").write_text(str(9 * 2**30))
    (cg / "memory.events").write_text("oom_kill 0\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.json").write_text(json.dumps({"pid": 10}))
    proc = tmp_path / "proc"
    (proc / "10/task/10").mkdir(parents=True)
    (proc / "10/task/10/children").write_text("20")
    (proc / "20/fd").mkdir(parents=True)
    (proc / "20/fd/0").touch()
    (proc / "20/status").write_text("Name:\tpython\nVmRSS:\t524288 kB\n")
    result = host_memory_metrics(run, proc=proc, cgroup=cg)
    assert result["host/anonymous_memory_gib"] == 1
    assert result["host/file_cache_gib"] == 8
    assert result["host/worker0/rss_gib"] == 0.5
    assert result["host/worker0/open_fds"] == 1


def test_optimizer_cpu_counter_does_not_retain_checkpoint_storage(tmp_path):
    from types import SimpleNamespace

    import torch

    from owac.training.raw_semantic_train import detach_cpu_optimizer_state

    path = tmp_path / "state.pt"
    torch.save({"step": torch.tensor(1162.0)}, path)
    loaded = torch.load(path, mmap=True, weights_only=True)
    original = loaded["step"]
    optimizer = SimpleNamespace(state={0: {"step": original}})
    detach_cpu_optimizer_state(optimizer)
    assert optimizer.state[0]["step"].data_ptr() != original.data_ptr()
    assert optimizer.state[0]["step"].item() == 1162
