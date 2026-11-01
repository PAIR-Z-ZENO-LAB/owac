import json
from types import SimpleNamespace

from owac.training.j5_monitoring import setup


def test_history_preserves_mse_and_tracks_each_run(tmp_path):
    history = tmp_path / "metrics.jsonl"
    history.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"step": 10, "metrics": {"train/flow32": 0.4, "val/sample_mae7": 0.3}},
                {"step": 20, "metrics": {"validation/action_mse": 0.2, "runtime/oom_kills": 0}},
            ]
        )
    )
    calls = []
    logger = SimpleNamespace(
        define_metric=lambda *a, **kw: None, log=lambda metrics, step: calls.append((step, metrics))
    )
    cfg = {"swanlab": {"id": "first"}, "monitoring": {"history": str(history)}}
    setup(logger, cfg, tmp_path)
    assert calls == [
        (10, {"objective/train_flow32": 0.4}),
        (20, {"validation/action_mse": 0.2, "runtime/oom_kills": 0}),
    ]
    setup(logger, cfg, tmp_path)
    assert len(calls) == 2
    cfg["swanlab"]["id"] = "continuation"
    setup(logger, cfg, tmp_path)
    assert calls[:2] == calls[2:]
