"""Training-run identity, checkpoint axes and closed-loop protocol separation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from owac.runtime import compact_swanlab


@pytest.fixture
def cfg():
    return {
        "swanlab": {"id": "training-id", "name": "training-name", "group": "C-raw-compact"},
        "research": {"selection_episodes": 200},
    }


def evaluation(tmp_path, step, successes, episodes=200, name=None):
    output = tmp_path / (name or f"development-{step:06d}")
    output.mkdir(exist_ok=True)
    result = {
        "checkpoint": f"/training/step-{step:06d}.pt",
        "native_exit": 0,
        "successes": successes,
        "episodes": episodes,
        "groups": {"libero_goal/Normal": {"successes": successes, "episodes": episodes}},
    }
    (output / "summary.json").write_text(json.dumps(result))
    return output, result


def test_train_and_evaluation_resume_the_same_named_run(cfg, monkeypatch, tmp_path):
    settings = Mock()
    sdk = SimpleNamespace(
        login=Mock(), merge_settings=Mock(), Settings=settings, init=Mock(), define_metric=Mock()
    )
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.setattr(Path, "read_text", lambda self: "test-key")
    compact_swanlab.init_run(cfg, tmp_path / "training")
    compact_swanlab.init_run(cfg, tmp_path / "evaluation", resume="must")
    for call in sdk.init.call_args_list:
        assert call.kwargs["id"] == "training-id"
        assert call.kwargs["experiment_name"] == "training-name"
        assert call.kwargs["group"] == "C-raw-compact"
        assert call.kwargs["job_type"] == "raw-proprio-language-flow"
    assert sdk.init.call_args.kwargs["resume"] == "must"
    sdk.define_metric.assert_any_call("success/*", section_name="05 闭环评估")


def test_completed_checkpoints_form_one_curve_in_training_step_order(cfg, tmp_path):
    evaluation(tmp_path, 10000, 152)
    evaluation(tmp_path, 5000, 128)
    evaluation(tmp_path, 1000, 1, episodes=40)
    (tmp_path / "development-020000").mkdir()  # Running evaluation has no summary yet.
    sdk = Mock()
    pending = compact_swanlab.log_results(sdk, cfg, compact_swanlab.completed_results(tmp_path))
    calls = sdk.log.call_args_list
    assert [call.kwargs["step"] for call in calls] == [1000, 5000, 10000]
    assert calls[0].args[0]["success/screening/overall"] == 1 / 40
    assert "success/overall" not in calls[0].args[0]
    assert [call.args[0]["success/overall"] for call in calls[1:]] == [0.64, 0.76]
    assert [call.args[0]["success/episodes"] for call in calls[1:]] == [200, 200]
    compact_swanlab.save_receipts(pending)
    sdk.reset_mock()
    assert compact_swanlab.log_results(sdk, cfg, compact_swanlab.completed_results(tmp_path)) == []
    sdk.log.assert_not_called()
    evaluation(tmp_path, 20000, 160)
    compact_swanlab.log_results(sdk, cfg, compact_swanlab.completed_results(tmp_path))
    assert sdk.log.call_count == 1
    assert sdk.log.call_args.kwargs["step"] == 20000


def test_final_does_not_overwrite_development_at_selected_checkpoint(cfg, tmp_path):
    development = evaluation(tmp_path, 10000, 152)
    final = evaluation(tmp_path, 10000, 1400, episodes=2000, name="independent-final")
    sdk = Mock()
    compact_swanlab.log_results(sdk, cfg, [development, final])
    first, second = sdk.log.call_args_list
    assert first.kwargs["step"] == second.kwargs["step"] == 10000
    assert first.args[0]["success/overall"] == 0.76
    assert second.args[0]["success/independent-final/overall"] == 0.7
    assert first.args[0].keys().isdisjoint(second.args[0])


def test_receipts_are_saved_only_after_finish_and_can_be_replayed(cfg, tmp_path, monkeypatch):
    item = evaluation(tmp_path, 5000, 128)
    sdk = Mock()
    monkeypatch.setattr(compact_swanlab, "init_run", lambda *args, **kwargs: sdk)
    sdk.finish.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        compact_swanlab.publish_results(cfg, [item], tmp_path / "swanlab")
    assert not (item[0] / "swanlab-training-link.json").exists()
    sdk.finish.side_effect = None
    compact_swanlab.publish_results(cfg, [item], tmp_path / "swanlab")
    assert (item[0] / "swanlab-training-link.json").exists()
    sdk.reset_mock()
    compact_swanlab.publish_results(cfg, [item], tmp_path / "swanlab", force=True)
    sdk.log.assert_called_once()


def test_explicit_checkpoint_metadata_wins_over_filename(cfg, tmp_path):
    output, result = evaluation(tmp_path, 5000, 128)
    result.update(checkpoint="/training/best.pt", step=4000, evaluation_kind="screening")
    step, metrics = compact_swanlab.result_metrics(cfg, result, output)
    assert step == 4000
    assert "success/screening/overall" in metrics


@pytest.mark.parametrize("field,value", [("native_exit", 1), ("episodes", 0)])
def test_failed_or_empty_evaluation_never_enters_curve(cfg, tmp_path, field, value):
    output, result = evaluation(tmp_path, 5000, 128)
    result[field] = value
    sdk = Mock()
    with pytest.raises(ValueError, match="completed native"):
        compact_swanlab.log_results(sdk, cfg, [(output, result)])
    sdk.log.assert_not_called()
