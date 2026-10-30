"""Check correspondence, masking, and episode-local history rather than loss thresholds."""

import torch

from owac.training.raw_relations import (
    diversity_loss,
    relation_loss,
    temporal_loss,
    temporal_targets,
)


def test_relations_preserve_geometry_and_ignore_masked_positions():
    target = torch.eye(8)[None]
    masks = torch.zeros(1, 8, dtype=torch.bool)
    masks[:, -1] = True
    student = target.clone()
    student[:, -1] = 10
    torch.testing.assert_close(relation_loss(student, target, masks), torch.tensor(0.0))
    student[:, 1] = student[:, 0]
    assert relation_loss(student, target, masks) > 0


def test_temporal_match_tracks_permuted_content_and_downweights_ambiguity():
    current = torch.eye(16)[None]
    permutation = torch.arange(16).reshape(4, 4).flip(-1).flatten()
    previous = current[:, permutation]
    target, confidence, displacement = temporal_targets(current, previous, 3, 0.01)
    assert torch.equal(target.argmax(-1), torch.arange(16)[None])
    assert displacement > 0
    _, ambiguous, _ = temporal_targets(torch.ones_like(current), torch.ones_like(current), 3, 0.01)
    assert confidence.mean() > ambiguous.mean()
    student = torch.randn_like(current, requires_grad=True)
    masks = torch.zeros(1, 16, dtype=torch.bool)
    masks[:, :4] = True
    temporal_loss(student, target, confidence, masks).backward()
    assert student.grad[:, :4].count_nonzero() == 0
    assert student.grad[:, 4:].abs().sum() > 0


def test_variance_penalizes_collapsed_actual_tokens():
    masks = torch.zeros(8, 256, dtype=torch.bool)
    collapsed, _ = diversity_loss(torch.ones(8, 256, 32), masks)
    spread, covariance = diversity_loss(torch.randn(8, 256, 32), masks)
    assert collapsed > spread
    assert covariance >= 0


def test_history_is_previous_same_camera_frame_and_same_crop(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from owac.training.raw_ssl_data import RawViews

    cfg = json.loads(Path("configs/experiments/j5_raw_relations.json").read_text())
    raw = np.stack([np.full((48, 64, 3), k, dtype=np.uint8) for k in (30, 60)])
    path = tmp_path / "episode.npz"
    np.savez(path, agentview_raw_uint8=raw, wrist_raw_uint8=raw)
    source = SimpleNamespace(path=path, domain="libero", lighting="Normal", num_frames=2)
    dataset = RawViews([source], cfg)
    views = dataset[20]
    torch.testing.assert_close(views[3] * 2, views[0][0])
    assert views[4]
    for first, second in zip(views[:4], dataset[20][:4], strict=True):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    dataset.close()
