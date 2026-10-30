import numpy as np
import torch

from owac.training.j5_action_data import action_chunk, geometric_augment


def test_official_action_quantiles_do_not_clip_and_repeat_terminal():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    result = action_chunk(actions, 1, {"q01": [0] * 7, "q99": [2] * 7})
    expected = (2 * actions[1:].astype(np.float64) / (2 + 1e-6) - 1).astype(np.float32)
    np.testing.assert_allclose(result[:2, :7], expected)
    np.testing.assert_allclose(result[2:, :7], np.repeat(expected[-1:], 8, axis=0))
    assert result[0, 0] > 1
    assert not result[:, 7:].any()


def test_raw_geometry_retains_linear_magnitude_without_rgb_clipping():
    image = torch.full((3, 256, 256), 2.0)
    transformed = geometric_augment(image, np.random.default_rng(4))
    torch.testing.assert_close(transformed[:, 64:192, 64:192], image[:, 64:192, 64:192])
