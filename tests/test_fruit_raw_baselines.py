import numpy as np
import pytest
import torch

from owac.training.fruit_baseline_cache import linear_planes
from owac.training.fruit_baseline_models import masked_error, rawvla_offsets
from owac.training.fruit_baseline_train import FruitInputs


def test_linear_sensor_input_keeps_low_light_counts_and_bggr_channel_order():
    pytest.importorskip("cv2")
    tile = np.array([[65, 67], [69, 73]], dtype=np.uint16)
    raw = np.tile(tile, (8, 8))
    result = linear_planes(raw, (4, 4))
    expected = np.array([9, 4, 1]) / 959
    np.testing.assert_allclose(result[:, 0, 0], expected, rtol=5e-4)
    assert np.all(result == result[:, :1, :1])


def test_rawvla_history_stops_at_episode_start_and_current_frame():
    bank = FruitInputs.__new__(FruitInputs)
    bank.method = "rawvla"
    bank.image_bank = torch.arange(80).reshape(40, 2, 1, 1, 1)
    bank.episode_start = torch.tensor([0] * 20 + [20] * 20)
    bank.state_norm = bank.state_q = torch.zeros(40, 8)
    bank.history = torch.as_tensor(rawvla_offsets(5))
    images = bank.observation(torch.tensor([20, 39]))["images"][..., 0, 0, 0]
    assert images.shape == (2, 2, 6, 6)
    assert (images[0, 0] == 40).all() and (images[0, 1] == 41).all()
    assert images[1].min() == 40 and images[1, 1, -1, -1] == 79


def test_flow_padding_has_zero_gradient():
    err = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]], requires_grad=True)
    loss = masked_error(err, torch.tensor([[True, False]]))
    loss.backward()
    assert loss.item() == 2 and torch.equal(err.grad[0, 1], torch.zeros(2))
