from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from owac.frameworks.raw_compact.fruit_placement import FruitPlacementPolicy
from owac.training.fruit_placement import placement_events, placement_rows


def test_expert_opens_before_pickup_and_again_over_basket(tmp_path):
    path = tmp_path / "episode.npz"
    t = np.arange(12, dtype=np.int64) * 100_000_000
    q = np.zeros((len(t), 8), dtype=np.float32)
    q[:, 7] = np.array([0, 0, 60, 80, 60, 10, 0, 0, 28, 55, 20, 0]) / 1000
    np.savez(path, observation_ns=t, command_ns=t, commands=q)
    src = SimpleNamespace(path=path, group_id="example")
    events = placement_events(src)
    assert (events.close, events.release, events.peak, events.opening_end) == (0.5, 0.9, 0.9, 1)
    assert (events.grasp_complete, events.opening_start) == (0.6, 0.7)
    pickup, carry, opening = placement_rows(events, 0.25)
    np.testing.assert_array_equal(pickup, [3, 4])
    np.testing.assert_array_equal(carry, [5, 6])
    np.testing.assert_array_equal(opening, [7, 8, 9, 10])
    q[8:, 7] = 0
    np.savez(path, observation_ns=t, command_ns=t, commands=q)
    with pytest.raises(ValueError, match="No release"):
        placement_events(src)


def test_each_owac_head_receives_its_own_rows_without_changing_actions():
    class Expert(nn.Module):
        def __init__(self, sign):
            super().__init__()
            self.sign = sign

        def sample(self, batch):
            assert "raw" not in batch
            return self.sign * batch["state"][:, None].expand(-1, 12, -1)

    class Gate(nn.Module):
        def forward(self, descriptor):
            flag = descriptor[:, -1]
            return torch.stack((-flag, flag), dim=1)

    model = FruitPlacementPolicy.__new__(FruitPlacementPolicy)
    nn.Module.__init__(model)
    model.cfg = {"model": {"action_horizon": 12}}
    model.pickup, model.carry, model.gate = Expert(1), Expert(-1), Gate()
    model.aperture = None
    model.grid = (4, 6)
    q = torch.arange(3 * 8).float().view(3, 8)
    q[:, 7] = torch.tensor([-1, 1, -1])
    batch = dict(state=q, features=torch.zeros(3, 2, 24, 8))
    expected = q.clone()
    expected[1] *= -1
    torch.testing.assert_close(model.sample(batch), expected[:, None].expand(-1, 12, -1))

    class Aperture(nn.Module):
        def geometry_features(self, state):
            return state

        def predict(self, descriptor, geometry):
            return geometry[:, :1].expand(-1, 12)

    model.aperture = Aperture()
    model.gate.mean = torch.zeros(104)
    model.gate.scale = torch.ones(104)
    expected[1, 7] = q[1, 0]
    torch.testing.assert_close(model.sample(batch), expected[:, None].expand(-1, 12, -1))
    only_carry = {k: v[1:2] for k, v in batch.items()}
    torch.testing.assert_close(model.sample(only_carry), expected[1:2, None].expand(-1, 12, -1))
