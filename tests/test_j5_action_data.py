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


def test_joint_state_uses_current_frame_and_official_discrete_prompt(tmp_path):
    from types import SimpleNamespace

    from owac.training.j5_action_data import RawActionFrames
    from owac.training.j5_proprio import normalize_proprio, tokenize_proprio

    class Tokenizer:
        def encode(self, text, add_bos=False):
            return ([1] if add_bos else []) + list(text.encode())

    tokenizer = Tokenizer()
    statistics = {"q01": [-1] * 9, "q99": [1] * 9}
    state = normalize_proprio(np.linspace(-0.8, 0.8, 9), statistics)
    tokens, mask = tokenize_proprio(tokenizer, "pick_up\ncup", state, max_length=500)
    bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
    expected = f"Task: pick up cup, State: {' '.join(map(str, bins))};\nAction: "
    assert tokens[mask].tolist() == tokenizer.encode(expected, add_bos=True)
    assert not state[9:].any()

    # Different current joint positions must produce different policy conditioning.
    other, other_mask = tokenize_proprio(tokenizer, "pick_up\ncup", -state, max_length=500)
    assert tokens[mask].tolist() != other[other_mask].tolist()
    assert normalize_proprio(np.zeros(9), statistics).shape == (32,)

    # Dataset endpoint and sensor source: object/velocity state columns never enter.
    archive = tmp_path / "episode.npz"
    states = np.full((3, 92), 777.0, dtype=np.float32)
    states[:, 1:10] = np.arange(3)[:, None] * 0.1
    np.savez(
        archive,
        actions=np.zeros((3, 7)),
        states=states,
        metadata_json='{"task_description":"pick cup"}',
    )
    dataset = object.__new__(RawActionFrames)
    from collections import OrderedDict

    dataset.validation = True
    dataset.validation_indices = [(0, 1)]
    dataset.sources = [SimpleNamespace(path=archive, group_id="demo", lighting="Normal")]
    dataset.cfg = {"seed": 0}
    dataset.labels, dataset.prompts = OrderedDict(), {}
    dataset.tokenizer = tokenizer
    dataset.stats = {"actions": {"q01": [-1] * 7, "q99": [1] * 7}}
    dataset.proprio = {"state_columns": list(range(1, 10)), "statistics": statistics}
    dataset.frame = lambda *args: torch.zeros(3, 16, 16)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["state"], normalize_proprio(states[1, 1:10], statistics))
