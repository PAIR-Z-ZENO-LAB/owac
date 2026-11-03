"""Joint/gripper measurements and the official pi05 discrete-state prompt."""

import numpy as np


def joint_gripper_from_observation(observation):
    return np.concatenate(
        (observation["robot0_joint_pos"], observation["robot0_gripper_qpos"])
    ).astype(np.float32)


def normalize_proprio(values, statistics):
    low, high = np.asarray(statistics["q01"]), np.asarray(statistics["q99"])
    state = np.zeros(32, np.float32)
    state[:9] = 2 * (np.asarray(values) - low) / (high - low + 1e-6) - 1
    return state


def tokenize_proprio(tokenizer, prompt, state, max_length=200):
    # Exact openpi PaligemmaTokenizer state binning and prompt syntax.
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
    ids = tokenizer.encode(text, add_bos=True)
    tokens, mask = np.zeros(max_length, np.int64), np.zeros(max_length, bool)
    tokens[: len(ids[:max_length])] = ids[:max_length]
    mask[: len(ids[:max_length])] = True
    return tokens, mask
