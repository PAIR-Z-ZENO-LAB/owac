"""Time-based causal indexing for recorded and live stereo-policy motion histories."""

import numpy as np


def causal_history_indices(times, lags, period):
    """Nearest sample to each past time, bounded by the current row and episode start."""
    times = np.asarray(times)
    before = times[:, None] - np.asarray(lags)[None] * period
    right = np.minimum(np.searchsorted(times, before), np.arange(len(times))[:, None])
    left = np.maximum(right - 1, 0)
    return np.where(abs(times[left] - before) <= abs(times[right] - before), left, right)
