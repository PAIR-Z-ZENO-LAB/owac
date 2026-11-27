"""Keep the global minibatch and learning-rate schedule explicit across resumes."""

import math


def batch_partition(batch, world, rank):
    small, extra = divmod(batch, world)
    if small == 0:
        raise ValueError("Each distributed rank must receive at least one example")
    count = small + int(rank < extra)
    offset = rank * small + min(rank, extra)
    return offset, count


class BatchIndices:
    def __init__(self, start, end, batch, rank, world):
        self.start, self.end, self.batch = start, end, batch
        self.offset, self.count = batch_partition(batch, world, rank)
        if (end - start) % batch:
            raise ValueError("Training sample interval must contain complete global batches")

    def __iter__(self):
        for start in range(self.start, self.end, self.batch):
            yield from range(start + self.offset, start + self.offset + self.count)

    def __len__(self):
        return (self.end - self.start) // self.batch * self.count


def learning_rate_factor(step, training):
    continuation = training.get("continuation_schedule")
    if continuation:
        start = continuation["start_step"]
        ramp = continuation["ramp_updates"]
        initial, peak, final = (
            continuation[key] for key in ("initial_factor", "peak_factor", "final_factor")
        )
        if step <= start + ramp:
            return initial + (peak - initial) * max(step - start, 0) / ramp
        progress = (step - start - ramp) / (training["updates"] - start - ramp)
        return final + (peak - final) * (1 + math.cos(math.pi * progress)) / 2
    return min(step / training["warmup"], 1.0) * (
        0.1
        + 0.9
        * (
            1
            + math.cos(
                math.pi
                * max(step - training["warmup"], 0)
                / max(training["updates"] - training["warmup"], 1)
            )
        )
        / 2
    )
