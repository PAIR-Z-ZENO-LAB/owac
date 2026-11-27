import torch

from owac.evaluation.real_template_tracking import TemplateReadout


def test_fixed_query_follows_a_known_translation_and_is_independent_of_other_frames():
    rng = torch.Generator().manual_seed(17)
    first = torch.randn(48, 30, 40, generator=rng)
    shifted = first.roll(shifts=(3, 4), dims=(-2, -1))
    distractor = torch.randn(first.shape, generator=rng)
    tracker = TemplateReadout(first, [156, 132, 236, 228])
    boxes, responses, peaks = tracker(torch.stack([first, shifted, distractor]))
    shift = torch.tensor([64, 48, 64, 48])
    torch.testing.assert_close(boxes[1], boxes[0] + shift, atol=0, rtol=0)
    # Index the map in source x/y order to catch transposition of nonsquare feature grids.
    center_xy = (boxes[:, :2] + boxes[:, 2:]) / 2
    ij = ((center_xy - 4) / 8).long()
    torch.testing.assert_close(responses[torch.arange(3), ij[:, 1], ij[:, 0]], peaks)
    one_box, one_map, one_peak = tracker(shifted[None])
    torch.testing.assert_close(one_box[0], boxes[1])
    torch.testing.assert_close(one_map[0], responses[1])
    torch.testing.assert_close(one_peak[0], peaks[1])


def test_temporal_search_follows_the_previous_prediction_and_native_frame_interval():
    # Eight-pixel pitch. A stronger distractor is always present outside the allowed window.
    tracker = TemplateReadout(torch.ones(2, 6, 12), [16, 16, 32, 32], size=(192, 96))
    scores = torch.full((3, 12, 24), -2.0)
    scores[:, 5, 20] = 0.99
    scores[0, 5, 5] = 0.72
    scores[1, 5, 7] = 0.78
    scores[2, 5, 11] = 0.75
    scales = torch.zeros_like(scores, dtype=torch.long)
    initial = tracker.centres[5, 3]
    dt = [1 / 30, 1 / 30, 2 / 30]
    last, boxes, peaks = tracker.select_over_time(scores, scales, initial, dt, 16.0)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    torch.testing.assert_close(centers, tracker.centres[5, [5, 7, 11]])
    torch.testing.assert_close(peaks, torch.tensor([0.72, 0.78, 0.75]))
    torch.testing.assert_close(last, centers[-1])

    first_last, _, _ = tracker.select_over_time(scores[:1], scales[:1], initial, dt[:1], 16.0)
    _, continuation, _ = tracker.select_over_time(scores[1:], scales[1:], first_last, dt[1:], 16.0)
    torch.testing.assert_close(continuation, boxes[1:])
