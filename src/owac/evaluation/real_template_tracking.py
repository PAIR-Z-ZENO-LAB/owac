"""Frozen first-frame box readout shared by all dense visual encoders."""

import torch
from torch.nn import functional as F


class TemplateReadout:
    """Match a fixed set of normalized patch descriptors across the full image."""

    def __init__(self, first, box, size=(640, 480)):
        self.size = size
        x0, y0, x1, y1 = box
        self.wh = (x1 - x0, y1 - y0)
        self.seed = tuple(box)
        # Sparse interior ellipse limits the contribution of rectangular background corners.
        offsets = [
            (u, v)
            for v in (-0.29, 0, 0.29)
            for u in (-0.33, -0.16, 0.0, 0.16, 0.33)
            if abs(v) < 0.2 or abs(u) < 0.3
        ]
        offsets = first.new_tensor(offsets)
        self.offsets = offsets * first.new_tensor(self.wh)
        self.factors = (1.0, 0.8, 1.25)
        w, h = size
        xy = self.offsets + first.new_tensor([(x0 + x1) / 2, (y0 + y1) / 2])
        grid = xy * first.new_tensor([2 / w, 2 / h]) - 1
        first = self.normalize(first)[None]
        templ = F.grid_sample(first, grid[None, None], align_corners=False, padding_mode="border")
        self.query = F.normalize(templ[0, :, 0].T, dim=-1)
        self.centres_h, self.centres_w = first.shape[-2] * 2, first.shape[-1] * 2
        yy = (torch.arange(self.centres_h, device=first.device) + 0.5) * h / self.centres_h
        xx = (torch.arange(self.centres_w, device=first.device) + 0.5) * w / self.centres_w
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        self.centres = torch.stack([xx, yy], -1)
        unit = first.new_tensor([2 / w, 2 / h])
        self.grids, self.valid = [], []
        for factor in self.factors:
            xy = self.centres[None] + self.offsets[:, None, None] * factor
            self.grids.append(xy * unit - 1)
            radius = first.new_tensor(self.wh) * factor / 2
            inside = (
                (self.centres - radius >= 0) & (self.centres + radius <= first.new_tensor(size))
            ).all(-1)
            self.valid.append(inside)

    @staticmethod
    def normalize(x):
        # Position-independent global descriptor mean is removed separately in each current frame.
        return F.normalize(x.float() - x.float().mean(dim=(-2, -1), keepdim=True), dim=-3)

    def score_fields(self, tokens):
        tokens = self.normalize(tokens)
        maps = torch.einsum("qd,bdyx->bqyx", self.query, tokens)
        b, q, h, w = maps.shape
        scores = []
        for grid, valid in zip(self.grids, self.valid, strict=True):
            sampled = F.grid_sample(
                maps.reshape(b * q, 1, h, w),
                grid.repeat(b, 1, 1, 1),
                align_corners=False,
                padding_mode="border",
            )
            sampled = sampled.reshape(b, q, self.centres_h, self.centres_w)
            score = sampled.mean(1)
            scores.append(score.masked_fill(~valid, -2.0))
        all_scores = torch.stack(scores, 1)
        return all_scores.max(1)

    def boxes_at(self, response, scale_id, index):
        y, x = index // self.centres_w, index % self.centres_w
        batch = torch.arange(len(index), device=response.device)
        centre = self.centres[y, x]
        side = (
            response.new_tensor(self.wh)
            * response.new_tensor(self.factors)[scale_id[batch, y, x]][:, None]
        )
        boxes = torch.cat([centre - side / 2, centre + side / 2], 1)
        return boxes, response, response[batch, y, x]

    def select_over_time(self, response, scales, previous, increments, diagonal):
        """Advance the predicted center in capture order, including across batch boundaries."""
        indices = []
        for score, dt in zip(response, increments, strict=True):
            radius = diagonal * 30 * dt
            in_window = ((self.centres - previous).abs() <= radius).all(dim=-1)
            allowed = in_window & (score > -2.0)
            if not allowed.any():
                raise ValueError("No valid target center inside the temporal search window")
            index = score.masked_fill(~allowed, -2.0).flatten().argmax()
            previous = self.centres.flatten(0, 1)[index]
            indices.append(index)
        boxes, _, peaks = self.boxes_at(response, scales, torch.stack(indices))
        return previous, boxes, peaks

    def __call__(self, tokens):
        response, scale_id = self.score_fields(tokens)
        return self.boxes_at(response, scale_id, response.flatten(1).argmax(1))
