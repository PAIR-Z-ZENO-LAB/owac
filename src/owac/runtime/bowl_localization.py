"""Locate the three bowl rims in the fixed overhead camera's native Bayer samples."""

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


def raw_grid(image):
    """Integrate native photosites in 4x4 blocks. Coordinates are in this 160x120 grid."""
    image = np.asarray(image)
    if image.shape == (120, 160):
        return image.astype(np.float32, copy=False)
    if image.shape != (480, 640):
        raise ValueError("U2 bowl localization expects 640x480 BG10")
    return image.reshape(120, 4, 160, 4).mean((1, 3), dtype=np.float32)


def texture(grid):
    return gaussian_filter(grid, 0.7) - gaussian_filter(grid, 5)


def rim_texture(grid):
    # Normalize locally in native intensity so a nearby bright white bowl cannot
    # displace the apparent dark rim when only the black bowl moves.
    smooth = gaussian_filter(np.sqrt(np.maximum(grid - 61, 0)), 0.7)
    detail = smooth - gaussian_filter(smooth, 2.5)
    scale = gaussian_filter(abs(detail), 2.5) + 0.15 * np.median(abs(detail)) + 0.001
    return detail / scale


def exposure(grid):
    # Three background strips distinguish the moving lamp from uniform low exposure.
    values = [grid[3:22, 48:140], grid[31:90, 20:38], grid[32:80, 126:146]]
    return np.log1p(np.maximum([np.mean(v) - 63.0 for v in values], 0.0))


class BowlLocator:
    def __init__(self, config):
        self.banks = config["banks"]
        self.exposures = np.array([r["exposure"] for r in self.banks])
        radius = config["patch_radius"]
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        self.mask = (x * x + y * y <= radius**2).astype(np.float32)
        self.camera = np.asarray(config["camera"])
        self.bounds = config["table_bounds"]
        self.r = radius
        self.shape = tuple(next_fast_len(n + 2 * radius) for n in (120, 160))
        templates = np.asarray([r["templates"] for r in self.banks], np.float32)
        self.filters = rfft2(templates[..., ::-1, ::-1], s=self.shape, axes=(-2, -1))
        self.floors = np.asarray([r["min_score"] for r in self.banks])

    def locate(self, raw, objects=(0, 1, 2)):
        grid = raw_grid(raw)
        # Match the complete triplet under one training exposure, using current RAW appearance.
        distance = np.linalg.norm(self.exposures - exposure(grid), axis=1)
        group = self.banks[int(np.argmin(distance))]["name"].split("-")[0]
        banks = np.asarray(
            [i for i, b in enumerate(self.banks) if b["name"].split("-")[0] == group]
        )
        count = len(banks)
        feat = rim_texture(grid)
        total = fftconvolve(feat, self.mask, mode="same")
        power = fftconvolve(feat**2, self.mask, mode="same") - total**2 / self.mask.sum()
        denom = np.sqrt(np.maximum(power, np.finfo(np.float32).eps))
        x0, y0, x1, y1 = self.bounds
        spectrum = rfft2(feat, s=self.shape)
        cols = list(objects)
        top = irfft2(self.filters[banks][:, cols] * spectrum, s=self.shape, axes=(-2, -1))
        r = self.r
        corr = top[..., r : r + 120, r : r + 160] / denom
        roi = corr[..., y0:y1, x0:x1]
        rows, cols_ix = np.unravel_index(
            roi.reshape(count, len(cols), -1).argmax(-1), roi.shape[-2:]
        )
        rows += y0
        cols_ix += x0
        scores = corr[np.arange(count)[:, None], np.arange(len(cols)), rows, cols_ix]
        points = np.stack((cols_ix, rows), axis=-1)
        distinct = np.ones(count, dtype=bool)
        for i in range(len(cols)):
            for j in range(i):
                distinct &= np.linalg.norm(points[:, i] - points[:, j], axis=-1) > 2 * (r - 2)
        k = int(np.argmax(scores.mean(1) + distinct))
        refined = []
        for j, (x, y) in enumerate(points[k]):
            c = corr[k, j]
            delta = []
            for a, b in ((c[y, x - 1], c[y, x + 1]), (c[y - 1, x], c[y + 1, x])):
                d = 2 * (2 * c[y, x] - a - b)
                delta.append(float(np.clip((b - a) / d, -0.5, 0.5)) if d > 0 else 0.0)
            refined.append(np.array([x, y]) + delta)
        return np.array(refined), scores[k], self.floors[banks[k], cols]

    def table_xy(self, pixel, z):
        """Back-project a detected center onto the demonstrated contact plane, in robot metres."""
        u, v = pixel
        p = self.camera
        eq = p[:2] - np.array([[u], [v]]) * p[2]
        return np.linalg.solve(eq[:, :2], -eq[:, 2] * z - eq[:, 3])


class StackAppearance:
    """Compare an empty white bowl, a patterned top, and a black top at the same location."""

    def __init__(self, templates, exposures, cutoffs, radius):
        self.templates = np.asarray(templates, dtype=np.float32)
        self.exposures = np.asarray(exposures)
        self.cutoffs = np.asarray(cutoffs)
        self.radius = radius
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        self.mask = (yy**2 + xx**2 <= radius**2).astype("f4")

    def read(self, raw, prior, search_radius=None):
        from numpy.lib.stride_tricks import sliding_window_view

        g = raw_grid(raw)
        feat = texture(g)
        r = self.radius
        x, y = np.rint(prior).astype(int)
        reach = r if search_radius is None else int(np.ceil(search_radius))
        x0, x1 = max(r, x - reach), min(g.shape[1] - r - 1, x + reach)
        y0, y1 = max(r, y - reach), min(g.shape[0] - r - 1, y + reach)
        win = sliding_window_view(feat[y0 - r : y1 + r + 1, x0 - r : x1 + r + 1], self.mask.shape)
        win = (win * self.mask).reshape(win.shape[0], win.shape[1], -1)
        win -= win.sum(-1, keepdims=True) / self.mask.sum() * self.mask.ravel()
        win /= np.maximum(np.linalg.norm(win, axis=-1, keepdims=True), 1.0e-8)
        light = exposure(g)
        scores = []
        points = []
        for label in range(3):
            # The lighting can change within an episode. Use the exposure measured at this stage.
            near = np.argsort(np.linalg.norm(self.exposures[:, label] - light, axis=1))[:8]
            t = self.templates[near, label].reshape(len(near), -1)
            corr = (win @ t.T).max(-1)
            if search_radius is not None:
                yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
                corr = np.where(
                    (xx - prior[0]) ** 2 + (yy - prior[1]) ** 2 <= search_radius**2, corr, -np.inf
                )
            iy, ix = np.unravel_index(corr.argmax(), corr.shape)
            scores.append(float(corr[iy, ix]))
            points.append([int(x0 + ix), int(y0 + iy)])
        label = int(np.argmax(scores))
        if scores[label] < self.cutoffs[label]:
            label = -1
        return label, np.asarray(points, dtype=float), scores
