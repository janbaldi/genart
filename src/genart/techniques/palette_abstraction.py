"""High t: recursive subdivision using only palette, weights and global statistics."""

import numpy as np

from genart.color import paper_tone
from genart.context import Context
from genart.genome import Genome
from genart.techniques._util import local_t, params_of

T_RANGE = (0.7, 1.0)
FULL_BLEED = False
PARAMS = {
    "depth": (7.0, 3.0, 10.0),  # max subdivision depth before density adjustment
    "gap": (0.006, 0.0, 0.02),  # paper gutter between blocks, fraction of short side
    "stripes": (0.3, 0.0, 1.0),  # probability a block is striped along the dominant orientation
}


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render a subdivided composition.

    Parameters
    ----------
    ctx : Context
    genome : Genome
    size : (h, w)

    Returns
    -------
    np.ndarray
        (h, w, 3) float32 linear RGB.
    """
    rng = np.random.default_rng(genome.seed)
    u = local_t(genome.t, T_RANGE)
    p = params_of(genome.params, PARAMS)
    h, w = size
    short = min(h, w)
    paper = paper_tone(ctx.palette)

    # Global statistics only.
    darkness = 1 - float(ctx.luminance.mean())
    c, a2 = ctx.coherence, 2 * ctx.angle.astype(np.float64)
    vx, vy = float((c * np.cos(a2)).sum()), float((c * np.sin(a2)).sum())
    theta = 0.5 * np.arctan2(vy, vx)
    strength = np.hypot(vx, vy) / (float(c.sum()) + 1e-9)

    # Subdivide in short-side units; splits cluster around focal points.
    H, W = h / short, w / short
    focals = rng.uniform(0.2, 0.8, (2, 2)) * (H, W)
    max_depth = int(p["depth"] + 2 * darkness)
    p_ycut = 0.5 + 0.4 * strength * np.cos(2 * theta)  # horizontal structure -> horizontal bands
    leaves = []
    stack = [(0.0, 0.0, H, W, 0)]
    while stack:
        y0, x0, y1, x1, d = stack.pop()
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        dist2 = float(((focals - (cy, cx)) ** 2).sum(1).min())
        split = d < 3 or rng.random() < 0.35 + 0.6 * np.exp(-dist2 / (2 * 0.22**2))
        if d >= max_depth or min(y1 - y0, x1 - x0) < 0.03 or not split:
            leaves.append((y0, x0, y1, x1))
            continue
        hh, ww = y1 - y0, x1 - x0
        ycut = hh > 1.8 * ww or (ww <= 1.8 * hh and rng.random() < p_ycut)
        f = rng.choice([0.382, 0.5, 0.618]) + rng.normal(0, 0.03)
        if ycut:
            m = y0 + f * hh
            stack += [(y0, x0, m, x1, d + 1), (m, x0, y1, x1, d + 1)]
        else:
            m = x0 + f * ww
            stack += [(y0, x0, y1, m, d + 1), (y0, m, y1, x1, d + 1)]

    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = paper
    k = len(ctx.palette)
    half_gap = 0.5 * p["gap"] * (0.5 + u)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for y0, x0, y1, x1 in leaves:
        small = (y1 - y0) * (x1 - x0) < 0.03
        is_paper = small and rng.random() < 0.2 + 0.4 * u
        ci, si = rng.choice(k, size=2, p=ctx.weights)
        striped = rng.random() < p["stripes"]
        period = rng.uniform(0.012, 0.035)
        if is_paper:
            continue
        ya, yb = round((y0 + half_gap) * short), round((y1 - half_gap) * short)
        xa, xb = round((x0 + half_gap) * short), round((x1 - half_gap) * short)
        if ya >= yb or xa >= xb:
            continue
        block = canvas[ya:yb, xa:xb]
        block[:] = ctx.palette[ci]
        if striped and si != ci:
            yy, xx = np.mgrid[ya:yb, xa:xb] / short
            s = (-xx * sin_t + yy * cos_t) / period
            # Duty cycle ramps across the block: a density gradient using only two inks.
            along = (xx * cos_t + yy * sin_t - (x0 * cos_t + y0 * sin_t)) / max(y1 - y0, x1 - x0)
            duty = np.clip(0.1 + 0.8 * np.abs(along), 0.1, 0.9)
            block[(s % 1.0) < duty] = ctx.palette[si]
    return canvas
