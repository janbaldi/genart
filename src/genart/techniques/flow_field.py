"""Mid t: tapered streamlines following the orientation field, over a fading underlay."""

import numpy as np
from numba import njit
from scipy import ndimage

from genart.color import nearest_palette_index, paper_tone
from genart.context import Context, fit
from genart.genome import Genome
from genart.techniques._util import local_t, params_of

T_RANGE = (0.35, 0.7)
FULL_BLEED = False
PARAMS = {
    "n_strokes": (2500.0, 500.0, 8000.0),
    "stroke_width": (0.0035, 0.001, 0.01),  # fraction of short side
    "length": (0.12, 0.03, 0.4),  # max streamline length, fraction of short side
    "turbulence": (0.15, 0.0, 1.5),  # noise mixed into the field where coherence is low
}
GRID = 256  # long side of the tracing grid; geometry lives here, independent of output size
STEP = 0.5  # tracing step in grid cells


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render streamlines.

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
    gh, gw = (GRID, max(2, round(GRID * w / h))) if h >= w else (max(2, round(GRID * h / w)), GRID)
    gshort = min(gh, gw)

    # Doubled-angle vectors interpolate cleanly across the 0/pi ambiguity.
    v = np.stack([np.cos(2 * ctx.angle), np.sin(2 * ctx.angle)], -1)
    v = ndimage.gaussian_filter(fit(v, (gh, gw)), (1.5, 1.5, 0))
    coh = fit(ctx.coherence, (gh, gw))
    noise = ndimage.gaussian_filter(rng.normal(size=(gh, gw)), gshort / 12)
    noise_ang = 2 * np.pi * noise / (noise.std() + 1e-9)
    vn = np.stack([np.cos(noise_ang), np.sin(noise_ang)], -1)
    v = v + p["turbulence"] * (1 - coh)[..., None] * vn

    # Seeds cluster in dark areas and around a focal point.
    lum = fit(ctx.luminance, (gh, gw))
    yy, xx = np.mgrid[0:gh, 0:gw] / gshort
    focal = rng.uniform(0.3, 0.7, 2) * (gh / gshort, gw / gshort)
    focus = np.exp(-((yy - focal[0]) ** 2 + (xx - focal[1]) ** 2) / (2 * 0.25**2))
    density = (1 - lum) ** 2.5 * (0.15 + 1.5 * focus) + 0.002
    density /= density.sum()
    n = int(p["n_strokes"] * (0.6 + 0.8 * float((1 - lum).mean())))
    cells = rng.choice(gh * gw, size=n, p=density.ravel())
    seeds = np.stack([cells // gw, cells % gw], -1) + rng.random((n, 2)) - 0.5
    signs = rng.choice(np.array([-1.0, 1.0]), n)
    steps = (rng.uniform(0.3, 1.0, n) * p["length"] * gshort / STEP).astype(np.int64) + 2
    pts, counts = _trace(seeds, signs, steps, v[..., 0].copy(), v[..., 1].copy(), STEP)

    # Color: mostly the palette color under the seed at low t, weighted palette at high t.
    k = len(ctx.palette)
    under = nearest_palette_index(fit(ctx.image, (gh, gw)), ctx.palette)
    sy = np.clip(seeds[:, 0].round().astype(int), 0, gh - 1)
    sx = np.clip(seeds[:, 1].round().astype(int), 0, gw - 1)
    from_image = rng.random(n) < 0.85 - 0.6 * u
    idx = np.where(from_image, under[sy, sx], rng.choice(k, size=n, p=ctx.weights))
    colors = ctx.palette[idx].astype(np.float64)
    radii = 0.5 * p["stroke_width"] * min(h, w) * rng.uniform(0.6, 1.4, n)

    paper = paper_tone(ctx.palette)
    # Underlay: the reference posterized to the palette, softened, fading into paper.
    uh, uw = 2 * gh, 2 * gw
    snapped = ctx.palette[nearest_palette_index(fit(ctx.image, (uh, uw)), ctx.palette)]
    snapped = ndimage.gaussian_filter(snapped, (uh / 200, uw / 200, 0))
    m = 0.45 + 0.55 * u
    canvas = (fit(snapped.astype(np.float32), size) * (1 - m) + paper * m).astype(np.float32)
    _draw(canvas, pts, counts, radii, colors, h / gh, w / gw)
    return canvas


@njit(cache=True)
def _bilerp(a, y, x):
    h, w = a.shape
    y0 = min(int(y), h - 2)
    x0 = min(int(x), w - 2)
    fy, fx = y - y0, x - x0
    return (
        a[y0, x0] * (1 - fy) * (1 - fx)
        + a[y0, x0 + 1] * (1 - fy) * fx
        + a[y0 + 1, x0] * fy * (1 - fx)
        + a[y0 + 1, x0 + 1] * fy * fx
    )


@njit(cache=True)
def _trace(seeds, signs, steps, vx, vy, step):
    """Trace streamlines in grid coordinates; returns points (n, max_steps, 2) and counts."""
    gh, gw = vx.shape
    n = len(seeds)
    pts = np.zeros((n, steps.max(), 2))
    counts = np.zeros(n, np.int64)
    for i in range(n):
        y, x = seeds[i, 0], seeds[i, 1]
        py, px = 0.0, 0.0
        for j in range(steps[i]):
            if y < 0 or y > gh - 1 or x < 0 or x > gw - 1:
                break
            pts[i, j, 0] = y
            pts[i, j, 1] = x
            counts[i] = j + 1
            a = 0.5 * np.arctan2(_bilerp(vy, y, x), _bilerp(vx, y, x))
            dx, dy = np.cos(a), np.sin(a)
            if j == 0:
                dx, dy = dx * signs[i], dy * signs[i]
            elif dx * px + dy * py < 0:
                dx, dy = -dx, -dy
            px, py = dx, dy
            x += dx * step
            y += dy * step
    return pts, counts


@njit(cache=True)
def _draw(canvas, pts, counts, radii, colors, sy, sx):
    """Stamp antialiased tapered discs along each streamline, in pixel space."""
    H, W = canvas.shape[0], canvas.shape[1]
    for i in range(len(counts)):
        n = counts[i]
        if n < 2:
            continue
        r0 = radii[i]
        for j in range(n - 1):
            y0 = (pts[i, j, 0] + 0.5) * sy - 0.5
            x0 = (pts[i, j, 1] + 0.5) * sx - 0.5
            y1 = (pts[i, j + 1, 0] + 0.5) * sy - 0.5
            x1 = (pts[i, j + 1, 1] + 0.5) * sx - 0.5
            m = int(np.hypot(y1 - y0, x1 - x0) / max(0.35 * r0, 0.3)) + 1
            for k in range(m):
                f = k / m
                s = (j + f) / (n - 1)
                r = r0 * (0.15 + 0.85 * np.sin(np.pi * s) ** 0.5)
                cy = y0 + (y1 - y0) * f
                cx = x0 + (x1 - x0) * f
                for yy in range(max(0, int(cy - r - 1)), min(H, int(cy + r + 2))):
                    for xx in range(max(0, int(cx - r - 1)), min(W, int(cx + r + 2))):
                        a = r + 0.5 - np.hypot(yy - cy, xx - cx)
                        if a <= 0:
                            continue
                        a = min(a, 1.0) * 0.9
                        for c in range(3):
                            canvas[yy, xx, c] = canvas[yy, xx, c] * (1 - a) + colors[i, c] * a
