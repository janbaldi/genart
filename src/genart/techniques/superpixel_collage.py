"""Low t: SLIC regions posterized toward the palette, partly shuffled and displaced."""

import numpy as np
from scipy import ndimage

from genart.color import nearest_palette_index, paper_tone
from genart.context import Context
from genart.genome import Genome
from genart.techniques._util import local_t, params_of

T_RANGE = (0.0, 0.35)
FULL_BLEED = False
PARAMS = {
    "displacement": (0.06, 0.0, 0.15),  # max fragment offset, fraction of short side
    "focal_sigma": (0.16, 0.08, 0.6),  # spread of the area where shuffling clusters
    "posterize": (0.6, 0.0, 1.0),  # pull toward the palette at the low end of the range
}


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render a collage of superpixel fragments.

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

    labels = ctx.sample("labels", size, order=0)
    image = ctx.sample("image", size)
    n = int(labels.max()) + 1
    flat = labels.ravel()
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    means = (
        np.stack([np.bincount(flat, image[..., c].ravel(), minlength=n) for c in range(3)], -1)
        / counts[:, None]
    )

    # Posterize: region mean pulled toward its nearest palette color.
    q = p["posterize"] + (1 - p["posterize"]) * u
    snap = ctx.palette[nearest_palette_index(means, ctx.palette)]
    region_col = ((1 - q) * means + q * snap).astype(np.float32)

    # Pick regions to move, clustered around a focal point.
    slices = ndimage.find_objects(labels + 1)
    centers = np.array(
        [
            [(s[0].start + s[0].stop) / 2 / h, (s[1].start + s[1].stop) / 2 / w]
            if s is not None
            else [-9.0, -9.0]
            for s in slices
        ]
        + [[-9.0, -9.0]] * (n - len(slices))
    )
    focal = rng.uniform(0.3, 0.7, 2)
    d2 = ((centers - focal) ** 2).sum(1)
    weight = np.exp(-d2 / (2 * p["focal_sigma"] ** 2))
    frac = 0.06 + 0.4 * u
    prob = np.minimum(1.0, frac * n * weight / weight.sum())
    selected = np.flatnonzero(rng.random(n) < prob)

    # Shuffle colors among the moved regions.
    region_col[selected] = region_col[rng.permutation(selected)]

    detail = 0.2 * (1 - u)  # some of the photo survives inside each region at low t
    src = region_col[labels] * (1 - detail) + image * detail
    canvas = src.copy()

    holes = selected[rng.random(len(selected)) < 0.3 + 0.6 * u]
    for i in holes:
        sl = slices[i]
        canvas[sl][labels[sl] == i] = paper

    angles = rng.uniform(0, 2 * np.pi, len(selected))
    mags = p["displacement"] * short * (0.4 + 0.6 * u) * rng.uniform(0.3, 1.0, len(selected))
    order = np.argsort(-d2[selected])  # far from focal first, near focal on top
    for j in order:
        i = selected[j]
        sl = slices[i]
        dy, dx = round(mags[j] * np.sin(angles[j])), round(mags[j] * np.cos(angles[j]))
        _paste(canvas, src[sl], labels[sl] == i, sl[0].start + dy, sl[1].start + dx)
    return np.clip(canvas, 0, 1).astype(np.float32)


def _paste(dst: np.ndarray, patch: np.ndarray, mask: np.ndarray, y0: int, x0: int) -> None:
    """Copy ``patch[mask]`` into ``dst`` at offset (y0, x0), clipped to bounds."""
    h, w = dst.shape[:2]
    ph, pw = mask.shape
    ya, xa = max(0, -y0), max(0, -x0)
    yb, xb = min(ph, h - y0), min(pw, w - x0)
    if ya >= yb or xa >= xb:
        return
    m = mask[ya:yb, xa:xb]
    dst[y0 + ya : y0 + yb, x0 + xa : x0 + xb][m] = patch[ya:yb, xa:xb][m]
