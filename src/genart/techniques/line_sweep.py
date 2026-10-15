"""Mid-high t: many thin segments between two moving points p1(u), p2(u).

Overlap density forms implied surfaces. A curve *source* says where the endpoints travel;
a *composition* says where on the sheet the sweep goes. Both are genome choices.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import ndimage
from skimage import measure

from genart.color import linear_to_oklab
from genart.context import Context, fit
from genart.genome import Genome
from genart.render import accumulate_lines, tone_map
from genart.techniques._util import local_t, params_of
from genart.techniques.superpixel_collage import select_fragments

T_RANGE = (0.45, 0.95)
FULL_BLEED = True  # each composition sets its own margin
PARAMS = {
    "lines": (4000.0, 1000.0, 10000.0),  # segments per 1000 px of short side
    "k_gain": (1.3, 0.6, 2.5),  # multiplies the auto-calibrated tone-map k
    "lag": (0.25, 0.02, 0.5),  # how far p2 trails p1 along the curve
    "advect": (0.15, 0.0, 0.4),  # endpoint travel through the orientation field
    "seg_len": (0.04, 0.01, 0.1),  # density source segment length, fraction of the box
    "ink2": (0.08, 0.0, 0.15),  # share of segments in the second ink
    "tiles": (3.0, 2.0, 5.0),  # columns of the tiled composition
    "damping": (1.2, 0.3, 3.0),  # harmonograph decay
}
CHOICES = {
    "source": ("density", "advected", "contour", "parametric"),
    "composition": ("radial", "tiled", "masked"),
    "curve": ("circle", "spiral", "lissajous", "harmonograph"),
    "polar": ("off", "on"),
    "mask_mode": ("clip", "cross"),
}
GRID = 256  # analysis grid long side; geometry never depends on output size
TAU = 2 * np.pi
HARMONO_PERIODS = 8


def init_params(t: float, rng: np.random.Generator) -> dict[str, str]:
    """Categorical choices for a new genome: the source follows t, the rest is drawn."""
    u = local_t(t, T_RANGE)
    choices = {k: str(rng.choice(v)) for k, v in CHOICES.items()}
    choices["source"] = CHOICES["source"][min(3, int(u * 4))]
    return choices


@dataclass(frozen=True)
class Region:
    """Reference fields on the analysis grid of one composition cell."""

    lum: np.ndarray
    v: np.ndarray  # doubled-angle orientation vectors (gh, gw, 2)
    coh: np.ndarray


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render a line sweep.

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
    rng = np.random.default_rng(genome.seed)  # size-independent draws only
    p = params_of(genome.params, PARAMS) | {
        k: genome.params.get(k, v) for k, v in init_params(genome.t, rng).items()
    }
    p |= {k: v for k, v in genome.params.items() if k not in p}  # freq/phase overrides
    h, w = size
    short = min(h, w)
    ground, ink, ink2, fill = _colors(ctx.palette)
    ink2_start = rng.random()
    # Seed-driven variation on top of the reference-derived structure (not stored params).
    p |= {
        "_phase": rng.uniform(0, TAU),
        "_variant": int(rng.integers(4)),
        "_focal": rng.uniform(-0.2, 0.2, 2),
    }
    mask = None

    if p["composition"] == "radial":
        side = 0.70 * short  # 15% margin on the short side
        cells = [((h - side) / 2, (w - side) / 2, side, side, _region(ctx, (GRID, GRID)), p)]
    elif p["composition"] == "tiled":
        cells = _tiles(ctx, h, w, p)
    else:
        m = 0.08 * short
        bh, bw = h - 2 * m, w - 2 * m
        gshape = _grid((bh, bw))
        cells = [(m, m, bh, bw, _region(ctx, gshape), p)]
        _, _, sel = select_fragments(fit(ctx.labels, gshape, order=0), rng, 0.25, 0.2)
        # Smooth the fragment mask at analysis resolution so edges stay curved at print size.
        soft = np.isin(fit(ctx.labels, gshape, order=0), sel).astype(np.float32)
        soft = ndimage.gaussian_filter(soft, 1.0)
        mask = np.zeros((h, w), bool)
        box = (round(m), round(m), h - round(m), w - round(m))
        mask[box[0] : box[2], box[1] : box[3]] = fit(soft, (box[2] - box[0], box[3] - box[1])) > 0.5

    if p["source"] == "density":  # fade tone out toward cell edges: no hard photo crops
        disc = p["composition"] == "radial"
        focal = p["_focal"] if disc else (0.0, 0.0)
        cells = [(*c[:4], _window(c[4], disc, focal), c[5]) for c in cells]
    n_total = p["lines"] * short / 1000
    p1s, p2s, groups = [], [], []
    for i, (oy, ox, bh, bw, reg, pc) in enumerate(cells):
        n = max(8, round(n_total * pc.get("share", 1.0)))
        cell_rng = np.random.default_rng([genome.seed, i + 1])
        a, b = SOURCES[p["source"]](reg, n, pc, cell_rng)
        scale = np.array([bh, bw])
        p1s.append(a * scale + (oy, ox))
        p2s.append(b * scale + (oy, ox))
        # Second ink: a soft band in u (triangular, mean share = ink2) so it has no hard seam.
        u = (np.arange(len(a)) + 0.5) / len(a)
        d = np.abs(((u - ink2_start) % 1.0) - pc["ink2"])
        prob = np.clip(1 - d / max(pc["ink2"], 1e-6), 0, 1)
        groups.append((cell_rng.random(len(a)) < prob).astype(np.int64))
    p1, p2, g = np.concatenate(p1s), np.concatenate(p2s), np.concatenate(groups)

    ss = 3 if h * w <= 4_000_000 else 2
    dens = accumulate_lines(p1, p2, size, ss, g)
    if mask is not None and p["mask_mode"] == "clip":
        dens *= mask
    cover, _ = tone_map(dens, p["k_gain"])
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = ground
    for layer, color in reversed(list(zip(cover, (ink, ink2), strict=False))):
        canvas += (color - canvas) * layer[..., None]
    if mask is not None and p["mask_mode"] == "cross":
        a = 1 - np.prod(1 - cover, axis=0)
        form = ndimage.gaussian_filter(a, 0.02 * short) > 0.02  # fragments only where the form is
        tint = ground + (fill - ground) * 0.6
        inside = tint + (ground - tint) * a[..., None]  # lines turn to ground inside fragments
        sel = mask & form
        canvas[sel] = inside[sel]
    return canvas


def _colors(palette: np.ndarray) -> tuple[np.ndarray, ...]:
    """Ground (light, low chroma), ink (darkest), second ink (most chromatic), fill (mid)."""
    lab = linear_to_oklab(palette)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    gi = int(np.argmax(lab[:, 0] - 3 * chroma))
    rest = [i for i in range(len(palette)) if i not in (0, gi)] or [0]
    i2 = max(rest, key=lambda i: chroma[i])
    fi = min(rest, key=lambda i: abs(lab[i, 0] - 0.75))
    return palette[gi], palette[0], palette[i2], palette[fi]


def _grid(box: tuple[float, float]) -> tuple[int, int]:
    bh, bw = box
    return (
        (GRID, max(4, round(GRID * bw / bh))) if bh >= bw else (max(4, round(GRID * bh / bw)), GRID)
    )


def _region(ctx: Context, shape: tuple[int, int]) -> Region:
    v = np.stack([np.cos(2 * ctx.angle), np.sin(2 * ctx.angle)], -1)
    v = ndimage.gaussian_filter(fit(v, shape), (2, 2, 0))
    return Region(fit(ctx.luminance, shape), v, fit(ctx.coherence, shape))


def _window(reg: Region, disc: bool, focal: tuple[float, float]) -> Region:
    """Lighten the region toward its edges (disc or rounded rectangle) so density fades out."""
    gh, gw = reg.lum.shape
    yy, xx = np.mgrid[-1 : 1 : gh * 1j, -1 : 1 : gw * 1j]
    yy, xx = yy - focal[0], xx - focal[1]
    if disc:
        win = np.clip((1.0 - np.hypot(yy, xx)) / 0.35, 0, 1) ** 1.5
    else:
        win = np.clip((1.0 - np.maximum(np.abs(yy), np.abs(xx)) ** 4) / 0.4, 0, 1)
    return Region(1 - (1 - reg.lum) * win, reg.v, reg.coh)


def _tiles(ctx: Context, h: int, w: int, p: dict) -> list:
    """Grid of cells; each tile's amplitude, phase and line share follow its reference region."""
    short = min(h, w)
    m, gut = 0.08 * short, 0.025 * short
    ih, iw = h - 2 * m, w - 2 * m
    cols = round(p["tiles"])
    rows = max(1, round(cols * ih / iw))
    th, tw = (ih - (rows - 1) * gut) / rows, (iw - (cols - 1) * gut) / cols
    gh, gw = _grid((ih, iw))
    full = _region(ctx, (gh, gw))
    regs = []
    for r in range(rows):
        for c in range(cols):
            sl = (
                slice(r * gh // rows, (r + 1) * gh // rows),
                slice(c * gw // cols, (c + 1) * gw // cols),
            )
            regs.append((r, c, Region(full.lum[sl], full.v[sl], full.coh[sl])))
    dark = np.array([1 - reg.lum.mean() for _, _, reg in regs])
    rel = (dark - dark.min()) / (np.ptp(dark) + 1e-9)
    cells = []
    for (r, c, reg), d in zip(regs, rel, strict=True):
        vx, vy = reg.v[..., 0].mean(), reg.v[..., 1].mean()
        pc = p | {
            "lag": p["lag"] + 0.5 * np.arctan2(vy, vx) / np.pi,
            "amp": 0.4 + 0.6 * d,
            "mult": 2 + (int(d * 3.99) + p["_variant"]) % 4,
            "share": (0.4 + d) / (0.4 + rel.mean()) / len(regs),
        }
        cells.append((m + r * (th + gut), m + c * (tw + gut), th, tw, reg, pc))
    return cells


# ---------------------------------------------------------------- curve sources
# Each returns p1, p2 as (n, 2) arrays of (y, x) in unit cell coordinates.


def _freqs(reg: Region, p: dict) -> tuple[int, int, float, float]:
    """Frequency ratio from luminance FFT peaks, phases from orientation-histogram modes."""
    gh, gw = reg.lum.shape
    lum = (reg.lum - reg.lum.mean()) * np.outer(np.hanning(gh), np.hanning(gw))
    spec = np.abs(np.fft.rfft2(lum))
    r = np.hypot(np.fft.fftfreq(gh)[:, None] * gh, np.fft.rfftfreq(gw)[None, :] * gw)
    ri = r.round().astype(int).ravel()
    prof = np.bincount(ri, spec.ravel()) / np.maximum(np.bincount(ri), 1)
    prof = (prof * np.arange(len(prof)))[: min(len(prof), 24)]  # whiten the 1/f falloff
    prof[:2] = 0
    f1, f2 = np.argsort(prof)[-2:] if len(prof) > 3 else (2, 3)
    ratio = Fraction(int(max(f1, 1)), int(max(f2, 1))).limit_denominator(6)
    a, b = np.clip([ratio.numerator, ratio.denominator], 1, 7)
    if a == b:
        b = a + 1
    ang = 0.5 * np.arctan2(reg.v[..., 1], reg.v[..., 0])
    hist, edges = np.histogram(ang, 36, (-np.pi / 2, np.pi / 2), weights=reg.coh)
    m1, m2 = np.argsort(hist)[-2:]
    ph1, ph2 = 2 * edges[m1] + p["_phase"], 2 * edges[m2] + 0.5 * p["_phase"]
    return (
        int(p.get("freq_a", a)),
        int(p.get("freq_b", b)),
        float(p.get("phase1", ph1)),
        float(p.get("phase2", ph2)),
    )


def _curve(kind: str, s: np.ndarray, a: int, b: int, ph1: float, ph2: float, damp: float):
    if kind == "circle":
        y, x = -np.cos(TAU * s), np.sin(TAU * s)
    elif kind == "spiral":
        r, t = 1 - 0.85 * s, TAU * (a + 2) * s + ph1
        y, x = -r * np.cos(t), r * np.sin(t)
    elif kind == "lissajous":
        y, x = np.sin(TAU * b * s + ph2), np.sin(TAU * a * s + ph1)
    else:  # damped harmonograph, scaled so its widest swing fills the cell
        y, x = _harmonograph(s, a, b, ph1, ph2, damp)
        peak = np.abs(_harmonograph(np.linspace(0, 1, 4096), a, b, ph1, ph2, damp)).max()
        y, x = y / max(peak, 1e-6), x / max(peak, 1e-6)
    return np.stack([0.5 + 0.45 * y, 0.5 + 0.45 * x], -1)


def _strata(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified jittered u in [0, 1): even coverage without the moire of a regular step."""
    return (np.arange(n) + rng.random(n)) / n


def _harmonograph(s: np.ndarray, a: int, b: int, ph1: float, ph2: float, damp: float):
    """Two decaying pendulums per axis at small frequency ratios, slightly detuned."""
    e, t = np.exp(-damp * s), TAU * HARMONO_PERIODS * s
    y = e * (np.sin((a + 0.013) * t + ph1 + np.pi / 2) + np.sin(b * t + ph2 + np.pi / 3))
    x = e * (np.sin(a * t + ph1) + np.sin((b - 0.011) * t + ph2))
    return np.stack([y, x])


def _amp(pts: np.ndarray, p: dict) -> np.ndarray:
    return 0.5 + (pts - 0.5) * p.get("amp", 1.0)


def parametric(reg: Region, n: int, p: dict, rng: np.random.Generator):
    """Circle, spiral, Lissajous or damped harmonograph, tuned by the reference."""
    a, b, ph1, ph2 = _freqs(reg, p)
    kind, u = p["curve"], _strata(n, rng)
    c = lambda s: _curve(kind, s, a, b, ph1, ph2, p["damping"])  # noqa: E731
    if kind == "circle":  # times-table chords: cardioid, nephroid, ...
        p2 = c(p.get("mult", 2 + (a * b + p["_variant"]) % 4) * u + p["lag"])
    elif kind == "lissajous":
        p2 = c(u + p["lag"])
    elif kind == "harmonograph":  # p2 trails by under a period: a twisting ribbon
        p2 = c(np.clip(u + 0.4 * p["lag"] / HARMONO_PERIODS, 0, 1))
    else:  # spiral: p2 trails closely
        p2 = c(np.clip(u + 0.15 * p["lag"], 0, 1))
    return _amp(c(u), p), _amp(p2, p)


def contour(reg: Region, n: int, p: dict, rng: np.random.Generator):
    """Two longest smoothed reference contours, arc-length resampled; optionally polar."""
    gh, gw = reg.lum.shape
    sm = ndimage.gaussian_filter(reg.lum, min(gh, gw) / 80)
    cs = [c for lv in np.quantile(sm, [0.3, 0.5, 0.7]) for c in measure.find_contours(sm, lv)]
    cs = sorted(cs, key=lambda c: -np.linalg.norm(np.diff(c, axis=0), axis=1).sum())[:2]
    if not cs:
        return parametric(reg, n, p | {"curve": "circle"}, rng)
    cs = [ndimage.gaussian_filter1d(c, 3, axis=0) / (gh - 1, gw - 1) for c in cs]
    a, b = cs[0], cs[-1]
    u = _strata(n, rng)
    p1, p2 = _along(a, u), _along(b, (u + p["lag"]) % 1.0)
    if p["polar"] == "on":
        p1, p2 = _polar(p1), _polar(p2)
    return _amp(p1, p), _amp(p2, p)


def _along(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Point at arc-length fraction ``s`` along polyline ``c``."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    t = s * d[-1]
    return np.stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])], -1)


def _polar(pts: np.ndarray) -> np.ndarray:
    r, th = 0.06 + 0.42 * pts[:, 0], TAU * pts[:, 1]
    return np.stack([0.5 - r * np.cos(th), 0.5 + r * np.sin(th)], -1)


def advected(reg: Region, n: int, p: dict, rng: np.random.Generator):
    """A parametric sweep whose endpoints drift along the orientation field, more as u grows."""
    p1, p2 = parametric(reg, n, p, rng)
    dist = p["advect"] * (0.2 + 0.8 * (np.arange(n) + 0.5) / n)
    return _advect(p1, reg, dist), _advect(p2, reg, dist)


def _advect(pts: np.ndarray, reg: Region, dist: np.ndarray, steps: int = 24) -> np.ndarray:
    gh, gw = reg.lum.shape
    pts = pts.copy()
    prev = np.zeros_like(pts)
    for _ in range(steps):
        coords = [np.clip(pts[:, 0], 0, 1) * (gh - 1), np.clip(pts[:, 1], 0, 1) * (gw - 1)]
        vx = ndimage.map_coordinates(reg.v[..., 0], coords, order=1)
        vy = ndimage.map_coordinates(reg.v[..., 1], coords, order=1)
        a = 0.5 * np.arctan2(vy, vx)
        d = np.stack([np.sin(a), np.cos(a)], -1)
        d *= np.where((d * prev).sum(1) < 0, -1.0, 1.0)[:, None]  # keep heading
        prev = d
        pts += d * (dist / steps)[:, None]
    return pts


def density(reg: Region, n: int, p: dict, rng: np.random.Generator):
    """Short segments along the orientation field, placed so line density follows darkness."""
    gh, gw = reg.lum.shape
    n *= 4  # short segments need more of them to build tone
    dark = (1 - reg.lum) ** 2 + 1e-3
    cells = rng.choice(gh * gw, n, p=(dark / dark.sum()).ravel())
    cy, cx = cells // gw, cells % gw
    c = np.stack([cy + rng.random(n), cx + rng.random(n)], -1) / (gh, gw)
    a = 0.5 * np.arctan2(reg.v[cy, cx, 1], reg.v[cy, cx, 0])
    half = 0.5 * p["seg_len"] * (0.5 + dark[cy, cx])[:, None] * np.stack([np.sin(a), np.cos(a)], -1)
    return _amp(c - half, p), _amp(c + half, p)


SOURCES = {"parametric": parametric, "contour": contour, "advected": advected, "density": density}
