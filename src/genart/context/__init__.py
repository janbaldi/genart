"""Per-reference context extraction and caching.

Techniques read only from a :class:`Context`, never from raw reference pixels.
"""

import hashlib
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage
from skimage import feature, segmentation, transform

from genart.color import extract_palette, linear_to_oklab, srgb_to_linear

WORK_SIZE = 1024  # long side of the analysis resolution
PALETTE_K = 7
VERSION = 1  # bump when extraction changes, so stale caches are ignored


@dataclass(frozen=True)
class Context:
    """Everything a technique may know about the reference(s).

    Attributes
    ----------
    image : (H, W, 3) float32 linear RGB at analysis resolution.
    palette : (k, 3) float32 linear RGB, sorted by lightness.
    weights : (k,) float32 palette weights, sum to 1.
    edges : (H, W) bool Canny edges.
    edge_dist : (H, W) float32 distance to nearest edge, as a fraction of the long side.
    angle : (H, W) float32 flow orientation in radians (along structures, x right, y down).
    coherence : (H, W) float32 orientation coherence in [0, 1].
    labels : (H, W) int32 SLIC superpixel labels.
    luminance : (H, W) float32 OKLab lightness.
    """

    image: np.ndarray
    palette: np.ndarray
    weights: np.ndarray
    edges: np.ndarray
    edge_dist: np.ndarray
    angle: np.ndarray
    coherence: np.ndarray
    labels: np.ndarray
    luminance: np.ndarray

    def sample(self, name: str, size: tuple[int, int], order: int = 1) -> np.ndarray:
        """Return field ``name`` center-cropped to the aspect of ``size`` and resized to it."""
        return fit(getattr(self, name), size, order)


def fit(arr: np.ndarray, size: tuple[int, int], order: int = 1) -> np.ndarray:
    """Center-crop ``arr`` to the aspect ratio of ``size`` (h, w), then resize to ``size``."""
    h, w = size
    H, W = arr.shape[:2]
    if W * h > H * w:
        cw = max(1, round(H * w / h))
        x0 = (W - cw) // 2
        arr = arr[:, x0 : x0 + cw]
    else:
        ch = max(1, round(W * h / w))
        y0 = (H - ch) // 2
        arr = arr[y0 : y0 + ch]
    out = transform.resize(
        arr, (h, w) + arr.shape[2:], order=order, anti_aliasing=False, preserve_range=True
    )
    return out.astype(arr.dtype)


def extract(srgb: np.ndarray, seed: int = 0) -> Context:
    """Extract a context from an sRGB image.

    Parameters
    ----------
    srgb : np.ndarray
        (H, W, 3) image, uint8 or float in [0, 1].
    seed : int
        Seed for palette clustering and superpixels.
    """
    rng = np.random.default_rng(seed)
    img = np.asarray(srgb)
    if img.dtype.kind in "ui":
        img = img / np.iinfo(img.dtype).max
    img = img[..., :3].astype(np.float32)
    scale = WORK_SIZE / max(img.shape[:2])
    if scale < 1:
        img = transform.resize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale), 3))
    rgb = srgb_to_linear(img)
    lum = linear_to_oklab(rgb)[..., 0]
    long_side = max(lum.shape)

    palette, weights = extract_palette(rgb, PALETTE_K, rng)

    edges = feature.canny(lum, sigma=long_side / 400)
    edge_dist = ndimage.distance_transform_edt(~edges).astype(np.float32) / long_side

    # Structure tensor: dominant gradient direction, flow runs perpendicular to it.
    arr, arc, acc = feature.structure_tensor(lum, sigma=long_side / 200, order="rc")
    grad = 0.5 * np.arctan2(2 * arc, acc - arr)
    angle = (grad + np.pi / 2).astype(np.float32)
    coherence = np.sqrt((acc - arr) ** 2 + 4 * arc**2) / (acc + arr + 1e-12)

    labels = segmentation.slic(img, n_segments=400, compactness=10, start_label=0)

    return Context(
        image=rgb,
        palette=palette,
        weights=weights,
        edges=edges,
        edge_dist=edge_dist,
        angle=angle,
        coherence=np.clip(coherence, 0, 1).astype(np.float32),
        labels=labels.astype(np.int32),
        luminance=lum.astype(np.float32),
    )


def ref_id(path: Path) -> str:
    """Content hash of a reference file, used as its id."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def ingest(path: Path, collection: Path) -> str:
    """Copy a reference into ``collection/refs`` and cache its context. Returns the ref id."""
    rid = ref_id(path)
    refs = collection / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    if not any(refs.glob(f"{rid}.*")):
        shutil.copyfile(path, refs / f"{rid}{path.suffix.lower()}")
    load(collection, rid)
    return rid


def load(collection: Path, rid: str) -> Context:
    """Load a reference's context from cache, extracting and caching it on a miss."""
    cache = collection / "context" / f"{rid}.v{VERSION}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return Context(**{f.name: z[f.name] for f in fields(Context)})
    (src,) = (collection / "refs").glob(f"{rid}.*")
    ctx = extract(iio.imread(src))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{f.name: getattr(ctx, f.name) for f in fields(Context)})
    return ctx


def load_many(collection: Path, rids: tuple[str, ...]) -> Context:
    """Combine several references: spatial fields from the first, palette pooled from all."""
    ctxs = [load(collection, r) for r in rids]
    if len(ctxs) == 1:
        return ctxs[0]
    # shortcut: pooled palette keeps the k heaviest colors overall, re-cluster if refs clash.
    pal = np.concatenate([c.palette for c in ctxs])
    w = np.concatenate([c.weights for c in ctxs]) / len(ctxs)
    top = np.sort(np.argsort(w)[-PALETTE_K:])
    pal, w = pal[top], w[top]
    order = np.argsort(linear_to_oklab(pal)[:, 0])
    first = ctxs[0]
    return Context(
        **{f.name: getattr(first, f.name) for f in fields(Context)}
        | {"palette": pal[order], "weights": w[order] / w[order].sum()}
    )
