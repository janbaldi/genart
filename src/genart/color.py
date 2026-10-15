"""Color conversions (sRGB, linear RGB, OKLab) and palette utilities."""

import numpy as np
from scipy.cluster.vq import kmeans2

# Palette lightness limits in OKLab L: keeps us away from pure black and pure white.
L_MIN, L_MAX = 0.18, 0.93
PAPER_L = 0.95

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB, clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB (..., 3) to OKLab (..., 3)."""
    lms = np.cbrt(np.asarray(rgb, dtype=np.float64) @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab (..., 3) to linear RGB (..., 3), clipped to [0, 1]."""
    lms = (np.asarray(lab, dtype=np.float64) @ _M2_INV.T) ** 3
    return np.clip(lms @ _M1_INV.T, 0.0, 1.0).astype(np.float32)


def extract_palette(
    rgb: np.ndarray, k: int, rng: np.random.Generator, n_samples: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """K-means palette in OKLab.

    Parameters
    ----------
    rgb : np.ndarray
        Linear RGB image (H, W, 3).
    k : int
        Number of palette colors.
    rng : np.random.Generator
        Source of randomness for sampling and k-means init.
    n_samples : int
        Number of pixels sampled for clustering.

    Returns
    -------
    colors : np.ndarray
        (k, 3) linear RGB colors sorted by lightness, L clamped to [L_MIN, L_MAX].
    weights : np.ndarray
        (k,) fraction of sampled pixels per color.
    """
    lab = linear_to_oklab(rgb.reshape(-1, 3))
    lab = lab[rng.choice(len(lab), size=min(n_samples, len(lab)), replace=False)]
    centers, labels = kmeans2(lab.astype(np.float64), k, minit="++", seed=rng)
    weights = np.bincount(labels, minlength=k).astype(np.float32)
    keep = weights > 0
    centers, weights = centers[keep], weights[keep] / weights[keep].sum()
    order = np.argsort(centers[:, 0])
    centers, weights = centers[order], weights[order]
    centers[:, 0] = np.clip(centers[:, 0], L_MIN, L_MAX)
    return oklab_to_linear(centers), weights.astype(np.float32)


def paper_tone(palette: np.ndarray) -> np.ndarray:
    """Warm off-white ground tinted by the lightest palette color (never pure white)."""
    lab = linear_to_oklab(palette[-1])
    warm = np.array([PAPER_L, 0.004, 0.014], dtype=np.float32)
    tint = np.array([PAPER_L, lab[1], lab[2]], dtype=np.float32)
    return oklab_to_linear(0.75 * warm + 0.25 * tint)


def nearest_palette_index(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (OKLab distance) for each pixel of (..., 3)."""
    lab = linear_to_oklab(rgb)
    plab = linear_to_oklab(palette)
    d = ((lab[..., None, :] - plab) ** 2).sum(-1)
    return d.argmin(-1)
