"""Render orchestration: sizes, margins, export and provenance sidecars."""

import json
import struct
import subprocess
import zlib
from datetime import UTC, datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numba import njit

from genart.color import linear_to_srgb, paper_tone
from genart.context import Context, load_many
from genart.genome import Genome

PREVIEW = (1000, 714)  # (h, w), 5:7 portrait
FINAL = (8268, 5906)  # 50x70 cm @ 300 dpi
MARGIN = 0.06  # fraction of the short side


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render a genome to a float32 linear RGB array of shape (h, w, 3).

    The technique fills the inner area; the margin is paper unless it is full-bleed.
    """
    from genart.techniques import REGISTRY  # late: techniques import the line renderer below

    (name,) = genome.technique_chain  # chains of length one for now
    tech = REGISTRY[name]
    h, w = size
    if tech.FULL_BLEED:
        return tech.render(ctx, genome, size)
    m = max(1, round(MARGIN * min(h, w)))
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = paper_tone(ctx.palette)
    canvas[m : h - m, m : w - m] = tech.render(ctx, genome, (h - 2 * m, w - 2 * m))
    return canvas


def accumulate_lines(
    p1: np.ndarray, p2: np.ndarray, size: tuple[int, int], ss: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Accumulate anti-aliased line density into float32 buffers.

    Parameters
    ----------
    p1, p2 : (n, 2) float arrays
        Segment endpoints as (y, x) in output pixel units.
    size : (h, w)
        Output size.
    ss : int
        Supersampling factor; lines are one sub-pixel wide.
    groups : (n,) int array, optional
        Ink index per segment; one density layer per group.

    Returns
    -------
    np.ndarray
        (n_groups, h, w) float32 density in line crossings per output pixel.
    """
    h, w = size
    groups = np.zeros(len(p1), np.int64) if groups is None else groups.astype(np.int64)
    n_groups = int(groups.max()) + 1 if len(groups) else 1
    buf = np.zeros((n_groups, h * ss, w * ss), np.float32)
    segs = np.concatenate([p1, p2], 1).astype(np.float64) * ss - 0.5
    _wu_lines(buf, segs, groups)
    return buf.reshape(n_groups, h, ss, w, ss).sum((2, 4)) / ss


@njit(cache=True)
def _wu_lines(buf, segs, groups):
    """Xiaolin Wu lines, additive; each step's weight is scaled so density follows length."""
    _, H, W = buf.shape
    for i in range(len(segs)):
        y0, x0, y1, x1 = segs[i, 0], segs[i, 1], segs[i, 2], segs[i, 3]
        g = groups[i]
        if not (np.isfinite(y0) and np.isfinite(x0) and np.isfinite(y1) and np.isfinite(x1)):
            continue
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:  # iterate along the major axis
            y0, x0, y1, x1 = x0, y0, x1, y1
        if x0 > x1:
            y0, x0, y1, x1 = y1, x1, y0, x0
        dx = x1 - x0
        if dx < 1e-9:
            continue
        grad = (y1 - y0) / dx
        wt = np.sqrt(1.0 + grad * grad)
        limit = H if steep else W
        other = W if steep else H
        for x in range(max(int(np.ceil(x0)), 0), min(int(np.floor(x1)), limit - 1) + 1):
            y = y0 + grad * (x - x0)
            yi = int(np.floor(y))
            f = y - yi
            for yy, a in ((yi, 1.0 - f), (yi + 1, f)):
                if 0 <= yy < other:
                    if steep:
                        buf[g, x, yy] += a * wt
                    else:
                        buf[g, yy, x] += a * wt


def tone_map(density: np.ndarray, gain: float = 1.0) -> tuple[np.ndarray, float]:
    """Map density to ink coverage ``1 - exp(-k d)``.

    ``k`` is auto-calibrated so the 99th percentile of the non-zero total density maps to
    0.95 ink, then multiplied by ``gain``. Returns (ink, k); ink has the shape of density.
    """
    total = density.sum(0) if density.ndim == 3 else density
    nz = total[total > 0]
    p99 = float(np.percentile(nz, 99)) if nz.size else 1.0
    k = -np.log(0.05) / max(p99, 1e-9) * gain
    return (1 - np.exp(-k * density)).astype(np.float32), k


def export(rgb: np.ndarray, path: Path, bits: int = 8) -> None:
    """Write linear RGB as an sRGB PNG with 8 or 16 bits per channel."""
    srgb = linear_to_srgb(rgb)
    dtype = np.uint16 if bits == 16 else np.uint8
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.round(srgb * np.iinfo(dtype).max).astype(dtype)
    if bits == 16:
        _write_png16(data, path)
    else:
        iio.imwrite(path, data)


def _write_png16(rgb: np.ndarray, path: Path) -> None:
    """Minimal 16-bit RGB PNG writer (Pillow cannot write 16-bit RGB)."""
    h, w, _ = rgb.shape
    rows = np.zeros((h, 1 + w * 6), np.uint8)  # leading 0 = no filter per row
    rows[:, 1:] = rgb.astype(">u2").view(np.uint8).reshape(h, w * 6)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
        + chunk(b"sRGB", b"\x00")
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def git_commit() -> str:
    """Current commit hash, suffixed with -dirty if the tree has local changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def render_to_file(
    collection: Path, genome: Genome, out_dir: Path, size: tuple[int, int], bits: int
) -> Path:
    """Render a genome, write ``<id>.png`` and a ``<id>.json`` provenance sidecar."""
    ctx = load_many(collection, genome.reference_ids)
    rgb = render(ctx, genome, size)
    path = out_dir / f"{genome.id}.png"
    export(rgb, path, bits)
    sidecar = {
        "id": genome.id,
        "genome": json.loads(genome.to_json()),
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "render": {"size_hw": list(size), "bits": bits, "margin": MARGIN, "color": "sRGB"},
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return path
