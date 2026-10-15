"""Render orchestration: sizes, margins, export and provenance sidecars."""

import json
import struct
import subprocess
import zlib
from datetime import UTC, datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from genart.color import linear_to_srgb, paper_tone
from genart.context import Context, load_many
from genart.genome import Genome
from genart.techniques import REGISTRY

PREVIEW = (1000, 714)  # (h, w), 5:7 portrait
FINAL = (8268, 5906)  # 50x70 cm @ 300 dpi
MARGIN = 0.06  # fraction of the short side


def render(ctx: Context, genome: Genome, size: tuple[int, int]) -> np.ndarray:
    """Render a genome to a float32 linear RGB array of shape (h, w, 3).

    The technique fills the inner area; the margin is paper unless it is full-bleed.
    """
    (name,) = genome.technique_chain  # Milestone 1: chains of length one
    tech = REGISTRY[name]
    h, w = size
    if tech.FULL_BLEED:
        return tech.render(ctx, genome, size)
    m = max(1, round(MARGIN * min(h, w)))
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = paper_tone(ctx.palette)
    canvas[m : h - m, m : w - m] = tech.render(ctx, genome, (h - 2 * m, w - 2 * m))
    return canvas


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
