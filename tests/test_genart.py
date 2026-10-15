import hashlib
import zlib

import imageio.v3 as iio
import numpy as np
import pytest
from skimage import data, transform

from genart import context
from genart.genome import Genome, mutate
from genart.render import render
from genart.techniques import REGISTRY


@pytest.fixture(scope="module")
def ctx():
    return context.extract(data.astronaut()[::2, ::2])


def _genome(name: str, seed: int = 3) -> Genome:
    lo, hi = REGISTRY[name].T_RANGE
    return Genome("test", ("ref",), (name,), (lo + hi) / 2, {}, seed)


@pytest.mark.parametrize("name", REGISTRY)
def test_determinism(ctx, name):
    g = Genome.from_json(_genome(name).to_json())  # round-trip through JSON too
    a = render(ctx, g, (140, 100))
    b = render(ctx, _genome(name), (140, 100))
    assert hashlib.sha256(a.tobytes()).digest() == hashlib.sha256(b.tobytes()).digest()
    assert not np.array_equal(a, render(ctx, _genome(name, seed=4), (140, 100)))


@pytest.mark.parametrize("name", REGISTRY)
def test_resolution_independence(ctx, name):
    small = render(ctx, _genome(name), (280, 200))
    big = render(ctx, _genome(name), (840, 600))
    down = transform.resize(big, small.shape, anti_aliasing=True)
    assert np.abs(down - small).mean() < 0.04


def test_context_cache(tmp_path, monkeypatch):
    ref = tmp_path / "ref.png"
    iio.imwrite(ref, data.astronaut()[::4, ::4])
    col = tmp_path / "col"
    rid = context.ingest(ref, col)
    assert (col / "context" / f"{rid}.v{context.VERSION}.npz").exists()

    def boom(*_a, **_k):
        raise AssertionError("cache miss")

    monkeypatch.setattr(context, "extract", boom)
    assert context.ingest(ref, col) == rid
    assert context.load(col, rid).palette.shape[1] == 3


def test_mutate_stays_in_bounds():
    rng = np.random.default_rng(0)
    g = _genome("flow_field")
    bounds = REGISTRY["flow_field"].PARAMS
    for _ in range(50):
        g = mutate(g, bounds, rng)
        assert 0 <= g.t <= 1
        assert all(bounds[k][1] <= v <= bounds[k][2] for k, v in g.params.items())


def test_png16_roundtrip(tmp_path):
    from genart.render import _write_png16

    a = np.random.default_rng(0).integers(0, 65536, (7, 5, 3)).astype(np.uint16)
    _write_png16(a, tmp_path / "x.png")
    raw = (tmp_path / "x.png").read_bytes()
    idat = raw[raw.index(b"IDAT") + 4 : raw.index(b"IEND") - 8]  # Pillow reads 16-bit RGB as 8-bit
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(7, 1 + 5 * 6)
    assert np.array_equal(rows[:, 1:].copy().view(">u2").reshape(7, 5, 3), a)
