# genart

Turns 1–3 reference photos into a curated collection of print-quality generative
artworks along an abstraction axis `t ∈ [0, 1]`: recognizable (t≈0) → collage /
fragments (t≈0.5) → only the reference's "DNA" (palette, orientation, density) at t≈1.
Works are sold as editioned prints: reproducibility and provenance are requirements.

## Architecture

- **Genome** (`src/genart/genome.py`): `(collection_id, reference_ids, technique_chain,
  t, params, seed)`. Same genome → bit-identical output. Stored as JSON in
  `collections/<name>/genomes/<id>.json`; the id is a hash of the canonical JSON.
- **Context** (`src/genart/context/`): extracted once per reference at 1024 px long side
  and cached as `context/<ref_id>.v<VERSION>.npz` (bump `VERSION` when extraction
  changes). Holds palette (OKLab k-means, sorted by lightness) + weights, Canny edges +
  distance field, structure-tensor orientation + coherence, SLIC labels, luminance and
  the working-resolution image. Techniques read ONLY the context, never raw pixels.
  With several refs, spatial fields come from the first and the palette is pooled.
- **Techniques** (`src/genart/techniques/`): `render(ctx, genome, size) -> (h, w, 3)`
  float32 linear RGB. Each module also defines `PARAMS` (`name -> (default, lo, hi)`),
  `T_RANGE` and `FULL_BLEED`. Spatial params are fractions of the short side, and
  context fields are fetched with `ctx.sample(name, size)`, so output is
  resolution-independent. Register new techniques in `techniques/__init__.py`.
  - `superpixel_collage` t∈[0, .35), `flow_field` t∈[.35, .7), `palette_abstraction` t∈[.7, 1].
  - Chains: Milestone 1 renders chains of length one.
- **Render** (`src/genart/render.py`): adds the paper margin (6% of the short side),
  converts to sRGB at export. Preview 714×1000 8-bit, final 5906×8268 16-bit
  (50×70 cm @ 300 dpi). Every PNG gets a JSON sidecar (genome, git commit, timestamp, settings).
- **Sheet** (`src/genart/sheet.py`): self-contained `collections/<name>/sheet.html`;
  clicking a thumbnail copies its genome id.

## Running

```bash
uv sync
uv run genart ingest demo path/to/ref.jpg          # 1–3 refs
uv run genart sweep demo --n 24                     # previews across t ∈ [0, 1]
uv run genart mutate demo <id> <id> --k 4 [--reseed]
uv run genart sheet demo                            # collections/demo/sheet.html
uv run genart final demo <id>                       # collections/demo/finals/
uv run pytest
uv run ruff check . && uv run ruff format .
```

Or `/new-collection <name> <refs...>`.

## Conventions (strict)

- uv, Python 3.12, ruff (lint + format), src layout, numpy-style docstrings, type hints.
- Deterministic RNG: `np.random.default_rng(seed)` passed explicitly; no global state.
  Numba kernels stay single-threaded so results are deterministic.
- numpy / scikit-image / scipy first; numba for hot loops. Ask before adding heavy
  dependencies (torch, taichi, ...).
- Color work in OKLab / linear RGB; convert to sRGB only at export.

## Aesthetic rules

- Generous margins (≥6% of short side) unless a technique is explicitly full-bleed.
- Palette discipline: only extracted palette colors plus one paper/ground tone.
  No pure black (#000), no pure white (#fff) — palette L is clamped, paper L is 0.95.
- Avoid uniform randomness: use clustering, gradients of density, and focal areas.
- Every image should read well both as a thumbnail and up close.
