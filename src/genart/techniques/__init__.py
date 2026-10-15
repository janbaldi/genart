"""Technique registry.

Every technique module exposes:

- ``render(context, genome, size) -> np.ndarray``: float32 linear RGB (h, w, 3) in [0, 1].
- ``PARAMS``: ``name -> (default, lo, hi)``; spatial params are fractions of the short side.
- ``T_RANGE``: the slice of the abstraction axis the technique covers.
- ``FULL_BLEED``: whether render.py should skip the paper margin.
- optionally ``init_params(t, rng) -> dict``: categorical choices for a new genome.
"""

from types import ModuleType

from genart.techniques import flow_field, line_sweep, palette_abstraction, superpixel_collage

REGISTRY: dict[str, ModuleType] = {
    "superpixel_collage": superpixel_collage,
    "flow_field": flow_field,
    "palette_abstraction": palette_abstraction,
    "line_sweep": line_sweep,
}


def for_t(t: float) -> list[str]:
    """Names of the techniques whose T_RANGE contains ``t``."""
    return [
        name
        for name, mod in REGISTRY.items()
        if mod.T_RANGE[0] <= t < mod.T_RANGE[1] or t == mod.T_RANGE[1] == 1.0
    ]
