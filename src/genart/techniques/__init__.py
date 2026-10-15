"""Technique registry.

Every technique module exposes:

- ``render(context, genome, size) -> np.ndarray``: float32 linear RGB (h, w, 3) in [0, 1].
- ``PARAMS``: ``name -> (default, lo, hi)``; spatial params are fractions of the short side.
- ``T_RANGE``: the slice of the abstraction axis the technique covers.
- ``FULL_BLEED``: whether render.py should skip the paper margin.
"""

from types import ModuleType

from genart.techniques import flow_field, palette_abstraction, superpixel_collage

REGISTRY: dict[str, ModuleType] = {
    "superpixel_collage": superpixel_collage,
    "flow_field": flow_field,
    "palette_abstraction": palette_abstraction,
}


def for_t(t: float) -> str:
    """Name of the technique whose T_RANGE contains ``t``."""
    for name, mod in REGISTRY.items():
        lo, hi = mod.T_RANGE
        if lo <= t < hi or (hi == 1.0 and t == 1.0):
            return name
    raise ValueError(f"no technique covers t={t}")
