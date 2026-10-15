"""Shared helpers for techniques."""

import numpy as np


def local_t(t: float, t_range: tuple[float, float]) -> float:
    """Map global t onto [0, 1] within a technique's range."""
    lo, hi = t_range
    return float(np.clip((t - lo) / (hi - lo), 0, 1))


def params_of(genome_params: dict[str, float], defaults: dict) -> dict[str, float]:
    """Genome params over technique defaults."""
    return {k: genome_params.get(k, d) for k, (d, _, _) in defaults.items()}
