"""Genome: the complete, serializable recipe for one artwork."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Genome:
    """Same genome -> bit-identical render.

    Attributes
    ----------
    collection_id : str
    reference_ids : tuple of str
        Content hashes of the references; the first supplies spatial fields.
    technique_chain : tuple of str
        Registered technique names. Milestone 1 renders chains of length one.
    t : float
        Abstraction in [0, 1].
    params : dict
        Technique parameters; missing keys fall back to the technique defaults.
    seed : int
    """

    collection_id: str
    reference_ids: tuple[str, ...]
    technique_chain: tuple[str, ...]
    t: float
    params: dict[str, float | str] = field(default_factory=dict)
    seed: int = 0

    @property
    def id(self) -> str:
        """Short content hash of the canonical JSON."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:10]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        d = json.loads(s)
        d["reference_ids"] = tuple(d["reference_ids"])
        d["technique_chain"] = tuple(d["technique_chain"])
        return cls(**d)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.json"
        path.write_text(json.dumps(asdict(self), sort_keys=True, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Genome":
        return cls.from_json(path.read_text())


def mutate(
    genome: Genome,
    bounds: dict[str, tuple[float, float, float]],
    rng: np.random.Generator,
    reseed: bool = False,
) -> Genome:
    """Small mutation: nudge t by up to 0.05 and jitter each param by ~10% within bounds.

    Parameters
    ----------
    genome : Genome
    bounds : dict
        ``name -> (default, lo, hi)`` for the genome's technique.
    rng : np.random.Generator
    reseed : bool
        Also draw a new seed.
    """
    params = dict(genome.params)  # categorical choices and overrides pass through
    for name, (default, lo, hi) in bounds.items():
        v = genome.params.get(name, default)
        params[name] = float(np.clip(v * (1 + rng.normal(0, 0.1)), lo, hi))
    t = float(np.clip(genome.t + rng.uniform(-0.05, 0.05), 0, 1))
    seed = int(rng.integers(2**31)) if reseed else genome.seed
    return replace(genome, t=round(t, 4), params=params, seed=seed)
