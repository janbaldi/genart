"""Command line: ingest, sweep, mutate, sheet, final."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from genart import context, sheet
from genart.genome import Genome
from genart.genome import mutate as mutate_genome
from genart.render import FINAL, PREVIEW, render_to_file
from genart.techniques import REGISTRY, for_t

app = typer.Typer(no_args_is_help=True, help="Reference-driven generative art collections.")
ROOT = Path("collections")


def _dir(name: str) -> Path:
    return ROOT / name


def _ref_ids(col: Path) -> tuple[str, ...]:
    ids = tuple(sorted(p.stem for p in (col / "refs").glob("*")))
    if not ids:
        raise typer.BadParameter(f"no references in {col}; run ingest first")
    return ids


def _genome(col: Path, gid: str) -> Genome:
    path = col / "genomes" / f"{gid}.json"
    if not path.exists():
        raise typer.BadParameter(f"unknown genome id {gid}")
    return Genome.load(path)


@app.command()
def ingest(name: str, images: list[Path]) -> None:
    """Copy reference images into a collection and cache their context."""
    if not 1 <= len(images) <= 3:
        raise typer.BadParameter("give 1 to 3 reference images")
    for img in images:
        typer.echo(f"{img} -> {context.ingest(img, _dir(name))}")


@app.command()
def sweep(
    name: str,
    n: int = 24,
    t_min: float = 0.0,
    t_max: float = 1.0,
    seed: Annotated[int | None, typer.Option(help="Master seed; random if omitted")] = None,
) -> None:
    """Render N previews evenly spaced over [t_min, t_max] with random seeds."""
    col = _dir(name)
    refs = _ref_ids(col)
    rng = np.random.default_rng(seed)
    for t in np.linspace(t_min, t_max, n):
        t = round(float(t), 4)
        g = Genome(name, refs, (for_t(t),), t, {}, int(rng.integers(2**31)))
        g.save(col / "genomes")
        render_to_file(col, g, col / "renders", PREVIEW, 8)
        typer.echo(f"{g.id}  t={t:.3f}  {g.technique_chain[0]}")


@app.command()
def mutate(
    name: str,
    ids: list[str],
    k: int = 4,
    reseed: bool = False,
    seed: Annotated[int | None, typer.Option(help="Master seed; random if omitted")] = None,
) -> None:
    """Make K small mutations of each genome and render previews."""
    col = _dir(name)
    rng = np.random.default_rng(seed)
    for gid in ids:
        parent = _genome(col, gid)
        bounds = REGISTRY[parent.technique_chain[0]].PARAMS
        for _ in range(k):
            g = mutate_genome(parent, bounds, rng, reseed)
            g.save(col / "genomes")
            render_to_file(col, g, col / "renders", PREVIEW, 8)
            typer.echo(f"{gid} -> {g.id}  t={g.t:.3f}")


@app.command(name="sheet")
def sheet_cmd(name: str) -> None:
    """Build the HTML contact sheet."""
    typer.echo(sheet.build(_dir(name)))


@app.command()
def final(name: str, ids: list[str]) -> None:
    """Re-render chosen genomes at print resolution as 16-bit PNGs."""
    col = _dir(name)
    for gid in ids:
        typer.echo(render_to_file(col, _genome(col, gid), col / "finals", FINAL, 16))
