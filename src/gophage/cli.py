"""Typer CLI entry point for GOPhage."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from gophage import __version__
from gophage.pipeline import run_pipeline


class PLM(str, Enum):
    esm2_12 = "esm2-12"
    esm2_33 = "esm2-33"


class Ont(str, Enum):
    BP = "BP"
    CC = "CC"
    MF = "MF"


app = typer.Typer(
    name="gophage",
    help="GOPhage: GO-term annotation for phage proteins.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gophage {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """GOPhage CLI."""


@app.command("run")
def run(
    contigs: Path = typer.Option(
        ...,
        "--contigs",
        exists=True,
        dir_okay=False,
        readable=True,
        help="DNA FASTA file of contigs.",
    ),
    plm: PLM = typer.Option(
        PLM.esm2_12,
        "--plm",
        case_sensitive=False,
        help="ESM2 model size (esm2-12 -> base, esm2-33 -> large).",
    ),
    ont: Ont = typer.Option(
        Ont.CC,
        "--ont",
        case_sensitive=False,
        help="GO ontology branch: BP, CC, or MF.",
    ),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Batch size."),
    mid_dir: Path = typer.Option(
        Path("CC_results"),
        "--mid-dir",
        help="Directory for intermediate + final outputs.",
    ),
    threshold: float = typer.Option(
        0.1,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Score cutoff for the GO summary expansion step.",
    ),
    data_dir: Path = typer.Option(
        Path("."),
        "--data-dir",
        help=(
            "Root directory containing DataBase/, ESM_model/, PhaGO_model/, "
            "Protein_annotation/ and Term_label/."
        ),
    ),
    diamond_threads: int = typer.Option(
        8, "--diamond-threads", min=1, help="Threads for diamond blastp."
    ),
    num_workers: int = typer.Option(
        2, "--num-workers", min=0, help="DataLoader worker processes for inference."
    ),
) -> None:
    """Run the full GOPhage+ pipeline (prodigal -> diamond -> ESM2 -> PhaGO)."""
    summary = run_pipeline(
        contigs=contigs,
        plm=plm.value,
        ont=ont.value,
        batch_size=batch_size,
        mid_dir=mid_dir,
        data_dir=data_dir,
        threshold=threshold,
        diamond_threads=diamond_threads,
        num_workers=num_workers,
    )
    typer.echo(f"Done. Summary written to: {summary}")


if __name__ == "__main__":
    app()
