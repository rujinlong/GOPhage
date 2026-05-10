"""Typer CLI entry point for GOPhage."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

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


class Device(str, Enum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


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
    contigs: Optional[Path] = typer.Option(
        None,
        "--contigs",
        exists=True,
        dir_okay=False,
        readable=True,
        help="DNA FASTA file of contigs (will run prodigal). "
             "Mutually exclusive with --proteins.",
    ),
    proteins: Optional[Path] = typer.Option(
        None,
        "--proteins",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Pre-translated protein FASTA (skips prodigal so your existing "
             "annotation IDs are preserved). Proteins must be ordered by "
             "genome position. Mutually exclusive with --contigs.",
    ),
    protein_contig_map: Optional[Path] = typer.Option(
        None,
        "--protein-contig-map",
        exists=True,
        dir_okay=False,
        readable=True,
        help="TSV mapping (protein_id<TAB>contig_id) for use with --proteins. "
             "If omitted, contig names are inferred by stripping the trailing "
             "_<idx> from each protein id (prodigal convention).",
    ),
    plm: PLM = typer.Option(
        PLM.esm2_12,
        "--plm",
        case_sensitive=False,
        help="ESM2 model size (esm2-12 -> base, esm2-33 -> large).",
    ),
    ont: List[Ont] = typer.Option(
        [Ont.BP, Ont.CC, Ont.MF],
        "--ont",
        case_sensitive=False,
        help="GO ontology branch(es) to run. Repeat the flag to pick a subset, "
             "e.g. `--ont BP --ont MF`. Defaults to running all three.",
    ),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Batch size."),
    mid_dir: Path = typer.Option(
        Path("gophage_results"),
        "--mid-dir",
        help="Directory for intermediate + final outputs. "
             "Per-ontology subdirectories will be created inside it.",
    ),
    threshold: float = typer.Option(
        0.1,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Score cutoff applied to the merged long-format CSV and the "
             "per-ontology summary CSVs.",
    ),
    data_dir: Path = typer.Option(
        Path("."),
        "--data-dir",
        help="Root directory containing DataBase/, ESM_model/, PhaGO_model/, "
             "Protein_annotation/ and Term_label/.",
    ),
    device: Device = typer.Option(
        Device.auto,
        "--device",
        case_sensitive=False,
        help="Compute device: auto (use CUDA if available), cuda (force GPU, "
             "error if missing), or cpu.",
    ),
    diamond_threads: int = typer.Option(
        8, "--diamond-threads", min=1, help="Threads for diamond blastp."
    ),
    num_workers: int = typer.Option(
        2, "--num-workers", min=0, help="DataLoader worker processes for inference."
    ),
) -> None:
    """Run the full GOPhage+ pipeline (prodigal -> diamond -> ESM2 -> PhaGO)."""
    # mutually-exclusive input checks (done here for nice typer error messages)
    if (contigs is None) == (proteins is None):
        raise typer.BadParameter(
            "Provide exactly one of --contigs or --proteins."
        )
    if protein_contig_map is not None and proteins is None:
        raise typer.BadParameter("--protein-contig-map requires --proteins.")

    long_csv, per_ont_summaries = run_pipeline(
        contigs=contigs,
        proteins=proteins,
        protein_contig_map=protein_contig_map,
        plm=plm.value,
        ont_list=[o.value for o in ont],
        batch_size=batch_size,
        mid_dir=mid_dir,
        data_dir=data_dir,
        threshold=threshold,
        diamond_threads=diamond_threads,
        num_workers=num_workers,
        device=device.value,
    )

    typer.echo("\nDone.")
    typer.echo(f"  combined long-format CSV: {long_csv}")
    for s in per_ont_summaries:
        typer.echo(f"  per-ontology summary:    {s}")


if __name__ == "__main__":
    app()
