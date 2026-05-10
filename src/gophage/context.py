"""Per-contig context-protein embedding assembly.

Per-protein ESM embeddings are read from `embedding_root` (shared across
ontologies); per-contig sequence embeddings, name lists, and the location CSV
are written into `work_dir` (ontology-specific subdirectory).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


def _plm_paths(plm_model_name: str) -> tuple[str, str]:
    """Return (per-protein dirname, sequence-embedding dirname suffix)."""
    if plm_model_name == "esm2-12":
        return "esm12_per_residual_embedding", "sequence_embedding_esm12"
    if plm_model_name == "esm2-33":
        return "esm33_per_residual_embedding", "sequence_embedding_esm33"
    raise ValueError(f"Unknown PLM model name: {plm_model_name}")


def preparing_context_protein_embedding(
    plm_model_name: str,
    contig_sentence: Path,
    work_dir: Path,
    embedding_root: Path,
) -> Path:
    """Concatenate per-protein embeddings into per-contig sequence embeddings.

    Returns the directory containing the per-contig .pt files.
    """
    print("Integrating the proteins embedding in the same contigs ...")
    per_protein_name, per_seq_name = _plm_paths(plm_model_name)
    protein_embedding_path = embedding_root / per_protein_name
    sequence_embedding_path = work_dir / per_seq_name

    if sequence_embedding_path.exists():
        shutil.rmtree(sequence_embedding_path)
    sequence_embedding_path.mkdir(parents=True)

    with contig_sentence.open() as fh:
        for lines in fh:
            line = lines.strip().split(",")
            sequence_name = line[0]
            proteins = [p for p in line[1:] if p != ""]

            if len(proteins) == 1:
                continue

            all_embedding = []
            for p in proteins:
                embedding = torch.load(
                    str(protein_embedding_path / f"{p}_embedding.pkl"),
                    map_location=torch.device("cpu"),
                )
                all_embedding.append(embedding["data"])

            sequence_embedding = torch.cat(all_embedding, dim=0)
            file_path = sequence_embedding_path / f"{sequence_name}_embedding.pt"
            torch.save({"data": sequence_embedding}, str(file_path))

    return sequence_embedding_path


def get_protein_names(contig_sentence: Path, work_dir: Path) -> Path:
    """Persist the protein-name list for each contig sentence (in work_dir)."""
    print("Preparing the protein names ...")

    protein_name_path = work_dir / "protein_names"
    if protein_name_path.exists():
        shutil.rmtree(protein_name_path)
    protein_name_path.mkdir(parents=True)

    with contig_sentence.open() as fh:
        for lines in fh:
            line = lines.strip().split(",")
            sequence_name = line[0]
            proteins = [p for p in line[1:] if p != ""]

            file_path_protein_name = protein_name_path / f"{sequence_name}_names.pt"
            torch.save({"proteins_name": proteins}, str(file_path_protein_name))

    print("Got sequence protein names successfully!")
    return protein_name_path


def get_sequence_location(
    model_name: str,
    contig_sentence: Path,
    work_dir: Path,
    embedding_root: Path,
) -> Path:
    """Write the (embedding_path, name_path) index CSV used by the dataloader.

    Multi-protein contigs reference per-contig embeddings under work_dir; the
    1-protein fallback references the shared per-protein embedding directly
    under embedding_root.
    """
    print("Preparing the location of the embedding and protein names ...")

    per_protein_name, per_seq_name = _plm_paths(model_name)
    sequence_embedding_path = work_dir / per_seq_name
    protein_embedding_path = embedding_root / per_protein_name

    if model_name == "esm2-12":
        location_csv = work_dir / "test_location_esm12.csv"
    else:
        location_csv = work_dir / "test_location_esm33.csv"

    protein_name_dir = work_dir / "protein_names"

    with contig_sentence.open() as src, location_csv.open("w") as dst:
        for lines in src:
            line = lines.strip().split(",")
            sequence_name = line[0]
            proteins = [p for p in line[1:] if p != ""]
            length = len(proteins)

            file_path_embedding = (
                sequence_embedding_path / f"{sequence_name}_embedding.pt"
            )
            file_path_protein_name = protein_name_dir / f"{sequence_name}_names.pt"

            if length == 1:
                protein_name = proteins[0]
                file_path_embedding = (
                    protein_embedding_path / f"{protein_name}_embedding.pkl"
                )

            dst.write(f"{file_path_embedding},{file_path_protein_name}\n")

    return location_csv
