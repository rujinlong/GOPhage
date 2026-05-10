"""Contig preprocessing: prodigal + diamond blastp."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from Bio import SeqIO


def _write_contig_sentence(
    contig_to_proteins: dict[str, list[str]],
    contig_order: list[str],
    out_path: Path,
) -> None:
    """Write the contig-sentence CSV preserving the original line shape used
    by the rest of the pipeline (1-based protein columns after the contig)."""
    with out_path.open("w") as fh:
        for k in contig_order:
            v = contig_to_proteins[k]
            fh.write(k + ",")
            for v1 in v[:-1]:
                fh.write(v1 + ",")
            fh.write(v[-1] + "\n")


def _infer_contig_from_protein_id(protein_name: str) -> str:
    """Strip trailing `_<idx>` from a prodigal-style protein id."""
    parts = protein_name.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot infer contig name from protein id {protein_name!r}. "
            "Provide --protein-contig-map (TSV: protein_id<TAB>contig_id)."
        )
    return "_".join(parts[:-1])


def _load_protein_contig_map(mapping_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with mapping_file.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                # tolerate whitespace-separated (still TSV-friendly for downstream)
                parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Bad line in {mapping_file}: {raw!r}. "
                    "Expected `protein_id<TAB>contig_id`."
                )
            mapping[parts[0]] = parts[1]
    return mapping


def translate_contigs_into_proteins(
    input_fasta: Path, output_protein: Path, mid_dir: Path
) -> tuple[Path, Path]:
    """Run prodigal and write a contig sentence CSV.

    Returns (protein_fasta, contig_sentence_csv).
    """
    if shutil.which("prodigal") is None:
        raise RuntimeError(
            "prodigal not found in PATH. Install via conda/micromamba: "
            "`micromamba install -c bioconda prodigal`."
        )

    cmd = [
        "prodigal",
        "-i", str(input_fasta),
        "-a", str(output_protein),
        "-f", "gff",
        "-p", "meta",
    ]
    print("Running prodigal ...")
    subprocess.check_call(cmd)

    print("Encoding the phage genome sentence ...")

    contig_sentence = mid_dir / "test_contig_sentence.csv"
    contig_to_proteins: dict[str, list[str]] = {}
    contig_order: list[str] = []

    for records in SeqIO.parse(str(output_protein), format="fasta"):
        protein_name = str(records.id)
        contig_name = _infer_contig_from_protein_id(protein_name)
        if contig_name not in contig_to_proteins:
            contig_to_proteins[contig_name] = []
            contig_order.append(contig_name)
        contig_to_proteins[contig_name].append(protein_name)

    _write_contig_sentence(contig_to_proteins, contig_order, contig_sentence)

    return output_protein, contig_sentence


def build_contig_sentence_from_proteins(
    protein_fasta: Path,
    mid_dir: Path,
    mapping_file: Path | None = None,
) -> Path:
    """Group user-provided proteins by contig (preserving fasta order).

    If `mapping_file` is given (TSV: protein_id<TAB>contig_id), every protein in
    the fasta must be present in the mapping. Otherwise contig names are
    inferred by stripping the trailing `_<idx>` from each protein id.
    """
    print("Building contig sentence from --proteins input ...")
    mapping = _load_protein_contig_map(mapping_file) if mapping_file else None

    contig_to_proteins: dict[str, list[str]] = {}
    contig_order: list[str] = []

    for record in SeqIO.parse(str(protein_fasta), "fasta"):
        protein_name = str(record.id)
        if mapping is not None:
            if protein_name not in mapping:
                raise KeyError(
                    f"Protein {protein_name!r} not found in --protein-contig-map; "
                    "every protein in the fasta must be mapped."
                )
            contig_name = mapping[protein_name]
        else:
            contig_name = _infer_contig_from_protein_id(protein_name)

        if contig_name not in contig_to_proteins:
            contig_to_proteins[contig_name] = []
            contig_order.append(contig_name)
        contig_to_proteins[contig_name].append(protein_name)

    contig_sentence = mid_dir / "test_contig_sentence.csv"
    _write_contig_sentence(contig_to_proteins, contig_order, contig_sentence)
    return contig_sentence


def check_number_protein(contig_sentence: Path, ont: str, work_dir: Path) -> Path:
    """Split overly long contig sentences into max-length sub-sentences."""
    dict_ontology_length = {"CC": 55, "BP": 17, "MF": 59}
    max_length = dict_ontology_length[ont]

    new_contig_sentence_name = work_dir / "test_contig_sentence_new.csv"

    with contig_sentence.open() as src, new_contig_sentence_name.open("w") as dst:
        for lines in src:
            line = lines.strip().split(",")
            contig_name = line[0]
            proteins_all = line[1:]

            final_proteins = [p for p in proteins_all if p != ""]
            protein_number = len(final_proteins)

            if protein_number > max_length:
                subsentences = [
                    proteins_all[i : i + max_length]
                    for i in range(0, len(proteins_all), max_length)
                ]
                for index in range(len(subsentences)):
                    dst.write(contig_name + "_" + str(index) + ",")
                    for j in subsentences[index]:
                        dst.write(j + ",")
                    dst.write("\n")
            else:
                dst.write(lines)

    return new_contig_sentence_name


def run_diamond_blastp_alignment(
    input_protein_fasta: Path,
    ont: str,
    work_dir: Path,
    data_dir: Path,
    threads: int = 8,
) -> Path:
    """Run diamond blastp and return the output txt path."""
    if shutil.which("diamond") is None:
        raise RuntimeError(
            "diamond not found in PATH. Install via conda/micromamba: "
            "`micromamba install -c bioconda diamond`."
        )

    database = data_dir / "DataBase" / f"{ont}_database.dmnd"
    if not database.exists():
        raise FileNotFoundError(
            f"Diamond database not found: {database}. "
            "See README for how to download the GOPhage data bundle."
        )

    output_file = work_dir / f"test_against_{ont}_database.txt"
    cmd = [
        "diamond", "blastp",
        "-d", str(database),
        "-q", str(input_protein_fasta),
        "-o", str(output_file),
        "-p", str(threads),
        "--sensitive",
    ]
    print("Running Diamond Blastp ...")
    subprocess.check_call(cmd)
    return output_file
