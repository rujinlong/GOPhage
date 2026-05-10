"""Contig preprocessing: prodigal + diamond blastp."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from Bio import SeqIO


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
    dict_contig_proteins: dict[str, list[str]] = {}

    for records in SeqIO.parse(str(output_protein), format="fasta"):
        protein_name = str(records.id)
        protein_name_split = protein_name.split("_")
        contig_name = protein_name_split[0]

        for p in protein_name_split[1:-1]:
            contig_name = contig_name + "_" + p

        dict_contig_proteins.setdefault(contig_name, []).append(protein_name)

    with contig_sentence.open("w") as fh:
        for k, v in dict_contig_proteins.items():
            fh.write(k + ",")
            for v1 in v[:-1]:
                fh.write(v1 + ",")
            fh.write(v[-1] + "\n")

    return output_protein, contig_sentence


def check_number_protein(contig_sentence: Path, ont: str, mid_dir: Path) -> Path:
    """Split overly long contig sentences into max-length sub-sentences."""
    dict_ontology_length = {"CC": 55, "BP": 17, "MF": 59}
    max_length = dict_ontology_length[ont]

    new_contig_sentence_name = mid_dir / "test_contig_sentence_new.csv"

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
    mid_dir: Path,
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

    output_file = mid_dir / f"test_against_{ont}_database.txt"
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
