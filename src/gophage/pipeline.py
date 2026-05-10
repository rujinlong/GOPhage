"""End-to-end GOPhage pipeline orchestration.

Layout under `mid_dir`:

    mid_dir/
    ├── test_protein.fa                  # prodigal output OR copy of --proteins
    ├── test_contig_sentence.csv         # raw, ontology-agnostic
    ├── esm12_per_residual_embedding/    # shared ESM2 embeddings (cached)
    ├── BP/  CC/  MF/                    # one work_dir per ontology
    │   ├── test_against_<ONT>_database.txt
    │   ├── <ONT>_test_diamondblastp_results.pkl
    │   ├── test_contig_sentence_new.csv
    │   ├── sequence_embedding_<plm>/
    │   ├── protein_names/
    │   ├── test_location_<plm>.csv
    │   ├── <ONT>_phago_<base|large>_results.pkl
    │   ├── <ONT>_phago_<base|large>_plus_results.pkl
    │   ├── <ONT>_GOPhage_<base|large>_plus_prediction_labels.csv
    │   └── <ONT>_GOPhage_<base|large>_plus_prediction_labels_summary.csv
    └── gophage_predictions_long.csv     # merged across ontologies
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch

from gophage.context import (
    get_protein_names,
    get_sequence_location,
    preparing_context_protein_embedding,
)
from gophage.diamond import get_diamondscore
from gophage.embedding import embedding_proteins_ESM2
from gophage.go_expand import expand_predictions
from gophage.inference import (
    combine_diamondblastp_phaGO,
    output_the_prediction_results,
    run_phaGO_model,
)
from gophage.merged_output import write_long_format
from gophage.preprocess import (
    build_contig_sentence_from_proteins,
    check_number_protein,
    run_diamond_blastp_alignment,
    translate_contigs_into_proteins,
)
from gophage.validate import validate_data_dir


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the user-supplied device string to a `torch.device`."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device=cuda requested but no CUDA device is available. "
                "Use --device=auto or --device=cpu."
            )
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device_arg}")


def run_pipeline(
    contigs: Path | None,
    proteins: Path | None,
    protein_contig_map: Path | None,
    plm: str,
    ont_list: list[str],
    batch_size: int,
    mid_dir: Path,
    data_dir: Path,
    threshold: float,
    diamond_threads: int = 8,
    num_workers: int = 2,
    device: str | torch.device = "auto",
) -> tuple[Path, list[Path]]:
    """Run the full GOPhage+ pipeline. Returns (long_csv, per_ont_summary_csvs)."""
    start_time = time.time()

    # input validation
    if contigs is None and proteins is None:
        raise ValueError("Provide --contigs or --proteins (one of them).")
    if contigs is not None and proteins is not None:
        raise ValueError("--contigs and --proteins are mutually exclusive.")
    if protein_contig_map is not None and proteins is None:
        raise ValueError("--protein-contig-map requires --proteins.")
    if not ont_list:
        raise ValueError("--ont must list at least one ontology.")

    mid_dir = Path(mid_dir)
    data_dir = Path(data_dir)
    mid_dir.mkdir(parents=True, exist_ok=True)

    # fail-fast on missing data bundle
    validate_data_dir(data_dir, plm, ont_list)

    if isinstance(device, str):
        device = resolve_device(device)
    print(f"Using device: {device}")

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # already set in this process; harmless
        pass

    # === ont-agnostic shared steps =========================================

    protein_fasta = mid_dir / "test_protein.fa"

    if contigs is not None:
        contigs = Path(contigs)
        protein_fasta, contig_sentence = translate_contigs_into_proteins(
            input_fasta=contigs,
            output_protein=protein_fasta,
            mid_dir=mid_dir,
        )
    else:
        proteins = Path(proteins)  # type: ignore[arg-type]
        # copy user-provided fasta to the canonical filename so downstream
        # caching paths stay stable across re-runs
        if protein_fasta.resolve() != proteins.resolve():
            shutil.copy(proteins, protein_fasta)
        contig_sentence = build_contig_sentence_from_proteins(
            protein_fasta=protein_fasta,
            mid_dir=mid_dir,
            mapping_file=protein_contig_map,
        )

    # ESM2 per-protein embeddings (shared across ontologies, cached)
    embedding_proteins_ESM2(
        plm_model_name=plm,
        mid_dir=mid_dir,
        data_dir=data_dir,
        device=device,
        protein_fasta=protein_fasta,
    )

    # === ont-specific loop ================================================

    per_ont_summaries: list[Path] = []
    for ont in ont_list:
        print(f"\n=== ontology: {ont} ===")
        work_dir = mid_dir / ont
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. diamond blastp + diamond-score baseline
        run_diamond_blastp_alignment(
            input_protein_fasta=protein_fasta,
            ont=ont,
            work_dir=work_dir,
            data_dir=data_dir,
            threads=diamond_threads,
        )
        get_diamondscore(
            ont=ont,
            protein_fasta=protein_fasta,
            work_dir=work_dir,
            data_dir=data_dir,
        )

        # 2. ontology-specific contig-sentence split + per-contig embedding
        cs_split = check_number_protein(
            contig_sentence=contig_sentence, ont=ont, work_dir=work_dir
        )
        preparing_context_protein_embedding(
            plm_model_name=plm,
            contig_sentence=cs_split,
            work_dir=work_dir,
            embedding_root=mid_dir,
        )
        get_protein_names(contig_sentence=cs_split, work_dir=work_dir)
        get_sequence_location(
            model_name=plm,
            contig_sentence=cs_split,
            work_dir=work_dir,
            embedding_root=mid_dir,
        )

        # 3. PhaGO inference + diamond fusion + threshold filter
        run_phaGO_model(
            plm_model_name=plm,
            ont=ont,
            batch_size=batch_size,
            work_dir=work_dir,
            data_dir=data_dir,
            device=device,
            num_workers=num_workers,
        )
        combine_diamondblastp_phaGO(plm_model_name=plm, ont=ont, work_dir=work_dir)
        results_csv = output_the_prediction_results(
            plm_model_name=plm, ont=ont, work_dir=work_dir, data_dir=data_dir
        )

        # 4. legacy per-ontology summary CSV (with ancestor expansion)
        summary_name = results_csv.with_name(results_csv.stem + "_summary.csv")
        expand_predictions(
            input_csv=results_csv,
            data_dir=data_dir,
            cutoff=threshold,
            out_summary=summary_name,
        )
        per_ont_summaries.append(summary_name)

    # === merged long-format output ========================================

    long_csv = write_long_format(
        mid_dir=mid_dir,
        plm=plm,
        ont_list=ont_list,
        data_dir=data_dir,
        threshold=threshold,
    )

    spend_time = (time.time() - start_time) / 60
    print(f"Running time: {spend_time:.2f} min")
    return long_csv, per_ont_summaries
