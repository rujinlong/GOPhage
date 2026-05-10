"""End-to-end GOPhage pipeline orchestration."""

from __future__ import annotations

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
from gophage.preprocess import (
    check_number_protein,
    run_diamond_blastp_alignment,
    translate_contigs_into_proteins,
)


def run_pipeline(
    contigs: Path,
    plm: str,
    ont: str,
    batch_size: int,
    mid_dir: Path,
    data_dir: Path,
    threshold: float,
    diamond_threads: int = 8,
    num_workers: int = 2,
) -> Path:
    """Run the full GOPhage+ pipeline. Returns the final summary CSV path."""
    start_time = time.time()

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # already set in this process; harmless
        pass

    mid_dir = Path(mid_dir)
    data_dir = Path(data_dir)
    contigs = Path(contigs)

    mid_dir.mkdir(parents=True, exist_ok=True)

    # 1. prodigal -> proteins + contig sentence
    protein_fasta = mid_dir / "test_protein.fa"
    protein_fasta, contig_sentence = translate_contigs_into_proteins(
        input_fasta=contigs,
        output_protein=protein_fasta,
        mid_dir=mid_dir,
    )

    # 2. diamond blastp
    run_diamond_blastp_alignment(
        input_protein_fasta=protein_fasta,
        ont=ont,
        mid_dir=mid_dir,
        data_dir=data_dir,
        threads=diamond_threads,
    )

    # 3. diamond-score baseline GO prediction
    get_diamondscore(
        ont=ont, protein_fasta=protein_fasta, mid_dir=mid_dir, data_dir=data_dir
    )

    # 4. ESM2 per-residue embedding
    embedding_proteins_ESM2(plm_model_name=plm, mid_dir=mid_dir, data_dir=data_dir)

    # 5. split overly-long contigs + assemble context embeddings
    contig_sentence = check_number_protein(
        contig_sentence=contig_sentence, ont=ont, mid_dir=mid_dir
    )
    preparing_context_protein_embedding(
        plm_model_name=plm, contig_sentence=contig_sentence, mid_dir=mid_dir
    )
    get_protein_names(contig_sentence=contig_sentence, mid_dir=mid_dir)
    get_sequence_location(
        model_name=plm, contig_sentence=contig_sentence, mid_dir=mid_dir
    )

    # 6. PhaGO inference
    run_phaGO_model(
        plm_model_name=plm,
        ont=ont,
        batch_size=batch_size,
        mid_dir=mid_dir,
        data_dir=data_dir,
        num_workers=num_workers,
    )

    # 7. fuse with diamond
    combine_diamondblastp_phaGO(plm_model_name=plm, ont=ont, mid_dir=mid_dir)

    # 8. apply per-term thresholds, write CSV
    results_csv_name = output_the_prediction_results(
        ont=ont, plm_model_name=plm, mid_dir=mid_dir, data_dir=data_dir
    )

    # 9. expand with GO ancestors
    summary_name = results_csv_name.with_name(results_csv_name.stem + "_summary.csv")
    expand_predictions(
        input_csv=results_csv_name,
        data_dir=data_dir,
        cutoff=threshold,
        out_summary=summary_name,
    )

    spend_time = (time.time() - start_time) / 60
    print(f"Running time: {spend_time:.2f} min")
    return summary_name
