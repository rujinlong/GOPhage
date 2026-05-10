"""Diamond-score-based GO prediction (similarity baseline)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


def read_test_protein(protein_fasta: Path) -> list[str]:
    return [str(records.id) for records in SeqIO.parse(str(protein_fasta), "fasta")]


def get_diamondscore(
    ont: str, protein_fasta: Path, mid_dir: Path, data_dir: Path
) -> Path:
    """Compute Diamond-score-based GO prediction and pickle to mid_dir.

    Returns the pickle path.
    """
    print("Running the DiamondBlastp method to get the prediction score ...")
    n_terms = {"BP": 126, "MF": 165, "CC": 23}
    number = n_terms[ont]

    known_proteins = data_dir / "Protein_annotation" / f"{ont}_known_proteins.csv"
    if not known_proteins.exists():
        raise FileNotFoundError(f"Known proteins file not found: {known_proteins}")

    dict_train_protein_go: dict[str, set[str]] = {}
    with known_proteins.open() as fh:
        next(fh)  # skip header
        for lines in fh:
            line = lines.strip().split(",")
            train_protein = line[0]
            label = line[2].split(";")
            final_label = {l for l in label if l != ""}
            dict_train_protein_go[train_protein] = final_label

    diamond_scores: dict[str, dict[str, float]] = {}
    input_diamond_file = mid_dir / f"test_against_{ont}_database.txt"
    with input_diamond_file.open() as f:
        for line in f:
            it = line.strip().split("\t")
            diamond_scores.setdefault(it[0], {})
            diamond_scores[it[0]][it[1]] = float(it[-1])

    test_protein_names = read_test_protein(protein_fasta)

    blast_preds = []
    for test_protein in test_protein_names:
        annots: dict[str, float] = {}
        prot_id = test_protein

        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos: set[str] = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= dict_train_protein_go[p_id]
                total_score += score

            allgos_sorted = list(sorted(allgos))
            sim = np.zeros(len(allgos_sorted), dtype=np.float32)
            for j, go_id in enumerate(allgos_sorted):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in dict_train_protein_go[p_id]:
                        s += score
                sim[j] = s / total_score

            for go_id, score in zip(allgos_sorted, sim):
                annots[go_id] = score
        blast_preds.append(annots)

    terms_file = data_dir / "Term_label" / f"{ont}_term.pkl"
    if not terms_file.exists():
        raise FileNotFoundError(f"Term file not found: {terms_file}")
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df["terms"].values.flatten()
    dict_go_id = {v: i for i, v in enumerate(terms)}

    all_test_blast_preds_score = []
    for pre in blast_preds:
        pre_score = [0.0] * number
        for go, score in pre.items():
            index = int(dict_go_id[go])
            pre_score[index] = score
        all_test_blast_preds_score.append(pre_score)

    test_results = {
        "prediction": list(all_test_blast_preds_score),
        "protein_name": test_protein_names,
    }

    out_pkl = mid_dir / f"{ont}_test_diamondblastp_results.pkl"
    with out_pkl.open("wb") as handle:
        pickle.dump(test_results, handle)
    return out_pkl
