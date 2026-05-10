"""GOPhage Transformer inference + diamond fusion + final CSV output.

`work_dir` is the per-ontology subdirectory; intermediate and output files for
this ontology live there. Model weights and threshold tables come from
`data_dir` (the GOPhage data bundle root).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch as th
from torch import nn

from gophage.dataset import get_combine_loader_cut_annot
from gophage.model import PhaGO_model


N_TERMS = {"BP": 126, "MF": 165, "CC": 23}
ONTOLOGY_LENGTH = {"CC": 55, "BP": 17, "MF": 59}


def run_phaGO_model(
    plm_model_name: str,
    ont: str,
    batch_size: int,
    work_dir: Path,
    data_dir: Path,
    device: torch.device,
    num_workers: int = 2,
) -> Path:
    """Run PhaGO Transformer; return the path of the prediction pickle."""
    if plm_model_name == "esm2-12":
        nhead = 12
        d_model = 480
        dim_feedforward = 128 if ont == "CC" else 480
        model_name = data_dir / "PhaGO_model" / f"{ont}_PhaGO_base_model.th"
        out_put_file = work_dir / f"{ont}_phago_base_results.pkl"
        data_file_input = work_dir / "test_location_esm12.csv"
    elif plm_model_name == "esm2-33":
        nhead = 16
        d_model = 1280
        dim_feedforward = 320 if ont == "CC" else 1280
        model_name = data_dir / "PhaGO_model" / f"{ont}_PhaGO_large_model.th"
        out_put_file = work_dir / f"{ont}_phago_large_results.pkl"
        data_file_input = work_dir / "test_location_esm33.csv"
    else:
        raise ValueError(f"Unknown PLM model name: {plm_model_name}")

    if not model_name.exists():
        raise FileNotFoundError(
            f"PhaGO model weights not found: {model_name}. "
            "See README for how to download the GOPhage data bundle."
        )

    length = ONTOLOGY_LENGTH[ont]

    test_loader = get_combine_loader_cut_annot(
        data_file=str(data_file_input),
        batch_size=batch_size,
        num_workers=num_workers,
        ont=ont,
        dict_ontology_length=ONTOLOGY_LENGTH,
        plm_model=plm_model_name,
    )

    net = PhaGO_model(
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=1,
        dropout=0.5,
        num_labels=N_TERMS[ont],
        d_model=d_model,
        vocab_size=length,
    )
    net.load_state_dict(torch.load(str(model_name), map_location=device))

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!\n")
        net = nn.DataParallel(net)

    net.to(device)
    net.eval()

    test_preds = []
    test_all_proteins = []
    with th.no_grad():
        for batch_idx, (embedding, test_proteins) in enumerate(test_loader):
            test_proteins = np.array(test_proteins)
            test_proteins = np.transpose(test_proteins)

            test_extract_protein = [
                protein
                for index, protein in enumerate(test_proteins.flatten())
                if protein != "none"
            ]
            for i in test_extract_protein:
                test_all_proteins.append(i)

            test_batch_features = embedding.to(device)
            test_logits = net(test_batch_features)

            for i, row in enumerate(test_proteins):
                for protein in row:
                    if protein != "none":
                        index = list(row).index(protein)
                        test_preds.append(test_logits[i][index].tolist())

        test_results = {
            "prediction": test_preds,
            "protein_name": test_all_proteins,
        }
        with out_put_file.open("wb") as handle:
            pickle.dump(test_results, handle)

    return out_put_file


def combine_diamondblastp_phaGO(
    plm_model_name: str, ont: str, work_dir: Path
) -> Path:
    """Late-fuse diamond and PhaGO scores. Returns the fused pickle path."""
    if plm_model_name == "esm2-12":
        dict_onyology_alpha = {"BP": 1.0, "CC": 0.83, "MF": 0.91}
        output_results = work_dir / f"{ont}_phago_base_plus_results.pkl"
        phago_prediction_results = work_dir / f"{ont}_phago_base_results.pkl"
    elif plm_model_name == "esm2-33":
        dict_onyology_alpha = {"BP": 0.9, "CC": 0.62, "MF": 0.82}
        output_results = work_dir / f"{ont}_phago_large_plus_results.pkl"
        phago_prediction_results = work_dir / f"{ont}_phago_large_results.pkl"
    else:
        raise ValueError(f"Unknown PLM model name: {plm_model_name}")

    alpha_parameter = dict_onyology_alpha[ont]

    diamond_blastp_result_file = work_dir / f"{ont}_test_diamondblastp_results.pkl"
    test_df = pd.read_pickle(diamond_blastp_result_file)
    diamond_blastp_preds = test_df["prediction"]
    diamond_protein_name = test_df["protein_name"]

    phago_df = pd.read_pickle(phago_prediction_results)
    phago_proteins = phago_df["protein_name"]
    phago_prediction = phago_df["prediction"]

    context_dl_protein_prediction = {
        phago_proteins[i]: phago_prediction[i] for i in range(len(phago_proteins))
    }

    diamondscore_protein_prediction: dict[str, list[float]] = {}
    for index in range(len(diamond_blastp_preds)):
        dia_pred = diamond_blastp_preds[index]
        pro = diamond_protein_name[index]
        prediction_diamondscore = list(dia_pred)
        preds = [int(i) for i in prediction_diamondscore]
        if all(x == 0 for x in preds):
            continue
        diamondscore_protein_prediction[pro] = prediction_diamondscore

    all_phagoplus = []
    proteins = []
    for k, v in context_dl_protein_prediction.items():
        proteins.append(k)
        if k in diamondscore_protein_prediction:
            diamondscore = diamondscore_protein_prediction[k]
            diamondscore = [i * alpha_parameter for i in diamondscore]
            phago_score = [i * (1 - alpha_parameter) for i in v]
            phagoplus = [
                diamondscore[i] + phago_score[i] for i in range(len(diamondscore))
            ]
        else:
            phagoplus = v
        all_phagoplus.append(phagoplus)

    df = pd.DataFrame({"protein_name": proteins, "prediction": all_phagoplus})
    df.to_pickle(output_results, protocol=4)
    return output_results


def output_the_prediction_results(
    plm_model_name: str, ont: str, work_dir: Path, data_dir: Path
) -> Path:
    """Apply per-term thresholds and write the final per-prediction CSV."""
    if plm_model_name == "esm2-12":
        phagoplus_prediction_results = work_dir / f"{ont}_phago_base_plus_results.pkl"
        results_csv_name = work_dir / f"{ont}_GOPhage_base_plus_prediction_labels.csv"
        threshold_file = data_dir / "PhaGO_model" / f"esm12_{ont}_label_threshold.csv"
    elif plm_model_name == "esm2-33":
        phagoplus_prediction_results = work_dir / f"{ont}_phago_large_plus_results.pkl"
        results_csv_name = work_dir / f"{ont}_GOPhage_large_plus_prediction_labels.csv"
        threshold_file = data_dir / "PhaGO_model" / f"esm33_{ont}_label_threshold.csv"
    else:
        raise ValueError(f"Unknown PLM model name: {plm_model_name}")

    if not threshold_file.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}")

    test_df = pd.read_pickle(phagoplus_prediction_results)
    phago_plus_preds = test_df["prediction"]
    phago_plus_protein = test_df["protein_name"]

    terms_file = data_dir / "Term_label" / f"{ont}_term.pkl"
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df["terms"].values.flatten()
    terms_dict = {i: v for i, v in enumerate(terms)}

    dict_label_threshold: dict[str, float] = {}
    with threshold_file.open() as fh:
        next(fh)  # header
        for lines in fh:
            line = lines.strip().split(",")
            dict_label_threshold[line[0]] = float(line[1])

    with results_csv_name.open("w") as fh:
        fh.write("Proteins,GO Term,Scores\n")
        for j in range(len(phago_plus_protein)):
            p = phago_plus_protein[j]
            for indice, score in enumerate(phago_plus_preds[j]):
                go_term = terms_dict[indice]
                cutoff = dict_label_threshold[go_term]
                if score > cutoff:
                    fh.write(p + "," + go_term + "," + str(score) + "\n")

    return results_csv_name
