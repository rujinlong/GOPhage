"""Combined long-format output across all ontologies."""

from __future__ import annotations

import csv
from pathlib import Path

from gophage.go_expand import get_all_ancestors, parse_go_obo


PHAGO_SUFFIX = {"esm2-12": "base", "esm2-33": "large"}


def _per_ont_csv(mid_dir: Path, ont: str, plm: str) -> Path:
    suffix = PHAGO_SUFFIX[plm]
    return mid_dir / ont / f"{ont}_GOPhage_{suffix}_plus_prediction_labels.csv"


def write_long_format(
    mid_dir: Path,
    plm: str,
    ont_list: list[str],
    data_dir: Path,
    threshold: float,
) -> Path:
    """Combine every per-ontology prediction CSV (with ancestor expansion) into
    one long-format CSV. Within each (protein, ontology, go_term):
      * a `self` annotation always wins over an `ancestor` annotation;
      * within the same source, the maximum score is kept;
      * `seed_term` is the direct prediction that contributed the score.
    """
    obo_path = data_dir / "DataBase" / "go-basic.obo"
    if not obo_path.exists():
        raise FileNotFoundError(f"GO obo file not found: {obo_path}")
    GO_info = parse_go_obo(obo_path)

    # (protein, ontology, go_term) -> (score, seed_term, source)
    best: dict[tuple[str, str, str], tuple[float, str, str]] = {}

    for ont in ont_list:
        csv_path = _per_ont_csv(mid_dir, ont, plm)
        if not csv_path.exists():
            print(f"  warning: no predictions file for {ont}: {csv_path}")
            continue
        with csv_path.open() as fh:
            header = fh.readline()  # noqa: F841 (Proteins,GO Term,Scores)
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                protein, go_term, score_s = parts[0], parts[1], parts[2]
                try:
                    score = float(score_s)
                except ValueError:
                    continue
                if score < threshold:
                    continue

                # self
                key = (protein, ont, go_term)
                prev = best.get(key)
                if prev is None or prev[2] == "ancestor":
                    best[key] = (score, go_term, "self")
                elif score > prev[0]:
                    best[key] = (score, go_term, "self")

                # ancestors (don't overwrite a self entry)
                for anc in get_all_ancestors(go_term, GO_info):
                    key_a = (protein, ont, anc)
                    prev_a = best.get(key_a)
                    if prev_a is None:
                        best[key_a] = (score, go_term, "ancestor")
                    elif prev_a[2] == "ancestor" and score > prev_a[0]:
                        best[key_a] = (score, go_term, "ancestor")

    out = mid_dir / "gophage_predictions_long.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "protein",
            "ontology",
            "go_term",
            "go_name",
            "namespace",
            "score",
            "source",
            "seed_term",
        ])
        for (protein, ont, go_term), (score, seed, source) in sorted(best.items()):
            info = GO_info.get(go_term, {"name": "", "namespace": ""})
            writer.writerow([
                protein,
                ont,
                go_term,
                info.get("name", ""),
                info.get("namespace", ""),
                f"{score:.6f}",
                source,
                seed,
            ])

    print(f"Combined long-format predictions: {out}")
    return out
