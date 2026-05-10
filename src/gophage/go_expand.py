"""Expand a GO-prediction CSV with all is_a / part_of ancestors."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path


def parse_go_obo(go_obo_path: Path) -> dict:
    """Parse go.obo, keep id, name, namespace, is_a + part_of relations."""
    GO_info: dict = {}
    current_id = None
    with go_obo_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
                GO_info[current_id] = {"name": "", "namespace": "", "parents": []}
            elif current_id:
                if line.startswith("name:"):
                    GO_info[current_id]["name"] = line.split("name: ")[1]
                elif line.startswith("namespace:"):
                    GO_info[current_id]["namespace"] = line.split("namespace: ")[1]
                elif line.startswith("is_a:"):
                    GO_info[current_id]["parents"].append(
                        line.split("is_a: ")[1].split()[0]
                    )
                elif "relationship: part_of" in line:
                    GO_info[current_id]["parents"].append(
                        line.split("relationship: part_of ")[1].split()[0]
                    )
    return GO_info


def get_all_ancestors(go_id: str, GO_info: dict) -> set[str]:
    """Recursively gather all is_a + part_of ancestors."""
    ancestors: set[str] = set()
    queue = deque(GO_info.get(go_id, {}).get("parents", []))
    while queue:
        parent = queue.popleft()
        if parent not in ancestors:
            ancestors.add(parent)
            queue.extend(GO_info.get(parent, {}).get("parents", []))
    return ancestors


def expand_predictions(
    input_csv: Path,
    data_dir: Path,
    cutoff: float = 0.1,
    out_summary: Path | None = None,
) -> Path:
    """Expand predictions with ancestors and write a per-protein summary CSV."""
    obo_path = data_dir / "DataBase" / "go-basic.obo"
    if not obo_path.exists():
        raise FileNotFoundError(f"GO obo file not found: {obo_path}")

    GO_info = parse_go_obo(obo_path)

    protein_to_go: dict[str, list] = defaultdict(list)
    with input_csv.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        # Honour either tab- or comma-separated, like the original
        if "\t" in first_line:
            reader = csv.DictReader(f, delimiter="\t")
        else:
            reader = csv.DictReader(f)
        for row in reader:
            score = float(row["Scores"])
            if score >= cutoff:
                protein = row["Proteins"]
                go_id = row["GO Term"]
                protein_to_go[protein].append((go_id, score))

    summary_rows = []
    for protein, go_list in protein_to_go.items():
        seen_terms: set[str] = set()
        all_terms: dict[str, str] = {}

        for go_id, _score in go_list:
            if go_id not in seen_terms:
                info = GO_info.get(go_id, {"name": "", "namespace": ""})
                seen_terms.add(go_id)
                all_terms[go_id] = info["name"]

            ancestors = get_all_ancestors(go_id, GO_info)
            for anc_id in ancestors:
                if anc_id not in seen_terms:
                    info = GO_info.get(anc_id, {"name": "", "namespace": ""})
                    seen_terms.add(anc_id)
                    all_terms[anc_id] = info["name"]

        ancestor_summary = "; ".join(f"{k}: {v}" for k, v in all_terms.items())
        summary_rows.append([protein, ancestor_summary])

    if out_summary is None:
        out_summary = input_csv.with_name(input_csv.stem + "_summary.csv")

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Protein_ID", "GO_terms_and_Ancestors"])
        writer.writerows(summary_rows)

    print(f"Summary file written to: {out_summary}")
    return out_summary
