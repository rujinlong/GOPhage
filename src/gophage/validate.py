"""Fail-fast validation of the GOPhage data bundle layout."""

from __future__ import annotations

from pathlib import Path

ESM_DIR = {
    "esm2-12": Path("ESM_model/facebook/esm2_t12_35M_UR50D"),
    "esm2-33": Path("ESM_model/facebook/esm2_t33_650M_UR50D"),
}

PHAGO_SUFFIX = {"esm2-12": "base", "esm2-33": "large"}
PHAGO_THRESHOLD_PREFIX = {"esm2-12": "esm12", "esm2-33": "esm33"}


def validate_data_dir(data_dir: Path, plm: str, ont_list: list[str]) -> None:
    """Check that every file the pipeline will read exists.

    Raises FileNotFoundError listing every missing path.
    """
    if plm not in ESM_DIR:
        raise ValueError(f"Unknown plm: {plm}")

    required: list[Path] = [
        data_dir / "DataBase" / "go-basic.obo",
        data_dir / ESM_DIR[plm],
    ]

    suffix = PHAGO_SUFFIX[plm]
    prefix = PHAGO_THRESHOLD_PREFIX[plm]

    for ont in ont_list:
        required.extend([
            data_dir / "DataBase" / f"{ont}_database.dmnd",
            data_dir / "Protein_annotation" / f"{ont}_known_proteins.csv",
            data_dir / "Term_label" / f"{ont}_term.pkl",
            data_dir / "PhaGO_model" / f"{ont}_PhaGO_{suffix}_model.th",
            data_dir / "PhaGO_model" / f"{prefix}_{ont}_label_threshold.csv",
        ])

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            f"GOPhage data bundle is incomplete under {data_dir}. "
            f"Missing files:{joined}\n"
            "See README for download instructions."
        )
