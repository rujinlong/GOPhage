# Archived files

These are the original pre-`uv/typer` scripts and the early `envs.yml`. They are
kept for historical reference only — do **not** use them. The maintained code
lives in `src/gophage/` and is invoked through the `gophage` CLI.

| Old file | Replaced by |
|---|---|
| `GOPhage.py` (argparse main) | `src/gophage/cli.py` + `src/gophage/pipeline.py` |
| `dataloder.py` | `src/gophage/dataset.py` |
| `model.py` | `src/gophage/model.py` |
| `get_esm_embedding.py` | `src/gophage/embedding.py` |
| `prepare_gophage_input.py` | `src/gophage/context.py` |
| `envs.yml` | `gophage.yaml` (top-level, cross-platform) |

Behaviour is preserved: model hyperparameters, alpha fusion weights, ontology
length tables, and per-term thresholds match the originals.
