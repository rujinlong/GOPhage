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

The new modules in `src/gophage/` add functionality not present in the old
scripts:
* `validate.py` — fail-fast check of the GOPhage data bundle layout.
* `merged_output.py` — combined cross-ontology long-format CSV.
* `--proteins` / `--protein-contig-map` paths in `preprocess.py` skip prodigal
  when the user already has an annotated protein FASTA.
* `embedding.py` caches per-protein `.pkl` files; re-runs on the same
  `--mid-dir` skip already-embedded proteins.
* `--device {auto,cuda,cpu}` to support CPU-only HPC nodes.
