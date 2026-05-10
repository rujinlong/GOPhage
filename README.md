
![icon](https://github.com/jiaojiaoguan/GOPhage/blob/main/gophage.png)

# Overview

GOPhage is a learning-based model for annotating phage proteins with Gene Ontology (GO) terms. The major improvements come from exploiting phage genomic context and a protein language foundation model (ESM2). A Transformer encoder learns the relationship of context proteins, and the predictions are late-fused with a DiamondBLASTp similarity baseline (GOPhage+). Two PLM sizes are supported: `esm2-12` (base) and `esm2-33` (large).

This branch ships GOPhage as an installable Python package with a [Typer](https://typer.tiangolo.com/) CLI (`gophage`) and a Dockerfile that builds on `mambaorg/micromamba:cuda12.6.3-ubuntu24.04` for one-shot deployment on HPC clusters.

## What you need

1. **The GOPhage data bundle** (databases + ESM2 weights + PhaGO weights + GO obo file). Download once and bind-mount it into the container.
2. **GPU drivers** on the host if you want CUDA inference (the container already ships CUDA 12.6 runtime libs and CUDA-enabled PyTorch). The CLI also supports `--device cpu`.

### Download the data bundle

#### Google Drive
https://drive.google.com/drive/folders/14IQ75pMW9FK0H4mwleGEAo6_M7vOJeG5?usp=sharing

#### Baidu NetDisk
链接: https://pan.baidu.com/s/1vzT4FeaCfNyilffra5dagg?pwd=phag 提取码: `phag`

After extracting, the directory should look like:

```
<DATA_DIR>/
├── DataBase/                     # diamond .dmnd files + go-basic.obo
├── ESM_model/facebook/           # esm2_t12_35M_UR50D/, esm2_t33_650M_UR50D/
├── PhaGO_model/                  # *_PhaGO_base_model.th, *_PhaGO_large_model.th, threshold csvs
├── Protein_annotation/           # {BP,CC,MF}_known_proteins.csv
└── Term_label/                   # {BP,CC,MF}_term.pkl
```

Pass that root path via `--data-dir` (defaults to the current directory). Missing files are reported up-front before any expensive step runs.

## Install — option A: uv (recommended for local / HPC head node)

```bash
git clone https://github.com/jiaojiaoguan/GOPhage.git
cd GOPhage

# 1. install prodigal + diamond (system tools) into a conda env
conda env create -f gophage.yaml -n gophage   # or: micromamba env create -f gophage.yaml
conda activate gophage

# 2. install the Python package + GPU PyTorch into that env via uv
uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch
uv pip install -e .
```

## Install — option B: Docker (recommended for HPC compute nodes)

```bash
docker build -t gophage:latest .

# Run it; mount the data bundle to /work and a result dir under /work/results.
docker run --gpus all --rm \
  -v /path/to/data_bundle:/work \
  -v $(pwd)/results:/work/results \
  gophage:latest run \
    --contigs /work/test.fasta \
    --plm esm2-12 \
    --mid-dir /work/results \
    --data-dir /work
```

For SLURM, wrap the same command in `srun --gpus=1 ...` or use a Singularity/Apptainer build derived from the same image.

## Usage

```text
gophage --help
gophage run --help
```

The pipeline runs prodigal (or skips it, see below) → ESM2 embedding → diamond blastp → PhaGO Transformer → late fusion with diamond → per-term thresholding → GO ancestor expansion → merged long-format CSV.

### Two input modes

| Flag | Behaviour |
|---|---|
| `--contigs DNA.fa` | Run prodigal to call ORFs, then run the rest of the pipeline. |
| `--proteins proteins.fa` | **Skip prodigal.** Use your existing protein annotation directly — IDs and sequences are preserved, so downstream merges with your other tooling are trivial. |

When using `--proteins`, GOPhage groups proteins by contig in one of two ways:

* **Default**: contig name is inferred by stripping the trailing `_<idx>` from each protein id (the prodigal convention, e.g. `NC_104266.1_3` → `NC_104266.1`).
* **`--protein-contig-map mapping.tsv`**: explicit two-column TSV (`protein_id<TAB>contig_id`). Use this when your IDs do not follow the prodigal pattern (NCBI WP_*, locus tags, etc.).

In both cases proteins must appear in the FASTA in **genome order** — GOPhage relies on the spatial relationship between adjacent proteins on the same contig.

### Ontologies

`--ont` defaults to running all three branches (`BP CC MF`) in a single invocation, sharing prodigal output and the ESM2 embeddings across them so the expensive steps run only once. Pick a subset with `--ont BP --ont MF`.

### Examples

```bash
# (A) DNA contigs, run all three ontologies (default), GPU autodetect
gophage run \
  --contigs test.fasta \
  --plm esm2-12 \
  --mid-dir BP_CC_MF_results \
  --data-dir /path/to/data_bundle

# (B) skip prodigal, use my existing protein annotation
gophage run \
  --proteins my_phage_proteins.faa \
  --protein-contig-map my_protein_to_contig.tsv \
  --ont BP --ont MF \
  --plm esm2-33 \
  --mid-dir results/ \
  --device cpu \
  --data-dir /path/to/data_bundle
```

### Outputs

Inside `--mid-dir` you will find:

```
<mid_dir>/
├── test_protein.fa                       # prodigal output (or copy of --proteins)
├── test_contig_sentence.csv              # contig → proteins index
├── esm12_per_residual_embedding/         # cached, reused on re-runs
├── BP/  CC/  MF/                         # one work-dir per ontology
│   ├── <ONT>_GOPhage_<base|large>_plus_prediction_labels.csv
│   └── <ONT>_GOPhage_<base|large>_plus_prediction_labels_summary.csv
└── gophage_predictions_long.csv          # merged across all ontologies
```

The **merged `gophage_predictions_long.csv`** is the recommended starting point for downstream analysis. Columns:

| column | meaning |
|---|---|
| `protein` | protein id (yours when `--proteins`, prodigal's otherwise) |
| `ontology` | `BP` / `CC` / `MF` |
| `go_term` | GO ID |
| `go_name` | human-readable name |
| `namespace` | `biological_process` / `cellular_component` / `molecular_function` |
| `score` | model score (after diamond fusion) |
| `source` | `self` (predicted directly) or `ancestor` (propagated through is_a / part_of) |
| `seed_term` | which directly-predicted GO term contributed this row |

Within each `(protein, ontology, go_term)` triple, `self` annotations always override `ancestor` propagations and the maximum score is kept. Filter, group, and join in pandas / dplyr as you wish:

```python
import pandas as pd
df = pd.read_csv("gophage_predictions_long.csv")
df_self = df[df.source == "self"]                  # only direct predictions
df_top = df.groupby("protein").score.max()         # best score per protein
```

The legacy per-ontology files are still written, so anything that consumed `<ONT>_GOPhage_*_summary.csv` keeps working.

### Re-running on the same `--mid-dir`

ESM2 per-residue embeddings are the slowest step. They are cached as one `.pkl` per protein under `<mid_dir>/esm{12,33}_per_residual_embedding/`; re-running with the same `--mid-dir` skips proteins whose embedding file already exists. Delete that directory if you want to force a re-embedding.

## Contact

If you have any questions, please email: jiaojguan2-c@my.cityu.edu.hk
