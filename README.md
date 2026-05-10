
![icon](https://github.com/jiaojiaoguan/GOPhage/blob/main/gophage.png)

# Overview

GOPhage is a learning-based model for annotating phage proteins with Gene Ontology (GO) terms. The major improvements come from exploiting phage genomic context and a protein language foundation model (ESM2). A Transformer encoder learns the relationship of context proteins, and the predictions are late-fused with a DiamondBLASTp similarity baseline (GOPhage+). Two PLM sizes are supported: `esm2-12` (base) and `esm2-33` (large).

This branch ships GOPhage as an installable Python package with a [Typer](https://typer.tiangolo.com/) CLI (`gophage`) and a Dockerfile that builds on `mambaorg/micromamba:cuda12.6.3-ubuntu24.04` for one-shot deployment on HPC clusters.

## What you need

1. **The GOPhage data bundle** (databases + ESM2 weights + PhaGO weights + GO obo file). Download once and bind-mount it into the container.
2. **GPU drivers** on the host if you want CUDA inference (the container already ships CUDA 12.6 runtime libs and CUDA-enabled PyTorch).

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

Pass that root path to `--data-dir` (it defaults to the current directory, which keeps backward compatibility with the original layout).

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
    --ont BP \
    --mid-dir /work/results \
    --data-dir /work
```

For SLURM, wrap the same command in `srun --gpus=1 ...` or use a Singularity/Apptainer build derived from the same image.

## Usage

```text
gophage --help
gophage run --help
```

The pipeline runs prodigal → diamond blastp → ESM2 embedding → PhaGO Transformer → late fusion with diamond → per-term thresholding → GO ancestor expansion.

### Example

```bash
gophage run \
  --contigs test.fasta \
  --plm esm2-12 \
  --ont BP \
  --batch-size 8 \
  --mid-dir BP_results \
  --data-dir /path/to/data_bundle \
  --threshold 0.1
```

### Outputs

Inside `--mid-dir` you will find all intermediate artefacts plus the final CSVs:

| Model | Per-prediction CSV | Summary CSV |
|---|---|---|
| `esm2-12` | `<ONT>_GOPhage_base_plus_prediction_labels.csv` | `<ONT>_GOPhage_base_plus_prediction_labels_summary.csv` |
| `esm2-33` | `<ONT>_GOPhage_large_plus_prediction_labels.csv` | `<ONT>_GOPhage_large_plus_prediction_labels_summary.csv` |

The per-prediction CSV has columns `Proteins, GO Term, Scores`. The summary CSV has columns `Protein_ID, GO_terms_and_Ancestors` and contains the GO term predictions expanded with all `is_a` / `part_of` ancestors.

## Contact

If you have any questions, please email: jiaojguan2-c@my.cityu.edu.hk
