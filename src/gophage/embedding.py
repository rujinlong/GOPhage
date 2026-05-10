"""Per-residue ESM2 protein embeddings (cached)."""

from __future__ import annotations

from pathlib import Path

import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def get_data(fasta_file: Path) -> tuple[list[str], list[str]]:
    protein_sequence: list[str] = []
    all_id: list[str] = []
    for record in SeqIO.parse(str(fasta_file), "fasta"):
        protein_sequence.append(str(record.seq))
        all_id.append(str(record.id))
    return protein_sequence, all_id


class VirDataset(Dataset):
    def __init__(self, ids: list[str], seqs: list[str]):
        self.id = ids
        self.texts = seqs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.id[index]


def _resolve_paths(plm_model_name: str, mid_dir: Path, data_dir: Path):
    if plm_model_name == "esm2-12":
        model_dir = data_dir / "ESM_model" / "facebook" / "esm2_t12_35M_UR50D"
        result_path = mid_dir / "esm12_per_residual_embedding"
    elif plm_model_name == "esm2-33":
        model_dir = data_dir / "ESM_model" / "facebook" / "esm2_t33_650M_UR50D"
        result_path = mid_dir / "esm33_per_residual_embedding"
    else:
        raise ValueError(f"Unknown PLM model name: {plm_model_name}")
    return model_dir, result_path


def embedding_proteins_ESM2(
    plm_model_name: str,
    mid_dir: Path,
    data_dir: Path,
    device: torch.device,
    protein_fasta: Path | None = None,
) -> Path:
    """Embed every protein in `protein_fasta` (default: mid_dir/test_protein.fa)
    with ESM2 and write per-residue tensors to mid_dir/<plm-prefix>_per_residual_embedding/.

    Per-protein .pkl files that already exist are skipped (cache).
    Returns the directory containing the .pkl files.
    """
    print(f"Embedding the protein sequences using ESM2 {plm_model_name} ...")

    if protein_fasta is None:
        protein_fasta = mid_dir / "test_protein.fa"

    model_dir, result_path = _resolve_paths(plm_model_name, mid_dir, data_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"ESM2 weights not found at {model_dir}. "
            "Download the GOPhage data bundle (see README)."
        )
    result_path.mkdir(parents=True, exist_ok=True)

    # Cache: skip proteins whose .pkl already exists.
    all_seqs, all_ids = get_data(protein_fasta)
    todo_ids: list[str] = []
    todo_seqs: list[str] = []
    skipped = 0
    for seq, pid in zip(all_seqs, all_ids):
        out = result_path / f"{pid}_embedding.pkl"
        if out.exists():
            skipped += 1
        else:
            todo_ids.append(pid)
            todo_seqs.append(seq)

    if skipped:
        print(f"  cache: skipping {skipped}/{len(all_ids)} proteins already embedded")
    if not todo_ids:
        return result_path

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = EsmModel.from_pretrained(str(model_dir))

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!\n")
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    loader = DataLoader(
        VirDataset(todo_ids, todo_seqs),
        batch_size=1,
        num_workers=1,
    )

    with torch.no_grad():
        for texts, proteins_names in tqdm(
            loader, total=len(todo_ids), desc="ESM2 embedding"
        ):
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=1024,
                truncation=True,
            )
            name = proteins_names[0]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state.to("cpu")

            file_path_embedding = result_path / f"{name}_embedding.pkl"
            torch.save({"data": last_hidden_states}, str(file_path_embedding))

    return result_path
