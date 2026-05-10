"""Per-residue ESM2 protein embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel


def get_data(fasta_file: Path) -> tuple[list[str], list[str]]:
    protein_sequence: list[str] = []
    all_id: list[str] = []
    for record in SeqIO.parse(str(fasta_file), "fasta"):
        protein_sequence.append(str(record.seq))
        all_id.append(str(record.id))
    return protein_sequence, all_id


class VirDataset(Dataset):
    def __init__(self, fasta_file: Path):
        self.texts, self.id = get_data(fasta_file)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.id[index]


def get_loader(fasta_file: Path, batch_size: int = 64, num_workers: int = 2):
    dataset = VirDataset(fasta_file)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)


def embedding_proteins_ESM2(
    plm_model_name: str, mid_dir: Path, data_dir: Path
) -> Path:
    """Run ESM2 per-residue embedding for every protein in mid_dir/test_protein.fa.

    Writes per-protein .pkl files into mid_dir/{esm12,esm33}_per_residual_embedding/
    and returns that directory.
    """
    print("Preparing the data for PhaGO model ...")
    print(f"Embedding the protein sequences using ESM2 {plm_model_name} ...")

    fasta_file = mid_dir / "test_protein.fa"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_loader(fasta_file, batch_size=1, num_workers=1)

    if plm_model_name == "esm2-12":
        model_dir = data_dir / "ESM_model" / "facebook" / "esm2_t12_35M_UR50D"
        result_path = mid_dir / "esm12_per_residual_embedding"
    elif plm_model_name == "esm2-33":
        model_dir = data_dir / "ESM_model" / "facebook" / "esm2_t33_650M_UR50D"
        result_path = mid_dir / "esm33_per_residual_embedding"
    else:
        raise ValueError(f"Unknown PLM model name: {plm_model_name}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"ESM2 weights not found at {model_dir}. "
            "Download the GOPhage data bundle (see README)."
        )

    result_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = EsmModel.from_pretrained(str(model_dir))

    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!\n")
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for texts, proteins_names in test_loader:
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=1024,
                truncation=True,
            )
            name = proteins_names[0]
            inputs = inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state.to("cpu")

            file_path_embedding = result_path / f"{name}_embedding.pkl"
            torch.save({"data": last_hidden_states}, str(file_path_embedding))

    return result_path
