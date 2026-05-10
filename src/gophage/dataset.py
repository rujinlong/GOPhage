import torch
from torch.utils.data import DataLoader, Dataset


class CombineDataset(Dataset):
    def __init__(self, data_file, ont, dict_ontology_length, plm_model):
        self.embedding_location = []
        self.protein_name_location = []

        with open(data_file) as fh:
            for lines in fh:
                line = lines.strip().split(",")
                self.embedding_location.append(line[0])
                self.protein_name_location.append(line[1])

        self.dict_ontology_length = dict_ontology_length
        self.length = dict_ontology_length[ont]
        self.plm_model = plm_model

    def __len__(self):
        return len(self.embedding_location)

    def __getitem__(self, index):
        embedding_location = self.embedding_location[index]
        protein_name_location = self.protein_name_location[index]

        data1 = torch.load(embedding_location, map_location=torch.device("cpu"))
        sequence_embedding = data1["data"]

        shapes = sequence_embedding.shape

        if self.plm_model == "esm2-12":
            emb_dim = 480
        elif self.plm_model == "esm2-33":
            emb_dim = 1280
        else:
            raise ValueError(f"Unknown PLM model name: {self.plm_model}")

        if shapes[0] < self.length:
            padding_number = self.length - shapes[0]
            padding_embedding = torch.zeros(
                (padding_number, 1024, emb_dim), dtype=torch.float32
            )
            all_embedding = torch.cat((sequence_embedding, padding_embedding), dim=0)
        else:
            all_embedding = sequence_embedding

        data3 = torch.load(protein_name_location, torch.device("cpu"))
        protein_name = data3["proteins_name"]

        while len(protein_name) < self.length:
            protein_name.append("none")

        all_embedding = torch.transpose(all_embedding, 1, 2)

        return all_embedding, protein_name


def get_combine_loader_cut_annot(
    data_file,
    dict_ontology_length,
    batch_size=64,
    num_workers=6,
    shuffle=False,
    pin_memory=False,
    drop=False,
    ont="CC",
    plm_model="esm2-12",
):
    dataset = CombineDataset(data_file, ont, dict_ontology_length, plm_model)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop,
    )
    return loader
