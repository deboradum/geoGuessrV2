import os
import torch
import random
import pandas as pd
import s2cell

from PIL import Image
from typing import Tuple
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler


class GeoGuessrDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=["path", "lat", "lng"])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        lat, lng = float(row["lat"]), float(row["lng"])
        target = torch.tensor([lng, lat], dtype=torch.float32)

        return image, target


class GeoGuessrEmbeddingDataset(Dataset):
    def __init__(self, csv_file, root_dir, cell_level, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=["path", "lat", "lng"])
        self.data.dropna(subset=["lat", "lng"], inplace=True)
        self.root_dir = root_dir
        self.transform = transform
        self.cell_level = cell_level

        self.data['cell_id'] = self.data.apply(
            lambda row: s2cell.lat_lon_to_cell_id(
                lat=float(row["lat"]),
                lon=float(row["lng"]),
                level=cell_level
            ),
            axis=1
        )
        self.data.dropna(subset=['cell_id'], inplace=True)
        self.data['cell_id'] = self.data['cell_id'].astype('uint64')
        # uint64 cellIDs to class albels to prevent overflow crash
        unique_cells = self.data['cell_id'].unique()
        self.cell_to_label = {cell_id: i for i, cell_id in enumerate(unique_cells)}
        self.data['label'] = self.data['cell_id'].map(self.cell_to_label)
        self.all_labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        cell_id = int(row["label"])

        return image, cell_id

def get_loaders_geoGuessr(batch_size: int, directory: str="geoGuessrDataset/") -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.RandomCrop((448, 448)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    datasets = {
        "train": GeoGuessrDataset(os.path.join(directory, "train.csv"), directory, transform),
        "val": GeoGuessrDataset(os.path.join(directory, "val.csv"), directory, transform),
        "test": GeoGuessrDataset(os.path.join(directory, "test.csv"), directory, transform),
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=4,
        )
        for split in ["train", "val", "test"]
    }

    return loaders["train"], loaders["val"], loaders["test"]


class PKBatchSampler(Sampler):
    def __init__(self, data_source, p, k):
        self.data_source = data_source
        self.p = p
        self.k = k

        self.cells = [
            cell for cell in self.data_source.keys()
            if len(self.data_source[cell]) >= k
        ]

        self.num_batches = len(self.cells) // self.p

    def __iter__(self):
        random.shuffle(self.cells)

        for i in range(self.num_batches):
            batch_cell_ids = self.cells[i * self.p : (i + 1) * self.p]

            batch_indices = []
            for cell_id in batch_cell_ids:
                possible_indices = self.data_source[cell_id]
                selected_indices = random.sample(possible_indices, self.k)
                batch_indices.extend(selected_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


def get_loaders_geoGuessrEmbedding(directory: str="geoGuessrDataset/", s2_cell_level: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.RandomCrop((448, 448)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    datasets = {
        "train": GeoGuessrEmbeddingDataset(os.path.join(directory, "train.csv"), directory, s2_cell_level, transform),
        "val": GeoGuessrEmbeddingDataset(os.path.join(directory, "val.csv"), directory, s2_cell_level, transform),
        "test": GeoGuessrEmbeddingDataset(os.path.join(directory, "test.csv"), directory, s2_cell_level, transform),
    }

    loaders = {}
    for split in ["train", "val", "test"]:
        cell_id_to_indices = defaultdict(list)
        for idx, cell_id in enumerate(datasets[split].all_labels):
            cell_id_to_indices[cell_id].append(idx)
        sampler = PKBatchSampler(cell_id_to_indices, p=16, k=4)
        loaders[split] = DataLoader(
            datasets[split],
            batch_sampler=sampler,
            num_workers=4,
        )

    return loaders["train"], loaders["val"], loaders["test"]
