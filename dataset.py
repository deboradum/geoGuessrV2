import os
import torch

import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        lat, lng = float(row["lat"]), float(row["lng"])
        target = torch.tensor([lat, lng], dtype=torch.float32)

        return image, target



def get_loaders_googleMaps(
    batch_size, model_name, directory="createDataset/mapsDataset/"
):
    transform = transforms.Compose(
        [
            transforms.Resize((240, 240))
            if "efficientnet" in model_name
            else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    datasets = {
        "train": GeoGuessrDataset(
            os.path.join(directory, "train.csv"), directory, transform
        ),
        "val": GeoGuessrDataset(
            os.path.join(directory, "val.csv"), directory, transform
        ),
        "test": GeoGuessrDataset(
            os.path.join(directory, "test.csv"), directory, transform
        ),
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


def get_loaders_geoGuessr(
    batch_size, model_name, directory="createDataset/geoGuessrDataset/"
):
    transform = transforms.Compose(
        [
            transforms.RandomCrop((448, 448)),
            transforms.Resize((240, 240))
            if "efficientnet" in model_name
            else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    datasets = {
        "train": GeoGuessrDataset(
            os.path.join(directory, "train.csv"), directory, transform
        ),
        "val": GeoGuessrDataset(
            os.path.join(directory, "val.csv"), directory, transform
        ),
        "test": GeoGuessrDataset(
            os.path.join(directory, "test.csv"), directory, transform
        ),
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
