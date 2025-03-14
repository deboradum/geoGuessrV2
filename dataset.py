import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


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
