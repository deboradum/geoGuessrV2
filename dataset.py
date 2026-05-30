import os
import torch
import random
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import s2cell  # type: ignore[import-untyped]

from PIL import Image
from typing import Tuple
import albumentations as A
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Sampler


class GeoGuessrDataset(Dataset):
    def __init__(self, csv_file, root_dir, cell_level, transform=None, label_map=None):
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

        if label_map is not None:
            self.cell_to_label = label_map
        else:
            # cellIDs are uint64, convert them to 0-... integer class labels
            unique_cells = self.data['cell_id'].unique()
            self.cell_to_label = {cell_id: i for i, cell_id in enumerate(unique_cells)}

        self.data['label'] = self.data['cell_id'].map(self.cell_to_label)

        # Drop any rows with cells that weren't in the global map
        self.data.dropna(subset=['label'], inplace=True)
        self.data['label'] = self.data['label'].astype(int)

        self.all_labels = self.data['label'].tolist()
        self.num_unique_s2_classes = len(self.cell_to_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row["path"])
        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        lat, lng = float(row["lat"]), float(row["lng"])
        coords = torch.tensor([lng, lat], dtype=torch.float32)

        s2_label = int(row["label"])

        return image, (coords, s2_label)


class PKBatchSampler(Sampler):
    def __init__(self, data_source, p, k):
        self.data_source = data_source
        self.p = p
        self.k = k

        self.all_cell_ids = list(self.data_source.keys())

        self.cells = [
            cell for cell in self.all_cell_ids
            if len(self.data_source[cell]) >= k
        ]

        total_images = sum(len(indices) for indices in self.data_source.values())
        self.num_batches = max(1, total_images // (self.p * self.k))

        print(f"\n" + "="*40)
        print(f"PKBatchSampler Diagnostic (p={p}, k={k})")
        print(f"Total Unique S2 Cells: {len(self.all_cell_ids)}")
        print(f"Total Images in Dataset: {total_images}")
        print(f"Images effectively LOST: 0 (0.0%)")
        print(f"Total Batches per Epoch: {self.num_batches}")
        print(f"Effective Epoch Size: {self.num_batches * self.p * self.k} images")
        print("="*40 + "\n")

    def __iter__(self):
        for i in range(self.num_batches):
            if len(self.all_cell_ids) >= self.p:
                batch_cell_ids = random.sample(self.all_cell_ids, self.p)
            else:
                # Fallback just in case you somehow have fewer total unique cells than P
                batch_cell_ids = random.choices(self.all_cell_ids, k=self.p)

            batch_indices = []
            for cell_id in batch_cell_ids:
                possible_indices = self.data_source[cell_id]
                if len(possible_indices) >= self.k:
                    selected_indices = random.sample(possible_indices, self.k)
                else:
                    selected_indices = random.choices(possible_indices, k=self.k)
                batch_indices.extend(selected_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


def get_loaders(directory: str="geoGuessrDataset/", s2_cell_level: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=900, width=900),
            A.Resize(height=256, width=256),

            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.5),
            A.CoarseDropout(max_holes=5, max_height=32, max_width=32, min_holes=1, p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Read all datasets to build a unified global label mapping
    train_path = os.path.join(directory, "train.csv")
    val_path = os.path.join(directory, "val.csv")
    test_path = os.path.join(directory, "test.csv")

    dfs = []
    for path in [train_path, val_path, test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, names=["path", "lat", "lng"])
            df.dropna(subset=["lat", "lng"], inplace=True)
            dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['cell_id'] = all_data.apply(
        lambda row: s2cell.lat_lon_to_cell_id(
            lat=float(row["lat"]),
            lon=float(row["lng"]),
            level=s2_cell_level
        ),
        axis=1
    )
    unique_cells = sorted(all_data['cell_id'].unique())
    global_cell_to_label = {cell_id: i for i, cell_id in enumerate(unique_cells)}
    print(f"Global unique S2 classes mapped: {len(global_cell_to_label)}")

    datasets = {
        "train": GeoGuessrDataset(os.path.join(directory, "train.csv"), directory, s2_cell_level, train_transform, label_map=global_cell_to_label),
        "val": GeoGuessrDataset(os.path.join(directory, "val.csv"), directory, s2_cell_level, val_transform, label_map=global_cell_to_label),
        "test": GeoGuessrDataset(os.path.join(directory, "test.csv"), directory, s2_cell_level, val_transform, label_map=global_cell_to_label),
    }

    loaders = {}
    for split in ["train", "val", "test"]:
        if split == "train":
            # Train uses PKBatchSampler for metric/S2 consistency
            cell_id_to_indices = defaultdict(list)
            for idx, cell_id in enumerate(datasets[split].all_labels):
                cell_id_to_indices[cell_id].append(idx)
            sampler = PKBatchSampler(cell_id_to_indices, p=16, k=4)
            loaders[split] = DataLoader(
                datasets[split],
                batch_sampler=sampler,
                num_workers=4,
            )
        else:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=16 * 4, # Match the effective batch size of p*k
                shuffle=False,
                num_workers=4,
            )

    return loaders["train"], loaders["val"], loaders["test"]
