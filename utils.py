import math
import yaml
import torch

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    net_name: str = "convnext-tiny"
    freeze_weights: bool = False
    dataset_dir: str = "dataset/"
    log_interval: int = 100
    seed: int = 123
    epochs: int = 2
    optimizer: str = "adamW"
    beta_2: float = 0.95
    learning_rate: float = 0.0001
    weight_decay: float = 0.05
    batch_size: int = 64
    gradient_clipping_norm: float = 1.0
    early_stop: int = 3
    run_name: str = "You forgot to change the run name"


def load_config(yaml_path: str) -> TrainConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def get_optimizer(config: TrainConfig, net: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer : torch.optim.Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, config.beta_2))
    elif config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise Exception("Invalid optimizer")

    return optimizer

def gcs_to_cartesian(lat, lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)

    return x, y, z

def gcs_to_cartesian_tensor(lat, lon):
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    return x, y, z

def cartesian_to_gcs_tensor(x, y, z):
    lat_rad = torch.arcsin(z)
    lon_rad = torch.atan2(y, x)

    lat_deg = torch.rad2deg(lat_rad)
    lon_deg = torch.rad2deg(lon_rad)

    return lat_deg, lon_deg
