import yaml
import torch

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    net_name: str = "convnext-tiny"
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
    num_classes: int = 1000
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


def latlng_to_class(lat: float, lng: float, num_classes: int) -> Tuple[int, int]:
    # lat in [-90, 90] and lng in [-180, 180]
    lat_class = int(((lat + 90) / 180) * num_classes)
    lng_class = int(((lng + 180) / 360) * num_classes)
    lat_class = min(max(lat_class, 0), num_classes - 1)
    lng_class = min(max(lng_class, 0), num_classes - 1)
    return lat_class, lng_class
