import math
import yaml  # type: ignore[import-untyped]
import torch

from dataclasses import dataclass

@dataclass
class TrainConfig:
    net_name: str = "convnext-tiny"
    num_experts: int = 8
    router_k: int = 2
    s2_loss_weight: float = 0.5
    load_balance_loss_weight: float = 0.01
    embedding_dim: int = 512
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
    s2_cell_level: int = 10
    pretrained_path: str = ""


def load_config(yaml_path: str) -> TrainConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def get_optimizer(config: TrainConfig, net: torch.nn.Module) -> torch.optim.Optimizer:
    base_lr = config.learning_rate
    backbone_lr = base_lr / 10.0

    backbone_params = []
    other_params = []

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        # If the parameter belongs to the backbone, route it to the discounted list
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})


    optimizer : torch.optim.Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, config.beta_2))
    elif config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
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
