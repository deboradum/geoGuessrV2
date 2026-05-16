import os
import math
import yaml  # type: ignore[import-untyped]
import torch
import time
import hashlib
import numpy as np
import cartopy.crs as ccrs
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

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
    s2_head_lr = base_lr * 10.0

    backbone_params = []
    s2_head_params = []
    other_params = []

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        # If the parameter belongs to the backbone, route it to the discounted list
        if name.startswith("backbone."):
            backbone_params.append(param)
        elif name.startswith("s2_feature_layer.") or name.startswith("s2_projection_layer."):
            s2_head_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if s2_head_params:
        param_groups.append({"params": s2_head_params, "lr": s2_head_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    optimizer : torch.optim.Optimizer
    if config.optimizer == "adam":
        print(f"Using {config.optimizer} optimizer")
        optimizer = torch.optim.Adam(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, config.beta_2))
    elif config.optimizer == "adamW":
        print(f"Using {config.optimizer} optimizer")
        optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        print(f"Using {config.optimizer} optimizer")
        optimizer = torch.optim.SGD(param_groups, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optimizer == "muon":
        print(f"Using {config.optimizer} optimizer")
        optimizer = torch.optim.Muon(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
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

def save_predictions(images, pred, target, distances, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
    pred_lon_deg, pred_lat_deg = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)
    true_lon_deg, true_lat_deg = target[:, 0], target[:, 1]

    distances_km = distances.detach().cpu().numpy()

    batch_size = images.shape[0]

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = images * std + mean

    for i in range(batch_size):
        fig = plt.figure(figsize=(12, 5))

        # original image
        ax_img = fig.add_subplot(1, 2, 1)
        img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax_img.imshow(img_np)
        ax_img.axis('off')
        ax_img.set_title("Input Image")

        # World Map
        ax_map = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.COASTLINE)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.set_global()

        # True coordinates (Blue)
        ax_map.plot(
            true_lon_deg[i].item(), true_lat_deg[i].item(),
            color='blue', marker='o', markersize=8,
            transform=ccrs.PlateCarree(), label='True'
        )

        # Predicted coordinates (Red)
        ax_map.plot(
            pred_lon_deg[i].item(), pred_lat_deg[i].item(),
            color='red', marker='x', markersize=8, markeredgewidth=2,
            transform=ccrs.PlateCarree(), label='Prediction'
        )

        ax_map.legend(loc='lower left')

        sample_dist = distances_km[i]
        ax_map.set_title(f"Prediction vs True Location (Error: {sample_dist:,.2f} km)")

        t_lon, t_lat = true_lon_deg[i].item(), true_lat_deg[i].item()
        coord_string = f"{t_lon:.5f}_{t_lat:.5f}".encode('utf-8')
        deterministic_hash = hashlib.md5(coord_string).hexdigest()[:10]
        filename = f"sample_{deterministic_hash}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
