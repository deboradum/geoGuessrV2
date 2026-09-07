import os
import math
import yaml  # type: ignore[import-untyped]
import torch
import time
import hashlib
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from dataclasses import dataclass

@dataclass
class TrainConfig:
    net_name: str = "convnext-tiny"
    num_experts: int = 8
    router_k: int = 2
    dist_loss_weight: float = 1.0
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
    pred_lat_deg, pred_lon_deg = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)
    true_lon_deg, true_lat_deg = target[:, 0], target[:, 1]

    distances_km = distances.detach().cpu().numpy()

    batch_size = images.shape[0]

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = images * std + mean

    for i in range(batch_size):
        fig = plt.figure(figsize=(12, 5))

        t_lon, t_lat = true_lon_deg[i].item(), true_lat_deg[i].item()
        p_lon, p_lat = pred_lon_deg[i].item(), pred_lat_deg[i].item()

        # original image
        ax_img = fig.add_subplot(1, 2, 1)
        img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax_img.imshow(img_np)
        ax_img.axis('off')
        ax_img.set_title("Input Image")

        # World Map (Cleaner Layout)
        ax_map = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

        # Add colored base layers for better contrast
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_map.set_global()

        # Connect True and Pred with a geodetic (curved great-circle) line
        ax_map.plot([t_lon, p_lon], [t_lat, p_lat], color='black', linestyle='--', linewidth=1, transform=ccrs.Geodetic())

        # True coordinates (Blue)
        ax_map.plot(
            t_lon, t_lat,
            color='blue', marker='o', markersize=6,
            transform=ccrs.PlateCarree(),
            label=f'True: {t_lat:.4f}°, {t_lon:.4f}°'
        )

        # Predicted coordinates (Red)
        ax_map.plot(
            p_lon, p_lat,
            color='red', marker='x', markersize=6, markeredgewidth=2,
            transform=ccrs.PlateCarree(),
            label=f'Pred: {p_lat:.4f}°, {p_lon:.4f}°'
        )

        ax_map.legend(loc='lower left')

        sample_dist = distances_km[i]
        ax_map.set_title(f"Prediction vs True Location (Error: {sample_dist:,.2f} km)")

        coord_string = f"{t_lon:.5f}_{t_lat:.5f}".encode('utf-8')
        hash = hashlib.md5(coord_string).hexdigest()[:10]
        filename = f"{hash}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)


def save_expert_heatmaps(predictions, routing_probs, distances, output_dir):
    """
    Creates an aggregated plot comprising individual geographical heatmaps
    for each expert based on what they predicted during evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_x, pred_y, pred_z = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    pred_lat_deg, pred_lon_deg = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)

    pred_lat = pred_lat_deg.numpy()
    pred_lon = pred_lon_deg.numpy()
    distances_np = distances.numpy()

    if routing_probs.dtype in [torch.int32, torch.int64]:
        primary_expert = routing_probs.numpy()
    else:
        primary_expert = routing_probs.argmax(dim=-1).numpy()

    if primary_expert.ndim == 2:
        primary_expert = primary_expert[:, 0]

    num_experts = routing_probs.shape[-1] if routing_probs.ndim == 2 else (primary_expert.max() + 1)
    num_experts = max(int(primary_expert.max()) + 1, num_experts)

    cols = 2
    rows = math.ceil(num_experts / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), subplot_kw={'projection': ccrs.PlateCarree()})

    if num_experts == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_experts:
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.set_global()

            expert_mask = (primary_expert == i)
            count = expert_mask.sum()
            total_preds = len(primary_expert)
            fraction = count / total_preds if total_preds > 0 else 0

            if count > 0:
                # Use hexbin for efficient and clear visualization of point densities
                hb = ax.hexbin(
                    pred_lon[expert_mask],
                    pred_lat[expert_mask],
                    gridsize=60,
                    cmap='YlOrRd',
                    transform=ccrs.PlateCarree(),
                    mincnt=1
                )
                plt.colorbar(hb, ax=ax, orientation='horizontal', pad=0.05, aspect=40, label='Prediction Density')

                avg_dist = distances_np[expert_mask].mean()
                ax.set_title(f"Expert {i} Primary Predictions (Count: {count}, {fraction:.1%}, Avg Error: {avg_dist:,.0f} km)")
            else:
                ax.set_title(f"Expert {i} Primary Predictions (Count: {count}, {fraction:.1%})")
        else:
            ax.axis('off')

    plt.tight_layout()
    filepath = os.path.join(output_dir, "expert_heatmaps.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)

class ViTAttentionHook:
    """Elegantly hooks into the last attention layer of standard ViTs (timm/torchvision)."""
    def __init__(self, model):
        self.attention = None
        self.hook_handle = None

        # Dynamically find the last attention dropout layer (works for most ViT implementations)
        target_module = None
        for name, module in model.named_modules():
            if 'attn_drop' in name or ('self_attention' in name and 'dropout' in name):
                target_module = module

        if target_module is not None:
            self.hook_handle = target_module.register_forward_hook(self._hook)
        else:
            print("Warning: Could not auto-detect ViT attention layer for visualization.")

    def _hook(self, module, input, output):
        # The input to the attention dropout layer is the post-softmax attention matrix
        # Shape: (Batch, Heads, SeqLen, SeqLen)
        self.attention = input[0].detach()

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()

def save_attention_maps(images, model, output_dir, run_name):
    """Generates and saves ViT attention heatmaps using a forward hook."""
    os.makedirs(output_dir, exist_ok=True)

    hook = ViTAttentionHook(model)
    with torch.no_grad():
        _ = model(images)

    if hook.attention is None:
        hook.remove()
        return

    attn = hook.attention

    # Get attention from the [CLS] token (index 0) to all patch tokens (index 1:)
    cls_attn = attn[:, :, 0, 1:]  # (Batch, Heads, Patches)
    cls_attn = cls_attn.mean(dim=1)  # Average across heads -> (Batch, Patches)

    # Reshape and Upsample
    B, num_patches = cls_attn.shape
    grid_size = int(math.sqrt(num_patches))
    cls_attn = cls_attn.view(B, 1, grid_size, grid_size)

    # Normalize per image to [0, 1]
    cls_attn = cls_attn - cls_attn.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    cls_attn = cls_attn / cls_attn.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

    img_size = images.shape[-1]
    attn_upsampled = F.interpolate(cls_attn, size=(img_size, img_size), mode='bicubic', align_corners=False)

    # Denormalize images for plotting
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Plot and Save
    for i in range(B):
        img_np = images_denorm[i].permute(1, 2, 0).cpu().numpy()
        heatmap = attn_upsampled[i, 0].cpu().numpy()

        # Apply colormap to heatmap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        overlay = (img_np * 0.5) + (heatmap_colored * 0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Attention Heatmap")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        filepath = os.path.join(output_dir, f"attention_map_{i}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)

    hook.remove()

def save_error_flow_map(predictions, targets, distances, output_dir, top_k=200):
    """Draws great-circle arcs connecting the true and predicted locations for the worst errors."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract coordinates
    pred_x, pred_y, pred_z = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    pred_lat_deg, pred_lon_deg = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)

    true_lon_deg, true_lat_deg = targets[:, 0], targets[:, 1]

    # Identify the worst errors (Top K largest distances)
    distances_np = distances.numpy()
    if len(distances_np) > top_k:
        worst_indices = np.argsort(distances_np)[-top_k:]
    else:
        worst_indices = np.argsort(distances_np)

    # Setup the Map
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#E0E0E0')
    ax.add_feature(cfeature.OCEAN, facecolor='#B0D0E0')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_global()

    # Setup colormap based on distance severity
    norm = mcolors.Normalize(vmin=distances_np[worst_indices].min(), vmax=distances_np[worst_indices].max())
    cmap = cm.Reds

    # Plot the error arcs
    for idx in worst_indices:
        t_lon, t_lat = true_lon_deg[idx].item(), true_lat_deg[idx].item()
        p_lon, p_lat = pred_lon_deg[idx].item(), pred_lat_deg[idx].item()
        dist = distances_np[idx]

        color = cmap(norm(dist))

        # Geodetic transform creates the curved great-circle line
        ax.plot([t_lon, p_lon], [t_lat, p_lat], transform=ccrs.Geodetic(), color=color, linewidth=1.5, alpha=0.7)

        # Mark True (Green Dot) and Predicted (Black X)
        ax.plot(t_lon, t_lat, transform=ccrs.PlateCarree(), marker='o', color='green', markersize=4, alpha=0.8)
        ax.plot(p_lon, p_lat, transform=ccrs.PlateCarree(), marker='x', color='black', markersize=4, alpha=0.8)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Error Distance (km)')

    plt.title(f"Global Error Flow: Top {len(worst_indices)} Worst Confusions")

    filepath = os.path.join(output_dir, "global_error_flow.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    plt.close(fig)


# Global cache so we only load the shapefile once during training
_COUNTRY_RECORDS = None

def get_continent(lon, lat):
    """Maps a single (lon, lat) coordinate to a continent using Cartopy's offline Natural Earth data."""
    global _COUNTRY_RECORDS
    if _COUNTRY_RECORDS is None:
        shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        _COUNTRY_RECORDS = list(shpreader.Reader(shpfilename).records())

    pt = sgeom.Point(lon, lat)
    for record in _COUNTRY_RECORDS:
        if record.geometry.contains(pt):
            return record.attributes.get('CONTINENT', 'Ocean/Unknown')
    return "Ocean/Unknown"

def save_continent_confusion_matrix(predictions, targets, output_dir):
    """Generates a continent-level confusion matrix showing counts and percentages."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert Cartesian predictions to GCS
    pred_x, pred_y, pred_z = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    pred_lat, pred_lon = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)
    true_lon, true_lat = targets[:, 0], targets[:, 1]

    # Map coordinates to continents
    true_continents = [get_continent(lon, lat) for lon, lat in zip(true_lon.numpy(), true_lat.numpy())]
    pred_continents = [get_continent(lon, lat) for lon, lat in zip(pred_lon.numpy(), pred_lat.numpy())]

    # Get unique labels, dynamically excluding 'Ocean/Unknown' if it clutters the matrix
    labels = sorted(list(set(true_continents + pred_continents)))
    if "Ocean/Unknown" in labels and true_continents.count("Ocean/Unknown") < 10:
        labels.remove("Ocean/Unknown")

    # Compute Confusion Matrices
    cm = confusion_matrix(true_continents, pred_continents, labels=labels)
    # Normalize by row (true label) to get percentages for the heatmap color
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_pct = np.nan_to_num(cm_pct)

    # Plot using Seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Percentage of True Class Predicted'})

    plt.ylabel('True Continent', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Continent', fontsize=12, fontweight='bold')
    plt.title('Continent-Level Confusion Matrix\n(Colors show row percentage, numbers show exact count)', pad=15)
    plt.xticks(rotation=45, ha='right')

    filepath = os.path.join(output_dir, "continent_confusion_matrix.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
