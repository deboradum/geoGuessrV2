import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep

from models import get_net
from dataset import get_loaders
from utils import load_config, gcs_to_cartesian_tensor

EARTH_RADIUS = 6371000  # meters

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Geographic Coordinate Model per Continent")
    parser.add_argument("--config", type=str, required=True, help="Path to conf.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained weights (.pth file)")
    return parser.parse_args()


def get_continent_mapper():
    """
    Creates a highly optimized function to map (lon, lat) to a continent.
    Uses shapely.prepared.prep to drastically speed up point-in-polygon checks.
    """
    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    records = list(shpreader.Reader(shpfilename).records())

    # Prep geometries for faster contains() checks
    prepared_geoms = [
        (prep(rec.geometry), rec.attributes.get('CONTINENT', 'Ocean/Unknown'))
        for rec in records
    ]

    def get_continent(lon, lat):
        pt = Point(lon, lat)
        for prep_geom, continent in prepared_geoms:
            if prep_geom.contains(pt):
                return continent
        return "Ocean/Unknown"

    return get_continent


def main():
    args = get_args()

    # Setup device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)

    # Load datasets
    print(f"Loading datasets from {config.dataset_dir}...")
    _, _, test_loader = get_loaders(
        directory=config.dataset_dir,
        s2_cell_level=config.s2_cell_level
    )

    # Initialize model
    print("Initializing model...")
    num_classes = test_loader.dataset.num_unique_s2_classes
    net = get_net(num_classes, config=config, device=device)

    # Load pretrained weights
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        if k.startswith('backbone.model.'):
            k = k.replace('backbone.model.', 'backbone.', 1)
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=False)
    net.to(device)
    net.eval()

    all_preds = []
    all_targets = []

    # Run inference on the test set
    print("Evaluating model...")
    with torch.no_grad():
        for i, (X, (y_coords, _)) in enumerate(test_loader):
            X = X.to(device)
            out, _, _ = net(X)

            all_preds.append(out.cpu())
            all_targets.append(y_coords.cpu())

    # Concatenate results
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute distances using Haversine logic
    print("Calculating distances and scores...")
    pred_normalized = F.normalize(predictions, p=2, dim=1, eps=1e-8)

    true_lon_deg, true_lat_deg = targets[:, 0], targets[:, 1]
    true_x, true_y, true_z = gcs_to_cartesian_tensor(true_lat_deg, true_lon_deg)
    target_cartesian = torch.stack([true_x, true_y, true_z], dim=1)
    target_cartesian = F.normalize(target_cartesian, p=2, dim=1, eps=1e-8)

    chordal_dist = torch.norm(pred_normalized - target_cartesian, p=2, dim=1)
    clamped_ratio = torch.clamp(chordal_dist / 2.0, min=0.0, max=1.0 - 1e-5)
    c = 2.0 * torch.asin(clamped_ratio)

    # Distance in km
    distances = EARTH_RADIUS * c / 1000.0
    # Geoguessr score formulation
    scores = 5000.0 * torch.exp(-distances / 2000.0)

    # --- Continent Mapping & Aggregation ---
    print("Mapping coordinates to continents...")
    get_continent = get_continent_mapper()

    continent_to_indices = defaultdict(list)
    for i in range(len(targets)):
        lon, lat = true_lon_deg[i].item(), true_lat_deg[i].item()
        continent = get_continent(lon, lat)
        continent_to_indices[continent].append(i)

    distances_np = distances.numpy()
    scores_np = scores.numpy()

    # --- Print Formatting ---
    print("\n" + "=" * 125)
    print(f"{'Continent':<20} | {'Samples':<7} | {'Dist Avg':<9} | {'Dist p20':<9} | {'Dist p50':<9} | {'Dist p80':<9} || {'Score Avg':<9} | {'Score p20':<9} | {'Score p50':<9} | {'Score p80':<9}")
    print("-" * 125)

    def print_stats(name, idxs):
        d = distances_np[idxs]
        s = scores_np[idxs]
        print(
            f"{name:<20} | {len(idxs):<7} "
            f"| {d.mean():<9.1f} | {np.percentile(d, 20):<9.1f} | {np.median(d):<9.1f} | {np.percentile(d, 80):<9.1f} || "
            f"{s.mean():<9.0f} | {np.percentile(s, 20):<9.0f} | {np.median(s):<9.0f} | {np.percentile(s, 80):<9.0f}"
        )

    # Sort continents alphabetically
    for continent in sorted(continent_to_indices.keys()):
        print_stats(continent, continent_to_indices[continent])

    print("-" * 125)
    print_stats("Overall", np.arange(len(targets)))
    print("=" * 125)


if __name__ == "__main__":
    main()
