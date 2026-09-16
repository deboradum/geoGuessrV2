import os
import torch
import argparse
import torch.nn.functional as F

from models import get_net
from dataset import get_loaders
from utils import load_config, save_error_flow_map, gcs_to_cartesian_tensor

EARTH_RADIUS = 6371000  # meters

def get_args():
    parser = argparse.ArgumentParser(description="Generate Error Flow Map from Pretrained Weights")
    parser.add_argument("--config", type=str, required=True, help="Path to conf.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained weights (.pth file)")
    parser.add_argument("--k", type=int, default=200, help="Top K worst errors to visualize")
    parser.add_argument("--output_dir", type=str, default="error_analysis", help="Output directory for the map")
    return parser.parse_args()

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
    train_loader, _, test_loader = get_loaders(
        directory=config.dataset_dir,
        s2_cell_level=config.s2_cell_level
    )

    # Initialize model
    print("Initializing model...")
    num_classes = train_loader.dataset.num_unique_s2_classes
    net = get_net(num_classes, config=config, device=device)

    # Load pretrained weights
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)

    # --- FIX FOR STATE DICT MISMATCH ---
    # Removes the unexpected 'model.' namespace that was likely injected during training
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle torch.compile prefix if present
        k = k.replace('_orig_mod.', '')

        # Handle the specific backbone.model mismatch
        if k.startswith('backbone.model.'):
            k = k.replace('backbone.model.', 'backbone.', 1)

        new_state_dict[k] = v

    # Load the corrected dictionary
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

    # Compute distances using the same Haversine logic from your loss_fn
    print("Calculating error distances...")
    pred_normalized = F.normalize(predictions, p=2, dim=1, eps=1e-8)

    true_lon_deg, true_lat_deg = targets[:, 0], targets[:, 1]
    true_x, true_y, true_z = gcs_to_cartesian_tensor(true_lat_deg, true_lon_deg)
    target_cartesian = torch.stack([true_x, true_y, true_z], dim=1)
    target_cartesian = F.normalize(target_cartesian, p=2, dim=1, eps=1e-8)

    chordal_dist = torch.norm(pred_normalized - target_cartesian, p=2, dim=1)
    clamped_ratio = torch.clamp(chordal_dist / 2.0, min=0.0, max=1.0 - 1e-5)
    c = 2.0 * torch.asin(clamped_ratio)
    distances = EARTH_RADIUS * c / 1000.0  # Convert to km

    # Generate and save the map
    print(f"Generating Error Flow Map for top {args.k} worst errors...")
    save_error_flow_map(
        predictions=predictions,
        targets=targets,
        distances=distances,
        output_dir=args.output_dir,
        top_k=args.k
    )

    print(f"Done! The map has been saved to {os.path.join(args.output_dir, 'global_error_flow.png')}")

if __name__ == "__main__":
    main()
