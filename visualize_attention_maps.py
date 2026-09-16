import os
import torch
import argparse

from models import get_net
from dataset import get_loaders
from utils import load_config, save_attention_maps

def get_args():
    parser = argparse.ArgumentParser(description="Generate ViT Attention Maps from Pretrained Weights")
    parser.add_argument("--config", type=str, required=True, help="Path to conf.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained weights (.pth file)")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to visualize")
    parser.add_argument("--output_dir", type=str, default="attention_analysis", help="Output directory for the maps")
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

    # Apply the same fix for the state_dict keys as before
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

    print(f"Generating attention maps for {args.num_batches} batch(es)...")

    for batch_idx, (X, (y_coords, _)) in enumerate(test_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"Processing batch {batch_idx + 1}/{args.num_batches} (Batch Size: {X.shape[0]})...")
        X = X.to(device)
        y_coords = y_coords.to(device)

        batch_output_dir = os.path.join(args.output_dir, f"batch_{batch_idx}")

        save_attention_maps(
            images=X,
            targets=y_coords,
            model=net,
            output_dir=batch_output_dir,
            run_name=config.run_name
        )

    print(f"\nDone! Attention maps have been saved to the '{args.output_dir}' directory.")

if __name__ == "__main__":
    main()
