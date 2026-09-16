import os
import math
import argparse
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_all_image_paths(directory: str, name: str):
    found_paths = []
    for root, _, files in os.walk(directory):
        if name in files:
            full_path = os.path.join(root, name)
            found_paths.append(full_path)

    return found_paths

def visualize(paths: List[str], save: bool, base_name: str = "visualizations_grid"):
    if not paths:
        print("No images found to visualize.")
        return

    initial_path = None
    test_path = None
    other_paths = []

    for p in paths:
        subdir = os.path.basename(os.path.dirname(p))
        if subdir == "initial":
            initial_path = p
        elif subdir == "test":
            test_path = p
        else:
            other_paths.append((subdir, p))

    def sort_key(item):
        try:
            return int(item[0])
        except ValueError:
            return item[0]

    other_paths.sort(key=sort_key)

    n_images = len(paths)
    cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    total_cells = rows * cols

    grid_paths = [None] * total_cells
    current_idx = 0
    if initial_path:
        grid_paths[0] = initial_path
        current_idx += 1
    for _, p in other_paths:
        grid_paths[current_idx] = p
        current_idx += 1
    if test_path:
        grid_paths[-1] = test_path

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if total_cells == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        path = grid_paths[i]

        if path is None:
            ax.axis('off')
            continue

        try:
            img = mpimg.imread(path)
            ax.imshow(img)
            subdir = os.path.basename(os.path.dirname(path))

            ax.set_title(f"Epoch {subdir}", y=-0.15)
        except Exception as e:
            print(f"Error loading {path}: {e}")

        ax.axis('off')

    plt.tight_layout()

    if save:
        save_path = f"{base_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Path to the visualizations directory", required=True)
    parser.add_argument("--save", action="store_true", help="Save visualization image")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", type=str, help="Image name to visualize")
    group.add_argument("--all", action="store_true", help="Iterate through all images in the initial directory")

    args = parser.parse_args()

    if args.all:
        initial_dir = os.path.join(args.dir, "initial")
        if not os.path.exists(initial_dir):
            print(f"Error: 'initial' directory not found in {args.dir}")
            exit(1)

        for filename in os.listdir(initial_dir):
            filepath = os.path.join(initial_dir, filename)
            if os.path.isfile(filepath):
                print(f"Processing: {filename}")
                paths = find_all_image_paths(args.dir, filename)
                name_without_ext = os.path.splitext(filename)[0]
                visualize(paths, args.save, base_name=f"visualizations_grid_{name_without_ext}")
    else:
        paths = find_all_image_paths(args.dir, args.name)
        visualize(paths, args.save)
