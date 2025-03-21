import os
import torch
import argparse

from PIL import Image
from torchvision import transforms

from trainJudge import get_net


def load_model(filepath, device):
    # Load the model and its state dict
    net = get_net().to(device)
    net.load_state_dict(torch.load(filepath, map_location=device))
    net.eval()
    return net


def predict_batch(net, image_paths, device, batch_size=16):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    # Create a DataLoader-like approach to process images in batches
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensors.append(image)

    batch = torch.cat(image_tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = net(batch)

    predictions = (
        torch.sigmoid(outputs).squeeze(1).cpu().numpy()
    )

    return predictions


def delete_decline_images(image_paths, net, device, batch_size=16, dry_run=True):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        predictions = predict_batch(net, batch_paths, device, batch_size)

        # Delete the images predicted as decline (prediction < 0.5)
        for idx, image_path in enumerate(batch_paths):
            if predictions[idx] < 0.25:  # Decline predicted
                print(f"Deleting {image_path} (Predicted: decline - {predictions[idx]})")
                if not dry_run:
                    os.remove(image_path)


def judge_maps(dataset_dir, dry_run=True):
    image_paths = []
    for country in os.listdir(dataset_dir):
        if not os.path.isdir(f"{dataset_dir}/{country}"):
            continue
        for img_path in os.listdir(f"{dataset_dir}/{country}/"):
            if not img_path.endswith(".jpg"):
                continue

            full_img_path = f"{dataset_dir}/{country}/{img_path}"
            image_paths.append(full_img_path)
    delete_decline_images(image_paths, net, device, batch_size=64, dry_run=dry_run)


def judge_geoguessr(dataset_dir, dry_run=True):
    image_paths = []
    for img_path in os.listdir(dataset_dir):
        if not img_path.endswith(".png"):
            continue

        full_img_path = f"{dataset_dir}/{img_path}"
        image_paths.append(full_img_path)
    delete_decline_images(image_paths, net, device, batch_size=64, dry_run=dry_run)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["geoGuessr", "googleMaps"],
        help="Select either geoguessr or googleMaps",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args = get_args()
    if args.dry:
        print("DRY RUN")

    if args.mode == "googleMaps":
        model_filepath = "mapsJudge.pth"
        net = load_model(model_filepath, device)
        judge_maps("mapsDataset", args.dry)
    elif args.mode == "geoGuessr":
        model_filepath = "geoGuessrJudge.pth"
        net = load_model(model_filepath, device)
        judge_geoguessr("geoGuessrDataset", args.dry)
