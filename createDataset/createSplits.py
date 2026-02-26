import os
import re
import random
import pandas as pd
import argparse


def create_splits_googleMaps(dataset_dir, split=(80, 10, 10)):
    # Collect all image paths and their labels
    all_data = []
    for country in os.listdir(dataset_dir):
        if not os.path.isdir(f"{dataset_dir}/{country}"):
            continue

        location_data = pd.read_csv(
            f"{dataset_dir}/{country}/locations.csv",
            header=None,
            names=["panoidID", "lat", "lng"],
        ).drop_duplicates(subset="panoidID", keep="first")
        location_dict = location_data.set_index("panoidID")[["lat", "lng"]].to_dict(
            orient="index"
        )
        location_dict = {k: (v["lat"], v["lng"]) for k, v in location_dict.items()}

        for img_path in os.listdir(f"{dataset_dir}/{country}/"):
            match = re.match(r"^(.*?)_\d+\.jpg$", img_path)
            if not match:
                continue

            panoidID = match.group(1)
            full_img_path = f"{country}/{img_path}"

            lat, lng = location_dict[panoidID]

            all_data.append((full_img_path, lat, lng))

    # Create splits
    random.shuffle(all_data)
    total = len(all_data)
    train_size = int(total * split[0] / 100)
    val_size = int(total * split[1] / 100)
    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size :]

    # Write splits
    def write_split(data, filename):
        with open(os.path.join(dataset_dir, filename), "w") as f:
            for path, lat, lng in data:
                f.write(f"{path},{lat},{lng}\n")

    write_split(train_data, "train.csv")
    write_split(val_data, "val.csv")
    write_split(test_data, "test.csv")


def create_splits_geoGuessr(dataset_dir, split=(80, 10, 10)):
    # Collects all labels
    location_dict = {}

    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(
            os.path.join(dataset_dir, fname),
            header=None,
            names=["panoidID", "lat", "lng"],
        )

        # normalize panoidID
        df["panoidID"] = df["panoidID"].str.replace(r"\.jpg$", "", regex=True)

        # keep first occurrence per panoidID
        df = df.drop_duplicates(subset="panoidID", keep="first")

        for panoidID, lat, lng in df.itertuples(index=False):
            location_dict[panoidID] = (lat, lng)

    print(len(location_dict), "locations")
    all_data = []

    for img_path in os.listdir(dataset_dir):
        match = re.match(r"^(.*?)\.jpg$", img_path)
        if not match:
            continue

        panoidID = match.group(1)

        if panoidID not in location_dict:
            continue  # safe skip

        lat, lng = location_dict[panoidID]
        all_data.append((img_path, lat, lng))

    random.shuffle(all_data)
    total = len(all_data)

    train_size = int(total * split[0] / 100)
    val_size = int(total * split[1] / 100)

    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size :]

    def write_split(data, filename):
        with open(os.path.join(dataset_dir, filename), "w") as f:
            for path, lat, lng in data:
                f.write(f"{path},{lat},{lng}\n")

    write_split(train_data, "train.csv")
    write_split(val_data, "val.csv")
    write_split(test_data, "test.csv")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["geoGuessr", "googleMaps"],
        help="Select either geoguessr or googleMaps",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "geoGuessr":
        create_splits_geoGuessr("finalDataset")
    elif args.mode == "googleMaps":
        create_splits_googleMaps("mapsDataset")
