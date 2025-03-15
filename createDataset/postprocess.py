import os
import re
import random
import pandas as pd

# Filter undesired images with a simple resnet trained on desirable and
# undesirable images. undesired images are thos of the sky, containing the map,
# containing black parts, etc.


def create_splits(dataset_dir, split=(80, 10, 10)):
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
        location_dict = location_data.set_index("panoidID")[["lat", "lng"]].to_dict(orient="index")
        location_dict = {k: (v["lat"], v["lng"]) for k, v in location_dict.items()}

        for img_path in os.listdir(f"{dataset_dir}/{country}/"):
            match = re.match(r"^(.*?)_\d+\.jpg$", img_path)
            if not match:
                continue

            panoidID = match.group(1)
            full_img_path = f"{dataset_dir}/{country}/{img_path}"

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


if __name__ == "__main__":
    create_splits("dataset")