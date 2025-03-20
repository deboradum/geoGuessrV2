import os
import re
import folium
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from folium.plugins import HeatMap


def get_all_coords(dataset_dir):
    coords = []
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
            lat, lng = location_dict[panoidID]
            coords.append((lat, lng))

    return coords


def get_all_coords_geoguessrdataset(dataset_dir):
    coords = []
    location_dict = {}
    for file in os.listdir(dataset_dir):
        if not file.endswith(".csv"):
            continue

        location_data = pd.read_csv(
            f"{dataset_dir}/{file}",
            header=None,
            names=["panoidID", "lat", "lng"],
        ).drop_duplicates(subset="panoidID", keep="first")
        tmp_location_dict = location_data.set_index("panoidID")[["lat", "lng"]].to_dict(
            orient="index"
        )
        tmp_location_dict = {k: (v["lat"], v["lng"]) for k, v in tmp_location_dict.items()}
        location_dict.update(tmp_location_dict)

    for img_path in os.listdir(dataset_dir):
        match = re.match(r"^(.*?)\.png$", img_path)
        if not match:
            continue
        panoidID = match.group(1)
        lat, lng = location_dict[panoidID]
        coords.append((lat, lng))

    return coords


def get_country_data(dataset_dir):
    country_data = []
    for country in os.listdir(dataset_dir):
        if not os.path.isdir(f"{dataset_dir}/{country}"):
            continue

        num_images = 0
        for img_path in os.listdir(f"{dataset_dir}/{country}/"):
            match = re.match(r"^(.*?)_\d+\.jpg$", img_path)
            if not match:
                continue

            num_images += 1

        country_data.append((country, num_images))

    return country_data


def plot_heatmap(coordinates, dataset_dir):
    m = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(coordinates, radius=15).add_to(m)
    m.save(f"{dataset_dir}_heatmap.html")


def plot_country_distribution(country_data):
    # Sort data from least to most images
    country_data.sort(key=lambda x: x[1])
    countries, num_images = zip(*country_data)

    plt.figure(figsize=(12, 6))
    plt.bar(countries, num_images, color="skyblue")
    plt.xlabel("Country")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Country")
    plt.xticks(rotation=90)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("option", choices=["heatmap", "countryData"])
    parser.add_argument("dataset_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.option == "heatmap":
        coords = get_all_coords_geoguessrdataset(args.dataset_dir)
        if len(coords) == 0:
            coords = get_all_coords(args.dataset_dir)
        plot_heatmap(coords, args.dataset_dir)
    elif args.option == "countryData":
        country_data = get_country_data(args.dataset_dir)
        plot_country_distribution(country_data)
