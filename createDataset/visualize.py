import os
import re
import folium
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


def plot_heatmap(coordinates):
    m = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(coordinates, radius=15).add_to(m)
    m.save("heatmap.html")


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


if __name__ == "__main__":
    # coords = get_all_coords("dataset/")
    # plot_heatmap(coords)

    country_data = get_country_data("dataset/")
    plot_country_distribution(country_data)
