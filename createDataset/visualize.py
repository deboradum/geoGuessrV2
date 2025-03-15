import os
import re
import folium
import pandas as pd
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


def visualize(coordinates):
    m = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(coordinates, radius=15).add_to(m)
    m.save("heatmap.html")

if __name__ == "__main__":
    coords = get_all_coords("dataset/")
    visualize(coords)
