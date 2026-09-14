import os
import re
import argparse
import folium
import pandas as pd
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep


def load_country_geometries():
    """Load Natural Earth country shapes using Cartopy."""
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    records = list(reader.records())
    prepared_records = [(prep(rec.geometry), rec) for rec in records]
    return records, prepared_records


def reverse_geocode(lat, lng, records, prepared_records):
    """Maps (lat, lng) to (country, continent) using polygon containment with a distance fallback."""
    point = Point(lng, lat)  # Shapely Point is (x, y) = (lng, lat)

    # 1. Direct containment check
    for prep_geom, rec in prepared_records:
        if prep_geom.contains(point):
            country = rec.attributes.get("ADMIN") or rec.attributes.get("NAME") or "Unknown"
            continent = rec.attributes.get("CONTINENT") or "Unknown"
            return country, continent

    # 2. Fallback for coastal/offshore points (find nearest country geometry)
    nearest_rec = min(records, key=lambda rec: rec.geometry.distance(point))
    country = nearest_rec.attributes.get("ADMIN") or nearest_rec.attributes.get("NAME") or "Unknown"
    continent = nearest_rec.attributes.get("CONTINENT") or "Unknown"
    return country, continent


def get_existing_image_ids(dataset_dir):
    """Scan dataset directory for existing image IDs."""
    existing_ids = set()
    for img_path in os.listdir(dataset_dir):
        match = re.match(r"^(.*?)(?:_\d+)?\.(?:jpg|jpeg|png)$", img_path, re.IGNORECASE)
        if match:
            existing_ids.add(match.group(1))
    return existing_ids


def load_split_coords(dataset_dir):
    """
    Loads coordinates organized by split (train, val, test).
    Returns a dict: { split_name: [(lat, lng), ...] }
    """
    existing_ids = get_existing_image_ids(dataset_dir)
    splits = ["train", "val", "test"]
    split_coords = {}

    for split in splits:
        csv_path = os.path.join(dataset_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, header=None, names=["panoidID", "lat", "lng"])
            # Remove image extensions if present in the CSV
            df["clean_id"] = df["panoidID"].astype(str).str.replace(r"\.[a-zA-Z0-9]+$", "", regex=True)

            # Keep only images that exist in dataset_dir
            df = df[df["clean_id"].isin(existing_ids)].drop_duplicates(subset="clean_id")

            coords = list(zip(df["lat"].astype(float), df["lng"].astype(float)))
            if coords:
                split_coords[split] = coords
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    # Fallback to all.csv if individual split CSVs are missing
    if not split_coords:
        all_csv = os.path.join(dataset_dir, "all.csv")
        if os.path.exists(all_csv):
            df = pd.read_csv(all_csv, header=None, names=["panoidID", "lat", "lng"])
            df["clean_id"] = df["panoidID"].astype(str).str.replace(r"\.[a-zA-Z0-9]+$", "", regex=True)
            df = df[df["clean_id"].isin(existing_ids)].drop_duplicates(subset="clean_id")
            split_coords["all"] = list(zip(df["lat"].astype(float), df["lng"].astype(float)))

    return split_coords


def plot_heatmaps(split_coords, dataset_dir):
    """Generates separate HTML heatmaps for train, val, and test splits."""
    if not split_coords:
        print("No valid coordinates found for heatmap generation.")
        return

    for split_name, coords in split_coords.items():
        m = folium.Map(location=[20, 0], zoom_start=2)
        HeatMap(coords, radius=15).add_to(m)

        out_file = os.path.join(dataset_dir, f"{split_name}_heatmap.html")
        m.save(out_file)
        print(f"Saved heatmap for '{split_name}' split ({len(coords)} points) -> {out_file}")


def process_country_data(split_coords):
    """Performs reverse geocoding to aggregate stats per split."""
    print("Loading country shapefiles via Cartopy...")
    records, prepared_records = load_country_geometries()

    split_stats = {}
    for split_name, coords in split_coords.items():
        print(f"Geocoding '{split_name}' set ({len(coords)} points)...")
        country_counts = {}
        continent_counts = {}

        for lat, lng in coords:
            country, continent = reverse_geocode(lat, lng, records, prepared_records)
            country_counts[country] = country_counts.get(country, 0) + 1
            continent_counts[continent] = continent_counts.get(continent, 0) + 1

        split_stats[split_name] = {
            "country": country_counts,
            "continent": continent_counts,
        }

    return split_stats


def print_stats_table(split_stats):
    """Prints continent and top country distribution tables per split with percentages."""
    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION SUMMARY")
    print("=" * 60)

    for split_name, stats in split_stats.items():
        total_count = sum(stats["continent"].values())
        print(f"\n--- SPLIT: {split_name.upper()} (Total Images: {total_count}) ---")

        if total_count == 0:
            print("No data available for this split.")
            continue

        print("\n[Continents]")
        cont_df = pd.DataFrame(list(stats["continent"].items()), columns=["Continent", "Count"])
        cont_df["Percentage"] = (cont_df["Count"] / total_count * 100).round(2).astype(str) + "%"
        cont_df = cont_df.sort_values(by="Count", ascending=False)
        print(cont_df.to_string(index=False))

        print("\n[Top 10 Countries]")
        c_df = pd.DataFrame(list(stats["country"].items()), columns=["Country", "Count"])
        c_df["Percentage"] = (c_df["Count"] / total_count * 100).round(2).astype(str) + "%"
        c_df = c_df.sort_values(by="Count", ascending=False)
        print(c_df.head(10).to_string(index=False))


def plot_distributions(split_stats):
    """Plots comparative grouped bar charts for Continents and Top 15 Countries."""
    # Continent Distribution Plot
    all_continents = sorted(list({c for s in split_stats.values() for c in s["continent"].keys()}))
    df_cont = pd.DataFrame(index=all_continents)
    for split_name, stats in split_stats.items():
        df_cont[split_name] = df_cont.index.map(lambda c: stats["continent"].get(c, 0))

    df_cont.plot(kind="bar", figsize=(10, 5))
    plt.title("Continent Distribution Across Splits")
    plt.xlabel("Continent")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Top 15 Countries Distribution Plot
    total_country_counts = {}
    for stats in split_stats.values():
        for country, count in stats["country"].items():
            total_country_counts[country] = total_country_counts.get(country, 0) + count

    top_countries = [c for c, _ in sorted(total_country_counts.items(), key=lambda x: x[1], reverse=True)[:15]]

    df_country = pd.DataFrame(index=top_countries)
    for split_name, stats in split_stats.items():
        df_country[split_name] = df_country.index.map(lambda c: stats["country"].get(c, 0))

    df_country.plot(kind="bar", figsize=(12, 6))
    plt.title("Top 15 Countries Distribution Across Splits")
    plt.xlabel("Country")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("option", choices=["heatmap", "countryData"])
    parser.add_argument("dataset_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    split_coords = load_split_coords(args.dataset_dir)

    if args.option == "heatmap":
        plot_heatmaps(split_coords, args.dataset_dir)
    elif args.option == "countryData":
        if not split_coords:
            print("No split CSVs found or no matching images.")
        else:
            split_stats = process_country_data(split_coords)
            print_stats_table(split_stats)
            plot_distributions(split_stats)
