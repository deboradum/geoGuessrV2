# Downloads and extracts the GeoGuessr dataset so it is ready to use.
# https://huggingface.co/datasets/deboradum/GeoCoordinates
import os
import tarfile

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    download_dir = "dataset"

    # 1. Download dataset
    snapshot_download(
        repo_id="deboradum/GeoCoordinates",
        repo_type="dataset",
        local_dir=download_dir,
        local_dir_use_symlinks=False,
    )

    # 2. Untar files
    for fname in [f for f in os.listdir(download_dir) if f.endswith(".tar.gz")]:
            tar_path = os.path.join(download_dir, fname)
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=download_dir)
