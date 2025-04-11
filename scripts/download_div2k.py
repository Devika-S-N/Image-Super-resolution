import os
import requests
from tqdm import tqdm

DIV2K_URLS = {
    "HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "LR_X4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
}

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path, total=total, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for name, url in DIV2K_URLS.items():
        zip_path = os.path.join("data", f"{name}.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {name}...")
            download_file(url, zip_path)
        else:
            print(f"{zip_path} already exists. Skipping.")
