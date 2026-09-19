"""
Auto-download textile weaving datasets from Mendeley Data on first run.
Called by the Streamlit app if data/ folder is empty.
"""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    "weaving_rejection_dataset.csv": "https://data.mendeley.com/public-files/datasets/6mwgj7tms3/files/34af6ecb-b14f-4986-a561-312a9b9b6244/file_downloaded",
    "weaving_dataset_full.csv": "https://data.mendeley.com/public-files/datasets/6mwgj7tms3/files/ada98cab-e091-4d99-a47e-85c409c2a75e/file_downloaded",
}

def ensure_data_exists():
    """Download datasets if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(filepath, "wb") as f:
                        f.write(resp.read())
                print(f"  ✅ {filename} downloaded ({os.path.getsize(filepath)} bytes)")
            except Exception as e:
                print(f"  ⚠️ Failed to download {filename}: {e}")

if __name__ == "__main__":
    ensure_data_exists()
