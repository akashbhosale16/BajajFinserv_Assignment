# ============================================================
# utils/downloader.py
# ============================================================

import os
import requests
from pathlib import Path

def download_file(url: str, output_dir: str = "downloads") -> str:
    """
    Downloads a file from a URL and returns the local file path.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0]
    local_path = os.path.join(output_dir, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return local_path

