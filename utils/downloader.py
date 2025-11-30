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


# ============================================================
# utils/file_utils.py
# ============================================================

import os
import fitz  # PyMuPDF
from PIL import Image

def is_pdf(file_path: str) -> bool:
    return file_path.lower().endswith(".pdf")

def pdf_to_images(pdf_path: str, output_dir: str = "page_images") -> list:
    """
    Converts a PDF into individual page images.
    Returns list of image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    output_files = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        out_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(out_path)
        output_files.append(out_path)

    return output_files

def load_image(image_path: str):
    return Image.open(image_path)


# ============================================================
# utils/regex_utils.py
# ============================================================

import re

# Common patterns for matching:
# Quantity, Rate, Amount, Decimal numbers, etc.

FLOAT_PATTERN = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
INT_PATTERN = r"\d+"
AMOUNT_PATTERN = rf"({FLOAT_PATTERN})"
RATE_PATTERN = rf"({FLOAT_PATTERN})"
QTY_PATTERN = rf"({FLOAT_PATTERN})"

def extract_numbers(text: str):
    """
    Returns all numbers in a string as float.
    Useful for quantity, rate, amount parsing.
    """
    matches = re.findall(FLOAT_PATTERN, text)
    cleaned = [float(x.replace(",", "")) for x in matches]
    return cleaned

def extract_amount(text: str):
    """
    Extract first amount-like number from a text.
    """
    match = re.search(FLOAT_PATTERN, text)
    if match:
        return float(match.group().replace(",", ""))
    return None

def looks_like_header(text: str) -> bool:
    """
    Detects typical table headers.
    """
    text = text.lower()
    header_keywords = ["item", "description", "qty", "quantity", "rate", "price", "amount", "mrp"]
    score = sum(1 for k in header_keywords if k in text)
    return score >= 2

def is_total_row(text: str) -> bool:
    """
    Identifies rows like:
    - TOTAL
    - SUBTOTAL
    - GROSS TOTAL
    - NET PAYABLE
    """
    text = text.lower()
    total_keywords = [
        "total", "sub total", "subtotal", "net total",
        "net payable", "grand total", "gross total", "amount payable"
    ]
    return any(k in text for k in total_keywords)
