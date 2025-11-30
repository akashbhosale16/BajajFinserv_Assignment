# utils/file_utils.py

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
