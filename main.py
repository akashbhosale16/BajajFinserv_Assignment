"""
FastAPI app: /extract-bill-data
Uses Tesseract OCR + OpenCV to extract line-item rows from invoice images/PDFs.

NOTES:
- Install Tesseract on host machine (system package).
  Ubuntu: sudo apt-get install tesseract-ocr
  macOS (homebrew): brew install tesseract
- This is a pragmatic heuristic approach (no LLMs / no training).
- Works best when invoices contain tabular rows with numeric qty/rate/amount on right.

Author: ChatGPT (example starter)
"""

import io
import os
import re
import tempfile
import math
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse
import uvicorn

from PIL import Image
import numpy as np
import cv2
import pytesseract
from pytesseract import Output

# PDF support
from pdf2image import convert_from_path, convert_from_bytes
import requests

app = FastAPI(title="Bill Extraction API")


# --------------- Helper functions ---------------

def download_document_to_images(url: str) -> List[Image.Image]:
    """
    Accepts an http(s) URL; returns list of PIL.Image for each page.
    Handles common image types and PDFs.
    """
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download document")

    content_type = resp.headers.get("content-type", "").lower()
    data = resp.content

    # If PDF
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        # convert_from_bytes returns list of PIL Images
        images = convert_from_bytes(data, dpi=300)
        return images

    # Otherwise try to open as an image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {e}")


def preprocess_image_for_ocr(pil_img: Image.Image) -> np.ndarray:
    """Basic preprocessing: convert to grayscale, denoise, adaptive thresholding."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize if tiny
    h, w = gray.shape
    scale = 1.0
    if max(w, h) < 1200:
        scale = 1200.0 / max(w, h)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Apply slight blur then adaptive threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 61, 11)
    # Morphological open to remove small noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return processed


def ocr_image_with_positions(img_np: np.ndarray) -> Dict[str, Any]:
    """
    Runs pytesseract.image_to_data and returns the result dict.
    Contains word-level boxes and text.
    """
    custom_oem_psm_config = r'--oem 3 --psm 6'  # assume a uniform block
    ocr_data = pytesseract.image_to_data(img_np, output_type=Output.DICT, config=custom_oem_psm_config)
    return ocr_data


def cluster_rows_by_y(ocr_data: Dict[str, Any], y_tol: int = 10) -> List[List[int]]:
    """
    Group word indices into row clusters by their top (y) coordinate.
    Returns list of lists of word indices belonging to same visual row.
    """
    n = len(ocr_data['text'])
    rows = []  # list of lists of indices
    y_centers = []
    for i in range(n):
        text = ocr_data['text'][i].strip()
        if text == "":
            continue
        top = int(ocr_data['top'][i])
        height = int(ocr_data['height'][i])
        cy = top + height // 2
        # assign to existing row cluster if within tol
        placed = False
        for j, center in enumerate(y_centers):
            if abs(center - cy) <= y_tol:
                rows[j].append(i)
                # update center as mean
                y_centers[j] = int((y_centers[j] * (len(rows[j]) - 1) + cy) / len(rows[j]))
                placed = True
                break
        if not placed:
            rows.append([i])
            y_centers.append(cy)
    # sort rows by y-center ascending (top to bottom)
    order = sorted(range(len(y_centers)), key=lambda k: y_centers[k])
    return [rows[i] for i in order]


money_regex = re.compile(r'^\(?\d{1,3}(?:[,\d]{0,}\d)?(?:\.\d{1,2})\)?$')  # e.g., 1,234.56 or (1,234.56)


def parse_row_fields(ocr_data: Dict[str, Any], row_indices: List[int]) -> Dict[str, Any]:
    """
    Given a list of word indices that belong to a single visual row, try to parse:
    item_name, qty, rate, amount.
    Heuristics:
    - Rightmost numeric token is likely item_amount.
    - Second-rightmost numeric token is likely rate.
    - Third-rightmost numeric token could be qty.
    - Remaining left tokens form item_name.
    """
    words = []
    for i in row_indices:
        txt = ocr_data['text'][i].strip()
        if txt == "":
            continue
        left = int(ocr_data['left'][i])
        words.append((left, txt))

    if not words:
        return None

    # sort left-to-right
    words = sorted(words, key=lambda x: x[0])
    tokens = [t for _, t in words]

    # normalize remove common non-breaking spaces and weird bullets
    tokens = [t.replace('\xa0', ' ').replace('•', '').strip() for t in tokens if t.strip() != ""]

    # identify numeric tokens (strip commas and parentheses)
    def norm_num(tok):
        tok2 = tok.replace(',', '').replace('₹', '').replace('Rs.', '').replace('Rs', '').strip()
        tok2 = tok2.replace('(', '-').replace(')', '')
        # remove stray non-digit chars except . and -
        tok2 = re.sub(r'[^0-9\.\-]', '', tok2)
        try:
            if tok2 == "":
                return None
            val = float(tok2)
            return val
        except:
            return None

    numeric_flags = [norm_num(t) for t in tokens]

    # find last numeric tokens
    numbers = []
    for idx, val in enumerate(numeric_flags):
        if val is not None:
            numbers.append((idx, val))

    # default fields
    item_amount = None
    item_rate = None
    item_qty = None

    if len(numbers) >= 1:
        # last numeric = amount
        item_amount = numbers[-1][1]
    if len(numbers) >= 2:
        # second last = maybe rate
        item_rate = numbers[-2][1]
    if len(numbers) >= 3:
        # third last = maybe qty
        item_qty = numbers[-3][1]

    # If some numeric fields look like integers and are small, treat as qty
    if item_qty is None and item_rate is not None and item_amount is not None:
        # If item_rate * round(qty) ~= amount, infer qty
        if item_rate != 0:
            inferred_qty = item_amount / item_rate
            # if inferred_qty close to small integer, accept
            if 0.01 < inferred_qty < 1000 and abs(inferred_qty - round(inferred_qty)) < 0.02:
                item_qty = round(inferred_qty)

    # item_name = tokens left of the first numeric token used for qty/rate/amount
    first_num_idx = numbers[0][0] if numbers else len(tokens)
    name_tokens = tokens[:first_num_idx]
    item_name = " ".join(name_tokens).strip()
    # cleanup common trailing words like 'pcs' 'nos' etc
    item_name = re.sub(r'\bpcs?\b|\bnos?\b|\bunits?\b', '', item_name, flags=re.I).strip()

    # fallback: if name empty, maybe the full row is name (no numbers)
    if item_name == "":
        item_name = " ".join(tokens)

    # Return as dict
    return {
        "item_name": item_name if item_name else None,
        "item_quantity": float(item_qty) if item_qty is not None else None,
        "item_rate": float(item_rate) if item_rate is not None else None,
        "item_amount": float(item_amount) if item_amount is not None else None
    }


def is_header_row(text_line: str) -> bool:
    """Basic keyword search to identify header-like rows (skip these)."""
    lower = text_line.lower()
    header_words = ["description", "qty", "quantity", "rate", "amount", "total", "price", "item", "sr no", "s.no", "s.no."]
    hits = sum(1 for w in header_words if w in lower)
    return hits >= 1


def detect_page_type(full_text: str) -> str:
    """Return a page_type for schema: 'Bill Detail' | 'Final Bill' | 'Pharmacy'."""
    t = full_text.lower()
    if "pharmacy" in t or "medicine" in t:
        return "Pharmacy"
    # final bill page often has grand total keywords
    if re.search(r'\bgrand total\b|\bnet total\b|\bfinal total\b|\bamount payable\b|\bbalance due\b', t):
        return "Final Bill"
    # default
    return "Bill Detail"


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple dedupe using (normalized name, amount) tuple. Keeps first occurrence."""
    seen = set()
    out = []
    for it in items:
        name = (it.get("item_name") or "").strip().lower()
        amt = it.get("item_amount")
        key = (re.sub(r'\s+', ' ', name), round(amt, 2) if amt is not None else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# --------------- API request/response models ---------------

class ExtractRequest(BaseModel):
    document: str


@app.post("/extract-bill-data")
async def extract_bill_data(req: ExtractRequest, request: Request):
    # 1) Download -> pages (PIL images)
    try:
        pages = download_document_to_images(req.document)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download/convert error: {e}")

    pagewise_line_items = []
    all_items = []

    # Iterate pages
    for pno, pil_img in enumerate(pages, start=1):
        try:
            proc = preprocess_image_for_ocr(pil_img)
            ocr = ocr_image_with_positions(proc)
        except Exception as e:
            # fallback: try OCR on original
            try:
                ocr = ocr_image_with_positions(np.array(pil_img.convert("L")))
            except Exception:
                ocr = {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}

        # build full text for page type detection
        page_text = " ".join([t for t in ocr.get('text', []) if t.strip() != ""])
        page_type = detect_page_type(page_text)

        # cluster into rows
        row_clusters = cluster_rows_by_y(ocr, y_tol=12)

        page_items = []
        for cluster in row_clusters:
            parsed = parse_row_fields(ocr, cluster)
            if not parsed:
                continue
            # ignore header-like rows
            text_line = parsed.get("item_name") or ""
            if is_header_row(text_line):
                continue
            # require at least an amount or numeric evidence; otherwise skip
            if parsed.get("item_amount") is None and parsed.get("item_rate") is None:
                # skip rows that have no numeric info (very likely non-line text)
                continue

            # If item_name is short (like a single digit), try to reconstruct from tokens
            if parsed.get("item_name") and len(parsed["item_name"]) < 2 and parsed.get("item_amount") is not None:
                # combine the entire cluster tokens into name
                tokens = [ocr['text'][i].strip() for i in cluster if ocr['text'][i].strip() != ""]
                parsed["item_name"] = " ".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]

            # set None -> keep null in schema; but convert to floats where present
            item = {
                "item_name": parsed.get("item_name") or "",
                "item_amount": float(parsed["item_amount"]) if parsed.get("item_amount") is not None else None,
                "item_rate": float(parsed["item_rate"]) if parsed.get("item_rate") is not None else None,
                "item_quantity": float(parsed["item_quantity"]) if parsed.get("item_quantity") is not None else None
            }
            page_items.append(item)
            all_items.append(item)

        # dedupe within page
        page_items = dedupe_items(page_items)

        pagewise_line_items.append({
            "page_no": str(pno),
            "page_type": page_type,
            "bill_items": page_items
        })

    # dedupe across pages (so we don't double count)
    unique_items = dedupe_items(all_items)

    total_item_count = len(unique_items)

    # Build response (token_usage are dummy realistic values since no LLM used)
    response = {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": pagewise_line_items,
            "total_item_count": total_item_count
        }
    }

    return JSONResponse(status_code=200, content=response)


# simple healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
