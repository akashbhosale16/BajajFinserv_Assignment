# app/parser/line_item_parser.py
import re
from typing import List, Dict, Any

money_like = re.compile(r'^\(?-?\d{1,3}(?:[,\d]{0,}\d)?(?:\.\d{1,2})\)?$')

def norm_number_token(tok: str):
    if tok is None:
        return None
    s = tok.replace(',', '').replace('₹', '').replace('rs.', '').replace('rs', '').strip()
    s = s.replace('(', '-').replace(')', '')
    s = re.sub(r'[^\d\.\-]', '', s)
    if s == "":
        return None
    try:
        return float(s)
    except:
        return None


def cluster_rows_by_y(ocr_data: Dict[str, Any], y_tol: int = 10) -> List[List[int]]:
    """
    Group word indices into approximate visual rows by y-coordinate center.
    Returns list of lists (word indices).
    """
    n = len(ocr_data.get('text', []))
    rows = []
    y_centers = []
    for i in range(n):
        txt = ocr_data['text'][i].strip()
        if txt == "":
            continue
        top = int(ocr_data['top'][i])
        height = int(ocr_data['height'][i])
        cy = top + height // 2
        placed = False
        for j, center in enumerate(y_centers):
            if abs(center - cy) <= y_tol:
                rows[j].append(i)
                # update average center
                y_centers[j] = int((y_centers[j] * (len(rows[j]) - 1) + cy) / len(rows[j]))
                placed = True
                break
        if not placed:
            rows.append([i])
            y_centers.append(cy)
    # sort rows top-to-bottom
    order = sorted(range(len(y_centers)), key=lambda k: y_centers[k])
    return [rows[i] for i in order]


def is_header_row(text_line: str) -> bool:
    lower = (text_line or "").lower()
    header_words = ["description", "qty", "quantity", "rate", "amount", "total", "price", "item", "sr no", "s.no", "sr.", "sl no"]
    hits = sum(1 for w in header_words if w in lower)
    return hits >= 1


def parse_row_fields(ocr_data: Dict[str, Any], row_indices: List[int]) -> Dict[str, Any]:
    """
    Parse a visual row to extract: item_name, item_quantity, item_rate, item_amount.
    Heuristic:
    - tokens sorted left-to-right
    - rightmost numeric -> amount
    - second-rightmost numeric -> rate
    - third-rightmost numeric -> qty (if present)
    - left tokens before first numeric are item name
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
    words = sorted(words, key=lambda x: x[0])
    tokens = [t for _, t in words]
    tokens = [t.replace('\xa0', ' ').strip() for t in tokens if t.strip() != ""]

    numerics = []
    for idx, t in enumerate(tokens):
        val = norm_number_token(t)
        if val is not None:
            numerics.append((idx, val))

    item_amount = None
    item_rate = None
    item_qty = None

    if len(numerics) >= 1:
        item_amount = numerics[-1][1]
    if len(numerics) >= 2:
        item_rate = numerics[-2][1]
    if len(numerics) >= 3:
        item_qty = numerics[-3][1]

    # If qty missing but rate and amount exist, infer qty
    if item_qty is None and item_rate not in (None, 0) and item_amount is not None:
        inferred_qty = item_amount / item_rate if item_rate != 0 else None
        if inferred_qty:
            # Accept inferred qty if reasonably near integer
            if 0.01 < inferred_qty < 10000 and abs(inferred_qty - round(inferred_qty)) < 0.05:
                item_qty = round(inferred_qty)

    first_num_idx = numerics[0][0] if numerics else len(tokens)
    name_tokens = tokens[:first_num_idx]
    item_name = " ".join(name_tokens).strip()
    item_name = re.sub(r'\bpcs?\b|\bnos?\b|\bunits?\b', '', item_name, flags=re.I).strip()
    if item_name == "":
        item_name = " ".join(tokens).strip()

    return {
        "item_name": item_name if item_name else None,
        "item_quantity": float(item_qty) if item_qty is not None else None,
        "item_rate": float(item_rate) if item_rate is not None else None,
        "item_amount": float(item_amount) if item_amount is not None else None
    }


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate items using normalized (name, rounded_amount) tuple.
    Keeps first occurrence.
    """
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
