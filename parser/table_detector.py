# app/parser/table_detector.py
from typing import Dict, Any, List
import numpy as np

def detect_table_columns(ocr_data: Dict[str, Any], x_gap_threshold: int = 30) -> List[int]:
    """
    Infer approximate column x boundaries from OCR word left positions.
    Returns list of x positions representing column separators (approx).
    Simple approach:
    - gather all left positions
    - cluster into groups by proximity (x_gap_threshold)
    - return cluster centers (sorted)
    This is a helper for downstream parsing if you prefer column splitting.
    """
    lefts = []
    for i, t in enumerate(ocr_data.get('text', [])):
        if t.strip() == "":
            continue
        lefts.append(int(ocr_data.get('left', [0]*len(ocr_data.get('text', [])))[i]))

    if not lefts:
        return []

    lefts = sorted(lefts)
    clusters = []
    current = [lefts[0]]
    for x in lefts[1:]:
        if x - current[-1] <= x_gap_threshold:
            current.append(x)
        else:
            clusters.append(current)
            current = [x]
    clusters.append(current)
    # centers
    centers = [int(sum(c)/len(c)) for c in clusters]
    return sorted(centers)


def assign_tokens_to_columns(ocr_data: Dict[str, Any], cols: List[int]) -> Dict[int, list]:
    """
    Assign each token index to the nearest column center.
    Returns mapping: column_center -> list of token indices
    """
    mapping = {c: [] for c in cols}
    lefts = ocr_data.get('left', [])
    for i, t in enumerate(ocr_data.get('text', [])):
        if t.strip() == "":
            continue
        l = int(lefts[i])
        # find nearest center
        best = min(cols, key=lambda c: abs(c - l))
        mapping[best].append(i)
    return mapping
