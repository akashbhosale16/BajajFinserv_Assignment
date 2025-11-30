# utils/regex_utils.py

import re

# Common numeric patterns
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
