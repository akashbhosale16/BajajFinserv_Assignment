
import re

def detect_page_type(full_text: str) -> str:
    """
    Determine page type based on keywords.
    Returns one of: "Bill Detail", "Final Bill", "Pharmacy"
    """
    if not full_text:
        return "Bill Detail"

    t = full_text.lower()
    # Pharmacy cues
    pharm_keywords = ["pharmacy", "medicine", "medicines", "batch", "mrp", "expiry", "drug"]
    for kw in pharm_keywords:
        if kw in t:
            return "Pharmacy"

    # Final bill / totals cues
    final_keywords = ["grand total", "net total", "final total", "amount payable", "amount due", "balance due", "total payable", "net payable", "total amount"]
    for kw in final_keywords:
        if kw in t:
            return "Final Bill"

    # Default
    return "Bill Detail"
