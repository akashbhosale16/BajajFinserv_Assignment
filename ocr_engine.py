import re
from typing import Dict, Any
import numpy as np
import pytesseract
from pytesseract import Output

# Wrapper around pytesseract to return consistent dict structure
def ocr_image_with_positions(img_np: np.ndarray) -> Dict[str, Any]:
    """
    Run pytesseract OCR and return a dict with keys similar to pytesseract Output.DICT:
    'text', 'left', 'top', 'width', 'height', 'conf'
    img_np: numpy array (grayscale or color)
    """
    # Ensure image is in a format pytesseract expects (uint8)
    if img_np.dtype != 'uint8':
        img_uint8 = (img_np).astype('uint8')
    else:
        img_uint8 = img_np

    # Use OEM LSTM and PSM 6 (assume a block of text) - can be tuned
    custom_oem_psm_config = r'--oem 3 --psm 6'
    try:
        ocr_data = pytesseract.image_to_data(img_uint8, output_type=Output.DICT, config=custom_oem_psm_config)
    except Exception:
        # fallback to single-line PSM if full page fails
        ocr_data = pytesseract.image_to_data(img_uint8, output_type=Output.DICT, config='--oem 3 --psm 3')

    # Normalize results: ensure lists are present
    keys = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text']
    for k in keys:
        if k not in ocr_data:
            ocr_data[k] = []

    # Clean texts (strip)
    texts = [t.strip() for t in ocr_data.get('text', [])]
    ocr_data['text'] = texts

    return ocr_data


# utility: join OCR text into page text
def ocr_full_text(ocr_data: Dict[str, Any]) -> str:
    return " ".join([t for t in ocr_data.get('text', []) if t and t.strip() != ""])
