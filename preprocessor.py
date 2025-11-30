# app/preprocessor.py
from PIL import Image, ImageOps
import numpy as np
import cv2
import math


def pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV (numpy) RGB array"""
    arr = np.array(pil_img)
    # If RGBA convert to RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def preprocess_image_for_ocr(pil_img: Image.Image, target_min_max_dim: int = 1200) -> np.ndarray:
    """
    Basic preprocessing pipeline:
    - convert to RGB
    - resize if small (scale up)
    - convert to gray
    - denoise with Gaussian
    - adaptive thresholding
    - return uint8 image (grayscale) suitable for pytesseract
    """
    # Convert to RGB numpy
    img_rgb = pil_to_cv2(pil_img.convert("RGB"))

    h, w = img_rgb.shape[:2]
    max_dim = max(w, h)
    if max_dim < target_min_max_dim:
        scale = target_min_max_dim / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to gray
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Normalize contrast using histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Slight blur to reduce small noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding (works for varying illumination)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 61, 11)

    # Morphological opening to remove small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # Optionally deskew if large skew detected
    deskewed = deskew_image(opened)

    return deskewed


def deskew_image(img):
    """
    Deskew binary image using moments or Hough line approach.
    Returns deskewed grayscale image.
    """
    # find coordinates of non-zero pixels
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # If negligible angle, skip
    if abs(angle) < 0.1:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
