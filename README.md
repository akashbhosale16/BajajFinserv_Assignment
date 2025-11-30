# Bill Extraction API – HackRx Challenge

This repository provides a complete pipeline and REST API for extracting line items, rates, quantities, and totals from multi-page bill images or PDFs. The system uses OCR and rule-based parsing only. No LLMs and no custom-trained models are used.

The API strictly follows the schema defined in the HackRx challenge. The solution focuses on accuracy, avoidance of double counting, and robust extraction across varied bill formats.

---

## 1. Overview

The system performs the following:

1. Downloads bill images or PDFs from provided URLs.
2. Converts PDFs into page-wise images.
3. Preprocesses images to improve OCR accuracy.
4. Extracts text using Tesseract or PaddleOCR.
5. Classifies each page as:
   - Bill Detail
   - Pharmacy
   - Final Bill
6. Detects table structures and extracts line items.
7. Produces structured output matching the required API format.
8. Ensures no duplicate line items or subtotal rows are counted.

---

## 2. Features

- Multi-page document handling  
- OCR-based text extraction  
- Automatic page classification  
- Table and line-item extraction  
- Amount, rate, and quantity parsing  
- Removal of totals, subtotals, and repeated headers  
- Output strictly matching HackRx format  
- FastAPI-based REST API  
- Deterministic, rules-based parsing (no ML models used)

---

## 3. Repository Structure
.
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── ocr_engine.py         # OCR wrapper using Tesseract or PaddleOCR
│   ├── preprocessor.py       # Image cleanup (deskew, denoise, threshold)
│   ├── parser/
│   │   ├── page_classifier.py
│   │   ├── line_item_parser.py
│   │   ├── table_detector.py
│   └── utils/
│       ├── downloader.py
│       ├── file_utils.py
│       ├── regex_utils.py
│
├── requirements.txt
├── README.md
└── sample_output.json
---

## 4. Installation

### System Dependencies

Install Tesseract OCR:

Ubuntu:
---

## 5. Running the API

Start the FastAPI server:

---

## 6. API Specification

### Endpoint  
POST /extract-bill-data

### Request Body

---

## 7. Processing Pipeline

### 1. Download & Normalize Files  
URLs are downloaded, PDFs are converted into page-wise images, and all images undergo preprocessing.

### 2. OCR Extraction  
Tesseract or PaddleOCR extracts text and bounding boxes.

### 3. Page Classification  
Simple rule-based logic determines:
- Bill Detail
- Pharmacy
- Final Bill

### 4. Table Detection  
- Detects headers using fuzzy matching.
- Identifies columns such as item name, quantity, rate, and amount.

### 5. Line Item Parsing  
Handles formats such as:
- item qty rate amount
- item qty x rate amount
- item - amount
- pharmacy-style codes

### 6. Duplicate and Subtotal Handling  
- Removes rows containing keywords like subtotal, total, net amount, gross.
- Avoids counting repeated headers.
- Uses row hashing to prevent duplicates.

### 7. Output Construction  
All detected line items are grouped by page and formatted according to the required schema.

---

## 8. Limitations

- Low-resolution images may reduce OCR accuracy.
- Handwritten bills are not supported.
- Some unique table layouts may require regex tuning.
- OCR misreads can affect decimal placement or quantity detection.

---

## 9. Future Enhancements

- PaddleOCR structured table recognition
- Bounding-box clustering for row grouping
- Heuristic optimization with more samples
- Confidence scoring for outputs

---

## 10. License

This project is provided for use in the HackRx challenge and for educational purposes. You may modify or distribute it as needed.

