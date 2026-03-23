#!/usr/bin/env python3
"""Extract text from a PDF file by page range.

Usage:
    python scripts/read_pdf.py <pdf_path> <start_page> <end_page>

Pages are 1-indexed. Output is plain text.
"""
import sys
import fitz  # pymupdf

pdf_path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
doc = fitz.open(pdf_path)
for i in range(start - 1, min(end, len(doc))):
    print(f"\n--- Page {i+1} ---")
    print(doc[i].get_text())
