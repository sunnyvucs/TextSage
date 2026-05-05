"""
pdf_splitter.py
Splits a source PDF into one PDF per page and saves them into a working directory.
"""

import fitz  # PyMuPDF
from pathlib import Path


def split_pdf(source_pdf: Path, output_dir: Path) -> list[Path]:
    """
    Split every page of source_pdf into a separate PDF file inside output_dir.

    Returns a list of paths to the created per-page PDFs, ordered by page index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(source_pdf))
    page_pdfs: list[Path] = []

    for page_index in range(len(doc)):
        page_filename = output_dir / f"page_{page_index + 1:04d}.pdf"
        single_page_doc = fitz.open()
        single_page_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        single_page_doc.save(str(page_filename))
        single_page_doc.close()
        page_pdfs.append(page_filename)

    doc.close()
    return page_pdfs
