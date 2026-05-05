"""
text_extractor.py
Extracts plain text from a single-page PDF and writes it to a .txt file.
Falls back to pytesseract OCR when native text extraction yields nothing.
"""

import fitz  # PyMuPDF
from pathlib import Path


def _extract_native_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    # Sort blocks by vertical position (y0) so text reads top-to-bottom
    # regardless of the order blocks appear in the PDF content stream
    blocks = doc[0].get_text("blocks")
    blocks_sorted = sorted(blocks, key=lambda b: b[1])  # b[1] = y0
    text = "\n".join(b[4].strip() for b in blocks_sorted if b[4].strip())
    doc.close()
    return text.strip()


def _extract_via_ocr(pdf_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        pix = doc[0].get_pixmap(dpi=200)
        doc.close()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except Exception as exc:
        return f"[OCR failed: {exc}]"


def extract_text(page_pdf: Path, output_dir: Path) -> Path:
    """
    Extract text from page_pdf (a single-page PDF) and save it as a .txt file
    in output_dir with the same stem name.

    Returns the path to the created .txt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / (page_pdf.stem + ".txt")

    text = _extract_native_text(page_pdf)
    if not text:
        text = _extract_via_ocr(page_pdf)

    txt_path.write_text(text, encoding="utf-8")
    return txt_path


def extract_text_all(page_pdfs: list[Path], output_dir: Path, workers: int = 8) -> list[Path]:
    """
    Extract text for every page PDF in the list using a thread pool.
    Returns list of .txt file paths in the same order as input.
    workers: how many pages to process in parallel (default 8).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(extract_text, pdf, output_dir): i
            for i, pdf in enumerate(page_pdfs)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()

    return [results[i] for i in range(len(page_pdfs))]
