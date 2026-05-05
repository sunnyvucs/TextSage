"""
toc_extractor.py
Sends ToC page images + extracted text to Groq vision model.
Using both images and text ensures chapter names in styled headers
that PyMuPDF may miss are still captured correctly.
"""

import base64
import json
import logging
import re
import time
from pathlib import Path

import fitz
from groq import Groq

from src.config import GROQ_API_KEY, GROQ_VISION_MODEL
from src.llm_client import _salvage_truncated_json, _strip_fences

log = logging.getLogger(__name__)

_PROMPT = """\
Below is the COMPLETE extracted text of the Table of Contents of a school textbook.
Images of the first few ToC pages are also provided to help read chapter titles clearly.

IMPORTANT: The text below is the authoritative and complete source — it contains ALL \
chapters including those not visible in the images. Extract every chapter present in \
the text, not just those visible in the images.

IMPORTANT: Some books include a reference list of chapters from a companion Part I or \
Part II volume. DO NOT extract those — extract ONLY the chapters that belong to THIS \
book (the ones whose content appears in this volume, i.e. the chapters with detailed \
topic listings under them).

Rules:
- chapter_number: use the word or numeral exactly as printed (e.g. "ONE", "1", "TWO")
- chapter_name: the full chapter title as printed — never null, never empty
- start_page: the first page number listed for that chapter or topic
- topics: all sub-sections with their numbers, names, and page numbers
- If a topic name or page number is genuinely absent, use null

Return ONLY this JSON, no explanation:

{
  "chapters": [
    {
      "chapter_number": "1",
      "chapter_name": "CHAPTER TITLE HERE",
      "start_page": 3,
      "topics": [
        {"topic_number": "1.1", "topic_name": "Introduction", "start_page": 3}
      ]
    }
  ]
}

COMPLETE TABLE OF CONTENTS TEXT:
"""


def _render_page_b64(pdf_path: Path, page_index: int = 0, zoom: float = 1.5) -> str:
    doc = fitz.open(str(pdf_path))
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return base64.b64encode(pix.tobytes("png")).decode()


def extract_toc(toc_txt_files: list[Path], output_path: Path, toc_page_pdfs: list[Path] = None) -> Path:
    """
    Send ToC page images + text to Groq vision and save structured JSON.

    toc_txt_files   : list of .txt files containing extracted ToC text
    output_path     : where to write toc.json
    toc_page_pdfs   : list of single-page PDF files for the ToC pages (optional)
                      if None, images are skipped and only text is sent
    """
    combined_text = "\n\n".join(
        f.read_text(encoding="utf-8") for f in toc_txt_files
    ).strip()
    toc_text = combined_text[:6000]

    # Build multimodal content: images first, then text + prompt
    content = []

    if toc_page_pdfs:
        for page_pdf in toc_page_pdfs[:5]:   # Groq vision max 5 images
            try:
                b64 = _render_page_b64(page_pdf)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except Exception as exc:
                log.warning("  [ToC] Could not render %s: %s", page_pdf.name, exc)

    content.append({
        "type": "text",
        "text": _PROMPT + toc_text,
    })

    # Save debug file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = output_path.parent / "toc_prompt_debug.txt"
    img_count = len([c for c in content if c["type"] == "image_url"])
    debug_path.write_text(
        f"=== ToC Extraction ===\nImages sent: {img_count}\n\n"
        f"=== Extracted text sent ===\n{toc_text}\n",
        encoding="utf-8",
    )

    log.info("  [ToC] Sending %d image(s) + text to Groq vision...", img_count)
    t0 = time.perf_counter()

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = _strip_fences(response.choices[0].message.content.strip())
        t_total = time.perf_counter() - t0
        log.info("  [ToC] Response received  (%.2fs)", t_total)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = _salvage_truncated_json(raw)
            if result:
                log.warning("  [ToC] Response was truncated — partial data recovered")
            else:
                raise ValueError(f"Response not valid JSON:\n{raw[:500]}")

        chapter_count = len(result.get("chapters", []))
        topic_count   = sum(len(c.get("topics") or []) for c in result.get("chapters", []))
        log.info("  [ToC] Parsed: %d chapters, %d topics", chapter_count, topic_count)

    except Exception as exc:
        t_total = time.perf_counter() - t0
        log.error("  [ToC] Call failed after %.2fs: %s", t_total, exc)
        result = {"error": str(exc), "chapters": []}

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("  [ToC] Saved -> %s", output_path.name)
    return output_path
