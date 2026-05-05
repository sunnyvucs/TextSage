"""
subject_identifier.py
Sends the first 5 page images of a PDF to Groq vision model to identify
subject, part number, and class/grade number.
Returns a 3-tuple: (subject, part, class_number).
"""

import base64
import json
import logging
import re
from pathlib import Path

import fitz
from groq import Groq

from src.config import GROQ_API_KEY, GROQ_VISION_MODEL

log = logging.getLogger(__name__)
_PAGES_TO_SEND = 5

_PROMPT = (
    "These are the first few pages of a school textbook. "
    "Identify the subject, part number (if any), and class/grade number. "
    "Rules: "
    "subject = subject name only e.g. Biology, Chemistry, Physics, Mathematics. "
    "part = Arabic numeral string e.g. '1' or '2' — use null if the book is not split into parts. "
    "class_number = Arabic numeral string e.g. '12' not 'XII' or 'Twelve'. "
    'Return ONLY this JSON, no explanation: {"subject": "", "part": null, "class_number": ""}'
)


def _page_pdf_to_b64(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    return base64.b64encode(pix.tobytes("png")).decode()


def _roman_to_arabic(value: str) -> str:
    """Convert Roman numeral class/part to Arabic if needed."""
    roman_map = {
        "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
        "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
        "XI": "11", "XII": "12",
    }
    upper = value.strip().upper()
    return roman_map.get(upper, value.strip())


def _parse_response(raw: str) -> tuple[str, str | None, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[1:end]).strip()

    # Try direct JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting first JSON object from response
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in response: {raw[:200]}")
        data = json.loads(m.group())

    subject = str(data.get("subject") or "Unknown").strip() or "Unknown"

    part = data.get("part")
    if part and str(part).strip().lower() not in ("null", "none", ""):
        part = _roman_to_arabic(str(part))
    else:
        part = None

    class_number = str(data.get("class_number") or "Unknown").strip() or "Unknown"
    class_number = _roman_to_arabic(class_number)

    return subject, part, class_number


def identify_subject(pdf_dir: Path) -> tuple[str, str | None, str]:
    """
    Send first _PAGES_TO_SEND page images (from pdf_dir) to Groq vision model.
    Returns (subject, part, class_number). Falls back to ("Unknown", None, "Unknown") on error.
    pdf_dir: the pdfs/ sub-directory containing page_XXXX.pdf files.
    """
    page_pdfs = sorted(pdf_dir.glob("page_*.pdf"))[:_PAGES_TO_SEND]
    if not page_pdfs:
        log.warning("  [Subject] No page PDFs found in %s", pdf_dir)
        return ("Unknown", None, "Unknown")

    content = []
    for p in page_pdfs:
        try:
            b64 = _page_pdf_to_b64(p)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        except Exception as exc:
            log.warning("  [Subject] Could not render %s: %s", p.name, exc)

    if not content:
        return ("Unknown", None, "Unknown")

    content.append({"type": "text", "text": _PROMPT})

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        subject, part, class_number = _parse_response(raw)
        log.info("  [Subject] subject=%s  part=%s  class=%s", subject, part, class_number)
        return subject, part, class_number
    except Exception as exc:
        log.error("  [Subject] Identification failed: %s", exc)
        return ("Unknown", None, "Unknown")
