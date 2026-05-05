"""
toc_detector.py
Scans extracted .txt files to find the page(s) that contain a Table of Contents.
Keywords are loaded from the configurable file defined in config.py.

Matching rule: a keyword must appear as a standalone line (heading) — i.e. the
stripped, upper-cased line equals the keyword exactly. This avoids false matches
where ToC keywords appear mid-sentence in body text.
"""

import re
from pathlib import Path
from src.config import TOC_KEYWORDS_FILE, TOC_SEARCH_MAX_PAGE, TOC_PAGES_TO_SEND


def load_toc_keywords() -> list[str]:
    """
    Read toc_keywords.txt and return a list of upper-cased keywords.
    Lines starting with # and blank lines are ignored.
    """
    if not TOC_KEYWORDS_FILE.exists():
        return ["CONTENTS", "CONTENT", "TABLE OF CONTENTS"]

    keywords = []
    for line in TOC_KEYWORDS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keywords.append(line.upper())
    return keywords


def _is_heading_match(page_text: str, keywords: list[str]) -> bool:
    """
    Return True if any keyword appears as a standalone heading line in page_text.
    Matches exact lines and short lines that START with a keyword followed only
    by a qualifier (e.g. "CONTENTS PART II") — the line must be ≤ 40 chars to
    avoid matching sentences that happen to begin with a keyword word.
    """
    for line in page_text.splitlines():
        stripped = line.strip().upper()
        if not stripped:
            continue
        if stripped in keywords:
            return True
        # Short heading lines only (avoids matching mid-sentence occurrences)
        if len(stripped) <= 40:
            for kw in keywords:
                if stripped.startswith(kw) and (
                    len(stripped) == len(kw) or stripped[len(kw)] in (" ", "\t")
                ):
                    return True
    return False


def find_toc_pages(txt_files: list[Path]) -> list[Path]:
    """
    Search through txt_files (ordered by page) up to TOC_SEARCH_MAX_PAGE pages.
    Returns a slice of txt_files starting from the first ToC page,
    up to TOC_PAGES_TO_SEND pages long.

    Returns an empty list if no ToC page is found.
    """
    keywords = load_toc_keywords()
    search_limit = min(TOC_SEARCH_MAX_PAGE, len(txt_files))

    for i in range(search_limit):
        page_text = txt_files[i].read_text(encoding="utf-8")
        if _is_heading_match(page_text, keywords):
            end = min(i + TOC_PAGES_TO_SEND, len(txt_files))
            return txt_files[i:end]

    return []
