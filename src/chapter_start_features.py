"""
chapter_start_features.py
Structural scoring helpers for identifying true chapter-start pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

_NEGATIVE_PREFIXES = (
    "EXERCISE",
    "MISCELLANEOUS EXERCISE",
    "SUMMARY",
    "CONTENTS",
    "FOREWORD",
    "PREFACE",
    "ACKNOWLEDGEMENT",
    "ACKNOWLEDGEMENTS",
    "RATIONALISATION",
    "APPENDIX",
    "ANSWERS",
)


def _normalize(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


@dataclass
class StructuralScore:
    score: float
    signals: dict[str, float]
    first_line: str


def score_page(txt_path: Path, chapter_number: str, chapter_name: str) -> StructuralScore:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    top_lines = lines[:6]
    opener_lines = lines[:20]
    top_text = " ".join(top_lines)
    opener_text = " ".join(opener_lines)
    top_upper = _normalize(top_text)
    opener_upper = _normalize(opener_text)
    first_upper = _normalize(first_line)
    chapter_number = str(chapter_number).strip()
    chapter_name = chapter_name.strip()

    title_match = 0.0
    for line in opener_lines[:12]:
        title_match = max(title_match, _fuzzy_ratio(chapter_name, line))
    title_match = max(title_match, _fuzzy_ratio(chapter_name, opener_text[:400]))

    chapter_num_match = 0.0
    chapter_patterns = [
        rf"\b{re.escape(chapter_number)}\b",
        rf"\bCHAPTER\s+{re.escape(chapter_number)}\b",
        rf"\b{re.escape(chapter_number)}\s*UNIT\b",
        rf"\bUNIT\s*{re.escape(chapter_number)}\b",
    ]
    if any(re.search(pat, opener_upper) for pat in chapter_patterns):
        chapter_num_match = 1.0

    section_start = 0.0
    if chapter_number.isdigit():
        if re.search(rf"\b{re.escape(chapter_number)}\.1\b", opener_upper):
            section_start = 1.0
        elif re.search(rf"\b{re.escape(chapter_number)}\.\d+\b", opener_upper):
            section_start = 0.5

    intro_marker = 1.0 if "INTRODUCTION" in opener_upper else 0.0
    objectives_marker = 1.0 if (
        "OBJECTIVES" in opener_upper or "AFTER STUDYING THIS UNIT" in opener_upper
    ) else 0.0
    quote_marker = 1.0 if any(
        line.startswith(("v ", "V ")) or " - " in line or " â€” " in line or " â€“ " in line
        for line in top_lines[1:4]
    ) else 0.0

    negative_marker = 1.0 if any(first_upper.startswith(prefix) for prefix in _NEGATIVE_PREFIXES) else 0.0
    exercise_marker = 1.0 if (
        "EXERCISES" in opener_upper
        or first_upper.startswith("EXERCISE")
        or first_upper.startswith("MISCELLANEOUS EXERCISE")
    ) else 0.0

    uppercase_heading = 0.0
    if first_line:
        letters = [ch for ch in first_line if ch.isalpha()]
        if letters:
            uppercase_heading = sum(1 for ch in letters if ch.isupper()) / len(letters)

    score = (
        0.30 * title_match
        + 0.20 * chapter_num_match
        + 0.16 * section_start
        + 0.10 * intro_marker
        + 0.20 * objectives_marker
        + 0.08 * quote_marker
        + 0.08 * uppercase_heading
        - 0.55 * negative_marker
        - 0.45 * exercise_marker
    )

    signals = {
        "title_match": round(title_match, 4),
        "chapter_number": round(chapter_num_match, 4),
        "section_start": round(section_start, 4),
        "introduction": round(intro_marker, 4),
        "objectives_marker": round(objectives_marker, 4),
        "quote_marker": round(quote_marker, 4),
        "uppercase_heading": round(uppercase_heading, 4),
        "negative_marker": round(negative_marker, 4),
        "exercise_marker": round(exercise_marker, 4),
        "structural_score": round(score, 4),
    }
    return StructuralScore(score=score, signals=signals, first_line=first_line)
