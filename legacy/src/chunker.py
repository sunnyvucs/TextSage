"""
chunker.py
Splits document content into chunks using MinerU's structured content_list.json
when available, falling back to regex-based txt scanning otherwise.

MinerU-based strategy (preferred):
  - text_level=1  → chapter-level headings
  - text_level=2  → topic/section headings (N.N pattern)
  - EXERCISE/Intext headings → exercise chunks
  - Example N.N headings → example chunks
  - Answers to... headings → skip section

Fallback strategy (txt files only):
  - structured: regex topic headings found in txt files
  - chapter:    toc has chapters but no topic markers
  - flat:       no structure detected

Output: chunks.json
"""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from src.config import CHUNK_TARGET_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_RATIO

log = logging.getLogger(__name__)

# ── Regex patterns (fallback txt mode only) ───────────────────────────────────
_TOPIC_HEADING_RE   = re.compile(r"^(\d+)\.(\d+)\s+(\S.*)", re.MULTILINE)
_EXERCISE_HEADING_RE = re.compile(
    r"^\s*(EXERCISES?|ADDITIONAL\s+EXERCISES?)(\s+\d+\.\d+)?\s*$", re.IGNORECASE
)
_INTEXT_HEADING_RE  = re.compile(r"^\s*Intext\s+Questions?\s*$", re.IGNORECASE)
_ANSWERS_HEADING_RE = re.compile(
    r"^\s*Answers\s+to\s+(?!.*\d{2,}\s*$).+", re.IGNORECASE
)
_EXAMPLE_HEADING_RE = re.compile(r"^Example\s+(\d+)\.(\d+)\s+(\S.*)", re.IGNORECASE)

# ── MinerU block classifiers ──────────────────────────────────────────────────
_TOPIC_NUM_RE    = re.compile(r"^(\d+)\s*\.\s*(\d+)")    # "7.2" or "7 . 2 Integration..."
_EXERCISE_BLK_RE = re.compile(
    r"^(EXERCISES?|ADDITIONAL\s+EXERCISES?)(\s+\d+\.\d+)?\s*$", re.IGNORECASE
)
_INTEXT_BLK_RE  = re.compile(r"^Intext\s+Questions?\s*$", re.IGNORECASE)
_ANSWERS_BLK_RE = re.compile(
    r"^Answers\s+to\s+(?!.*\d{2,}\s*$).+", re.IGNORECASE
)
_EXAMPLE_BLK_RE = re.compile(r"^Example\s+(\d+)[\.\s]", re.IGNORECASE)
# Matches "CHAPTER 7" or "Chapter 7" or "CHAPTER 7 EVOLUTION" — extracts the number
_CHAPTER_NUM_RE = re.compile(r"^chapter\s+(\d+)\b", re.IGNORECASE)
_UNIT_HEADER_RE = re.compile(r"^unit\s+", re.IGNORECASE)  # "UNIT VII GENETICS..." — section dividers


# ── Helpers ───────────────────────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return text.split()


def _split_into_chunks(text: str, target: int, min_words: int, overlap_ratio: float) -> list[str]:
    words = _words(text)
    if not words:
        return []
    overlap = max(1, int(target * overlap_ratio))
    chunks, start = [], 0
    while start < len(words):
        end = min(start + target, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        if len(chunk_words) >= min_words:
            chunks.append(chunk)
        elif chunks:
            chunks[-1] = chunks[-1] + " " + chunk
        else:
            chunks.append(chunk)
        start += target - overlap
    return chunks


def _build_topic_map(toc: dict) -> dict[str, dict]:
    topic_map = {}
    for ch in toc.get("chapters", []):
        ch_num = str(ch["chapter_number"])
        for t in ch.get("topics", []):
            t_num = str(t["topic_number"])
            topic_map[t_num] = {
                "chapter_number": ch_num,
                "chapter_name":   ch["chapter_name"],
                "topic_number":   t_num,
                "topic_name":     t["topic_name"],
            }
    return topic_map


def _build_chapter_map(toc: dict) -> dict[str, dict]:
    ch_map = {}
    for ch in toc.get("chapters", []):
        ch_num = str(ch["chapter_number"])
        ch_map[ch_num] = {
            "chapter_number": ch_num,
            "chapter_name":   ch["chapter_name"],
            "topic_number":   None,
            "topic_name":     None,
            "start_page":     ch.get("start_page"),  # from toc.json when available
        }
    return ch_map


def _build_toc_page_map(toc: dict) -> dict[str, int]:
    """Return {chapter_number: start_page} for chapters that have start_page in toc.json."""
    page_map = {}
    for ch in toc.get("chapters", []):
        ch_num = str(ch["chapter_number"])
        sp = ch.get("start_page")
        if sp and isinstance(sp, int):
            page_map[ch_num] = sp
    return page_map


def _page_num(txt_file: Path) -> int:
    m = re.search(r"(\d+)", txt_file.stem)
    return int(m.group(1)) if m else 0


def _fuzzy_match_topic(text: str, topic_map: dict, threshold: float = 0.75) -> str | None:
    """
    Given a heading text like "7.2 Integration as an Inverse Process",
    return the matching topic_number from topic_map using:
      1. Exact N.N prefix match (fast path)
      2. Fuzzy name match as fallback
    """
    # Fast path: extract N.N prefix
    m = _TOPIC_NUM_RE.match(text.strip())
    if m:
        t_num = f"{m.group(1)}.{m.group(2)}"
        if t_num in topic_map:
            return t_num

    # Fuzzy match on topic names (handles truncated or slightly different names)
    text_clean = re.sub(r"^\d+\.\d+\s*", "", text.strip()).lower()
    best_score, best_key = 0.0, None
    for t_num, meta in topic_map.items():
        name = meta["topic_name"].lower()
        if not name:
            continue
        score = SequenceMatcher(None, text_clean, name).ratio()
        if score > best_score:
            best_score, best_key = score, t_num
    if best_score >= threshold:
        return best_key
    return None


def _dedup_ocr(text: str) -> str:
    """
    Fix MinerU OCR duplication artifacts like "ElectrochemistryElectrochemistry"
    or "Haloalkanes andHaloalkanes and HaloarenesHaloarenes".
    Finds the shortest prefix that, when repeated/interleaved, reconstructs the text.
    """
    t = text.strip()
    n = len(t)
    # Try every split point from half-length down to 4 chars
    for split in range(n // 2, 3, -1):
        half = t[:split]
        # Direct repeat: "ABCABC"
        if t == half + half:
            return half
        # Overlapping repeat: remainder starts with beginning of half
        remainder = t[split:]
        if remainder and half.upper().startswith(remainder.upper()[:4]):
            return half
    return t


def _fuzzy_match_chapter(text: str, chapter_map: dict, threshold: float = 0.62) -> str | None:
    """
    Match a heading block text to a chapter number.
    Handles MinerU OCR duplication (e.g. "ElectrochemistryElectrochemistry")
    via deduplication before matching. Four strategies:
      0. Deduplicate OCR artifacts, then retry substring match
      1. Exact substring: chapter name appears verbatim inside block text
      2. Fuzzy ratio on the full cleaned text vs chapter name
      3. Sliding window fuzzy match for partial/merged OCR text
    """
    raw = re.sub(r"^chapter\s+\w+\s*", "", text.strip(), flags=re.IGNORECASE)
    text_up = raw.upper()
    deduped_up = _dedup_ocr(raw).upper()

    # Check both raw and deduped forms
    candidates = list({text_up, deduped_up})

    # Strategy 1: exact substring containment — only for multi-word or long chapter names
    # (short names like "AMINES" appear too often in body text to be reliable)
    for candidate in candidates:
        for ch_num, meta in chapter_map.items():
            name = meta["chapter_name"].upper()
            if name and len(name) >= 12 and name in candidate:
                return ch_num

    # Strategy 2: fuzzy ratio on full text (both raw and deduped)
    best_score, best_key = 0.0, None
    for candidate in candidates:
        for ch_num, meta in chapter_map.items():
            name = meta["chapter_name"].upper()
            if not name:
                continue
            score = SequenceMatcher(None, candidate, name).ratio()
            if score > best_score:
                best_score, best_key = score, ch_num
    if best_score >= threshold:
        return best_key

    # Strategy 3: sliding window on deduped text
    for ch_num, meta in chapter_map.items():
        name = meta["chapter_name"].upper()
        if not name or len(name) < 5:
            continue
        name_len = len(name)
        for candidate in candidates:
            for start in range(0, max(1, len(candidate) - name_len + 1), max(1, name_len // 3)):
                window = candidate[start:start + name_len + 10]
                score = SequenceMatcher(None, window[:name_len], name).ratio()
                if score >= 0.82:
                    return ch_num

    return None


def _detect_chapter_start(
    text: str,
    level,
    chapter_map: dict,
    matched_chapters: set,
    current_chapter: str | None,
    threshold_l1: float = 0.62,
    threshold_l2: float = 0.72,
) -> str | None:
    """
    Detect a chapter-start signal. Returns chapter_number or None.

    Priority:
      1. "CHAPTER N name" on one block → regex match on number (exact, no fuzzy needed).
      2. level=1 fuzzy match → any block at top heading level.
      3. level=2 fuzzy match → for books (e.g. Biology) that use level=2 for chapter headings.

    Does NOT handle "CHAPTER N" standalone (no name) — caller handles that with
    pending_chapter_num because it needs to look at the next block.
    """
    # Strategy 1: "CHAPTER N ..." — extract number directly from regex.
    # Reject multi-line blocks — unit index pages list "Chapter N\nName" on multiple lines.
    if "\n" not in text:
        m_ch = _CHAPTER_NUM_RE.match(text)
        if m_ch:
            ch_n = m_ch.group(1)
            rest = text[m_ch.end():].strip()
            if ch_n in chapter_map and ch_n not in matched_chapters and ch_n != current_chapter:
                if rest:  # "CHAPTER N name" on one block — switch immediately
                    return ch_n
                # "CHAPTER N" alone — caller handles via pending_chapter_num
                return None

    # Strategy 2: level=1 heading — fuzzy match, but only on short single-line blocks.
    # Long or multi-line level=1 blocks are section headers (e.g. "EXERCISES"), not chapters.
    if level == 1 and len(text) <= 100 and "\n" not in text:
        ch = _fuzzy_match_chapter(text, chapter_map, threshold=threshold_l1)
        if ch and ch not in matched_chapters and ch != current_chapter:
            return ch

    # Strategy 3: level=2 heading — fuzzy match with tighter threshold.
    # Only accept short single-line blocks — multi-line or long blocks are
    # unit index pages or paragraph text, not chapter headings.
    if level == 2:
        if len(text) <= 60 and "\n" not in text:
            ch = _fuzzy_match_chapter(text, chapter_map, threshold=threshold_l2)
            if ch and ch not in matched_chapters and ch != current_chapter:
                return ch

    return None


# ── MinerU-based chunker (primary) ───────────────────────────────────────────

def _mineru_chunks(
    content_list_path: Path,
    topic_map: dict,
    chapter_map: dict,
    images_index: dict,
    doc_id: str,
    target: int,
    min_words: int,
    overlap: float,
    skip_pages: set[int] | None = None,
) -> list[dict]:
    """
    Chunk using MinerU's structured content_list.json.
    Uses text_level metadata and fuzzy matching instead of regex.
    """
    blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
    skip = skip_pages or set()

    img_by_page: dict[int, list[str]] = {}
    for img in images_index.get("images", []):
        pg = img["page_number"]
        img_by_page.setdefault(pg, []).append(img["image_id"])

    # ── State ─────────────────────────────────────────────────────────────────
    current_meta: dict | None = None
    current_text = ""
    current_pages: list[int] = []
    current_type = "content"
    all_chunks: list[dict] = []
    chunk_seq = 0

    in_exercises = False
    in_example   = False
    in_answers   = False
    current_chapter: str | None = None

    def flush():
        nonlocal chunk_seq, current_meta, current_text, current_pages, current_type
        if not current_text.strip() or current_meta is None:
            return
        for segment in _split_into_chunks(current_text.strip(), target, min_words, overlap):
            chunk_seq += 1
            imgs = []
            for pg in current_pages:
                imgs.extend(img_by_page.get(pg, []))
            all_chunks.append({
                "chunk_id":       f"{doc_id}_ch{current_meta['chapter_number']}_t{current_meta['topic_number']}_{chunk_seq:04d}",
                "document_id":    doc_id,
                "chapter_number": current_meta["chapter_number"],
                "chapter_name":   current_meta["chapter_name"],
                "topic_number":   current_meta["topic_number"],
                "topic_name":     current_meta["topic_name"],
                "content":        segment,
                "word_count":     len(_words(segment)),
                "page_start":     current_pages[0] if current_pages else None,
                "page_end":       current_pages[-1] if current_pages else None,
                "has_images":     bool(imgs),
                "image_refs":     imgs,
                "chunk_type":     current_type,
                "mode":           "mineru",
            })

    def start_segment(meta: dict, ctype: str, first_text: str, page: int):
        nonlocal current_meta, current_text, current_pages, current_type
        flush()
        current_meta  = meta
        current_text  = first_text
        current_pages = [page]
        current_type  = ctype

    def append_text(text: str, page: int):
        nonlocal current_text, current_pages
        if text.strip():
            current_text += ("\n" if current_text else "") + text
        if page not in current_pages:
            current_pages.append(page)

    # Build a set of already-detected chapter starts to avoid re-triggering
    matched_chapters: set[str] = set()

    # ── Process blocks ────────────────────────────────────────────────────────
    # pending_chapter_num: set when we see "CHAPTER N" as a standalone block.
    # chapter_detections_this_page: track detections per page — if a page fires
    # multiple chapter detections it's a unit index page, not real content.
    # We cancel all detections from such pages (set chapter to None = limbo).
    pending_chapter_num: str | None = None
    _last_chapter_detection_page: int = -1
    _multi_chapter_page: set[int] = set()  # pages where multiple chapters fired

    for block in blocks:
        btype  = block.get("type", "")
        page   = block.get("page_idx", 0) + 1  # 1-based
        text   = block.get("text", "").strip()
        level  = block.get("text_level")

        # Skip non-content page ranges (ToC etc.)
        if page in skip:
            continue

        # Skip running headers, footers, page numbers — layout noise
        if btype in ("header", "footer", "page_number", "page_footnote"):
            continue

        # Images/charts/tables: just track page for image linking
        if btype in ("image", "chart", "table"):
            if current_meta and page not in current_pages:
                current_pages.append(page)
            continue

        # Equations and code — append as-is to current segment
        if btype in ("equation", "code"):
            eq_text = block.get("text") or block.get("code_body") or ""
            if eq_text and current_meta:
                append_text(eq_text, page)
            continue

        if not text:
            continue

        # ── Answer key heading → skip until next chapter ─────────────────────
        if _ANSWERS_BLK_RE.match(text):
            flush()
            in_answers      = True
            in_exercises    = False
            in_example      = False
            pending_chapter_num = None
            continue

        if in_answers:
            # Exit answer-skip mode only when we see a new chapter start
            ch_exit = _detect_chapter_start(text, level, chapter_map, matched_chapters, current_chapter)
            if ch_exit:
                in_answers = False
                # fall through to chapter detection below
            else:
                continue

        # ── Exercise / Intext heading ─────────────────────────────────────────
        # Only activate exercise mode when inside a chapter (prevents front-matter poisoning).
        if (_EXERCISE_BLK_RE.match(text) or _INTEXT_BLK_RE.match(text)) and current_chapter:
            flush()
            in_exercises = True
            in_example   = False
            pending_chapter_num = None
            if current_meta:
                ex_meta = {
                    "chapter_number": current_meta["chapter_number"],
                    "chapter_name":   current_meta["chapter_name"],
                    "topic_number":   f"ex_{current_meta['chapter_number']}",
                    "topic_name":     text,
                }
                current_meta  = ex_meta
                current_text  = ""
                current_pages = [page]
                current_type  = "exercise"
            continue

        # ── Chapter detection ─────────────────────────────────────────────────
        # Strategy (in order):
        #   1. "CHAPTER N" regex → get number directly, no name matching needed.
        #      If "CHAPTER N name" on one block → switch immediately.
        #      If "CHAPTER N" alone → store as pending, resolve on next text block.
        #   2. pending_chapter_num + this block is the name → switch now.
        #   3. level=1 fuzzy match → switch (Physics/Chemistry style).
        #   4. level=2 fuzzy match → switch (Biology style where level=2 is ch heading).

        # Skip chapter detection for topic-numbered headings (N.N ...) or UNIT dividers.
        _is_topic_heading = bool(_TOPIC_NUM_RE.match(text)) or bool(_UNIT_HEADER_RE.match(text))

        if not _is_topic_heading:
            ch_num = _detect_chapter_start(text, level, chapter_map, matched_chapters, current_chapter)
            if ch_num:
                # Guard: if this page already fired a chapter detection, it's a unit index
                # page listing multiple chapters — cancel both and mark page as bad.
                if page == _last_chapter_detection_page:
                    _multi_chapter_page.add(page)
                    # Undo the previous detection from this page
                    if current_chapter in matched_chapters:
                        matched_chapters.discard(current_chapter)
                    current_chapter     = None
                    pending_chapter_num = None
                    matched_chapters.discard(ch_num)
                    log.debug("  [Chunker] pg=%d: multi-chapter page detected, cancelling Ch%s", page, ch_num)
                    continue

                _last_chapter_detection_page = page
                flush()
                matched_chapters.add(ch_num)
                current_chapter     = ch_num
                pending_chapter_num = None
                in_exercises        = False
                in_example          = False
                in_answers          = False
                current_meta        = None
                current_text        = ""
                current_pages       = []
                current_type        = "content"
                continue

        # Resolve a pending "CHAPTER N" standalone block: next non-empty text block
        # is the chapter name — use the pending number directly.
        if pending_chapter_num and pending_chapter_num not in matched_chapters:
            ch_num = pending_chapter_num
            if page in _multi_chapter_page or page == _last_chapter_detection_page:
                # This page already had multiple detections — skip this pending too
                matched_chapters.discard(ch_num)
                pending_chapter_num = None
            else:
                _last_chapter_detection_page = page
                flush()
                matched_chapters.add(ch_num)
                current_chapter     = ch_num
                pending_chapter_num = None
                in_exercises        = False
                in_example          = False
                in_answers          = False
                current_meta        = None
                current_text        = ""
                current_pages       = []
                current_type        = "content"
                continue

        # Detect "CHAPTER N" standalone (number only, single-line block)
        if "\n" not in text:
            m_ch = _CHAPTER_NUM_RE.match(text)
            if m_ch and not text[m_ch.end():].strip():
                ch_n = m_ch.group(1)
                if ch_n in chapter_map and ch_n not in matched_chapters:
                    pending_chapter_num = ch_n
                    continue  # wait for the name on the next block

        # Not a chapter heading — clear pending since a non-empty text block
        # arrived that wasn't the chapter name.
        pending_chapter_num = None

        # ── Topic heading (text_level=2 with N.N pattern) ────────────────────
        if level == 2:
            t_num = _fuzzy_match_topic(text, topic_map)
            if t_num:
                meta = topic_map[t_num]
                new_ch = meta["chapter_number"]
                if new_ch != current_chapter:
                    if new_ch not in matched_chapters:
                        matched_chapters.add(new_ch)
                    current_chapter = new_ch
                    in_exercises    = False
                    in_example      = False
                    in_answers      = False

                if in_exercises:
                    ex_meta = {
                        "chapter_number": meta["chapter_number"],
                        "chapter_name":   meta["chapter_name"],
                        "topic_number":   t_num,
                        "topic_name":     f"Exercise {t_num}",
                    }
                    start_segment(ex_meta, "exercise", text, page)
                else:
                    in_example = False
                    start_segment(meta, "content", text, page)
                continue

            # text_level=2 but not in topic_map — body text
            if not in_answers:
                append_text(text, page)
            continue

        # ── Example heading (no text_level, starts with "Example N") ─────────
        if _EXAMPLE_BLK_RE.match(text) and not in_exercises:
            in_example = True
            if current_meta:
                raw_num = text.split()[1] if len(text.split()) > 1 else "0"
                ex_meta = {
                    "chapter_number": current_meta["chapter_number"],
                    "chapter_name":   current_meta["chapter_name"],
                    "topic_number":   f"ex_{re.sub(r'[^0-9.]', '', raw_num)}",
                    "topic_name":     f"Example {raw_num}",
                }
                start_segment(ex_meta, "example", text, page)
            else:
                append_text(text, page)
            continue

        # ── Exercise questions inside exercise section ─────────────────────────
        if in_exercises and current_meta:
            m = _TOPIC_NUM_RE.match(text)
            if m:
                t_num  = f"{m.group(1)}.{m.group(2)}"
                ch_num = current_chapter or current_meta["chapter_number"]
                ex_meta = {
                    "chapter_number": ch_num,
                    "chapter_name":   current_meta["chapter_name"],
                    "topic_number":   t_num,
                    "topic_name":     f"Exercise {t_num}",
                }
                start_segment(ex_meta, "exercise", text, page)
                continue

        # ── Regular body text ─────────────────────────────────────────────────
        if current_meta and not in_answers:
            append_text(text, page)
        elif not current_meta and not in_answers and current_chapter:
            ch_meta = chapter_map.get(current_chapter)
            if ch_meta:
                stub = dict(ch_meta)
                stub["topic_number"] = f"{current_chapter}.0"
                stub["topic_name"]   = "Introduction"
                current_meta  = stub
                current_text  = text
                current_pages = [page]
                current_type  = "content"

    flush()

    all_chapters_found = set(c["chapter_number"] for c in all_chunks if c.get("chapter_number"))
    missing_chapters   = set(chapter_map.keys()) - all_chapters_found
    if missing_chapters:
        log.warning("  [Chunker] Missing chapters after chunking: %s", sorted(missing_chapters))
    else:
        log.info("  [Chunker] All %d chapters found.", len(chapter_map))

    return all_chunks


# ── Fallback: regex-based txt chunker ────────────────────────────────────────

def _detect_mode(txt_files: list[Path], topic_map: dict, chapter_map: dict) -> str:
    if topic_map:
        sample = txt_files[:30]
        hits = sum(1 for f in sample
                   if _TOPIC_HEADING_RE.search(f.read_text(encoding="utf-8", errors="ignore")))
        return "structured" if hits >= 2 else "chapter"
    if chapter_map:
        return "chapter"
    return "flat"


def _structured_chunks(
    txt_files: list[Path],
    topic_map: dict,
    images_index: dict,
    doc_id: str,
    target: int,
    min_words: int,
    overlap: float,
) -> list[dict]:
    img_by_page: dict[int, list[str]] = {}
    for img in images_index.get("images", []):
        pg = img["page_number"]
        img_by_page.setdefault(pg, []).append(img["image_id"])

    current_meta: dict | None = None
    current_text = ""
    current_pages: list[int] = []
    current_type = "content"
    all_chunks: list[dict] = []
    chunk_seq = 0

    def flush(meta, text, pages, ctype):
        nonlocal chunk_seq
        if not text.strip():
            return
        for segment in _split_into_chunks(text.strip(), target, min_words, overlap):
            chunk_seq += 1
            imgs = []
            for pg in pages:
                imgs.extend(img_by_page.get(pg, []))
            all_chunks.append({
                "chunk_id":       f"{doc_id}_ch{meta['chapter_number']}_t{meta['topic_number']}_{chunk_seq:04d}",
                "document_id":    doc_id,
                "chapter_number": meta["chapter_number"],
                "chapter_name":   meta["chapter_name"],
                "topic_number":   meta["topic_number"],
                "topic_name":     meta["topic_name"],
                "content":        segment,
                "word_count":     len(_words(segment)),
                "page_start":     pages[0] if pages else None,
                "page_end":       pages[-1] if pages else None,
                "has_images":     bool(imgs),
                "image_refs":     imgs,
                "chunk_type":     ctype,
                "mode":           "structured",
            })

    carried_meta: dict | None = None
    in_exercises = False
    in_example   = False
    in_answers   = False
    current_chapter: str | None = None

    for txt_file in sorted(txt_files, key=_page_num):
        pg   = _page_num(txt_file)
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        in_answers = False  # reset at page boundary

        page_segments: list[tuple[dict, str, str]] = []
        seg_meta  = None
        seg_lines: list[str] = []
        seg_type  = "exercise" if in_exercises else ("example" if in_example else "content")

        for line in lines:
            if _ANSWERS_HEADING_RE.match(line):
                if seg_meta and seg_lines:
                    page_segments.append((seg_meta, "\n".join(seg_lines), seg_type))
                    seg_lines = []
                in_answers = True
                seg_meta = None
                continue

            if in_answers:
                continue

            if _EXERCISE_HEADING_RE.match(line) or _INTEXT_HEADING_RE.match(line):
                if seg_meta and seg_lines:
                    page_segments.append((seg_meta, "\n".join(seg_lines), seg_type))
                    seg_lines = []
                in_exercises = True
                in_example   = False
                seg_type     = "exercise"
                seg_meta     = carried_meta
                continue

            em = _EXAMPLE_HEADING_RE.match(line)
            if em and not in_exercises:
                ex_num = f"{em.group(1)}.{em.group(2)}"
                if seg_meta and seg_lines:
                    page_segments.append((seg_meta, "\n".join(seg_lines), seg_type))
                in_example = True
                ch_num = carried_meta["chapter_number"] if carried_meta else em.group(1)
                ch_name = (carried_meta["chapter_name"] if carried_meta else
                           next((v["chapter_name"] for v in topic_map.values()
                                 if v["chapter_number"] == ch_num), ""))
                ex_meta = {
                    "chapter_number": ch_num,
                    "chapter_name":   ch_name,
                    "topic_number":   f"ex{ex_num}",
                    "topic_name":     f"Example {ex_num}",
                }
                seg_meta  = ex_meta
                seg_lines = [line]
                seg_type  = "example"
                continue

            m = _TOPIC_HEADING_RE.match(line)
            if m:
                t_num = f"{m.group(1)}.{m.group(2)}"
                if t_num in topic_map:
                    new_chapter = topic_map[t_num]["chapter_number"]
                    if new_chapter != current_chapter:
                        in_exercises    = False
                        in_example      = False
                        in_answers      = False
                        current_chapter = new_chapter

                    if not in_exercises:
                        if seg_meta and seg_lines:
                            page_segments.append((seg_meta, "\n".join(seg_lines), seg_type))
                        in_example   = False
                        seg_meta     = topic_map[t_num]
                        seg_lines    = [line]
                        carried_meta = seg_meta
                        seg_type     = "content"
                        continue
                    else:
                        if seg_meta and seg_lines:
                            page_segments.append((seg_meta, "\n".join(seg_lines), "exercise"))
                        ch_num  = carried_meta["chapter_number"] if carried_meta else t_num.split(".")[0]
                        ch_name = (carried_meta["chapter_name"] if carried_meta else
                                   next((v["chapter_name"] for v in topic_map.values()
                                         if v["chapter_number"] == ch_num), ""))
                        seg_meta  = {"chapter_number": ch_num, "chapter_name": ch_name,
                                     "topic_number": t_num, "topic_name": f"Exercise {t_num}"}
                        seg_lines = [line]
                        seg_type  = "exercise"
                        continue

            seg_lines.append(line)

        if seg_meta and seg_lines:
            page_segments.append((seg_meta, "\n".join(seg_lines), seg_type))
        elif seg_lines and carried_meta:
            page_segments.append((carried_meta, "\n".join(seg_lines), seg_type))

        for s_meta, s_text, s_ctype in page_segments:
            if current_meta is None:
                current_meta, current_text, current_pages, current_type = s_meta, s_text, [pg], s_ctype
            elif s_meta["topic_number"] == current_meta["topic_number"]:
                current_text += "\n" + s_text
                if pg not in current_pages:
                    current_pages.append(pg)
            else:
                flush(current_meta, current_text, current_pages, current_type)
                current_meta, current_text, current_pages, current_type = s_meta, s_text, [pg], s_ctype

    flush(current_meta, current_text, current_pages, current_type)
    return all_chunks


def _chapter_chunks(
    txt_files: list[Path],
    chapter_map: dict,
    topic_map: dict,
    images_index: dict,
    doc_id: str,
    target: int,
    min_words: int,
    overlap: float,
    skip_pages: set[int] | None = None,
) -> list[dict]:
    img_by_page: dict[int, list[str]] = {}
    for img in images_index.get("images", []):
        pg = img["page_number"]
        img_by_page.setdefault(pg, []).append(img["image_id"])

    _skip = skip_pages or set()
    _content_start = (max(_skip) + 1) if _skip else 1
    _CH_TITLE_RE   = re.compile(r"^chapter\s+\d+\s+(.+)", re.IGNORECASE)

    page_ch_matches: dict[int, list[str]] = {}
    for txt_file in sorted(txt_files, key=_page_num):
        pg = _page_num(txt_file)
        if pg < _content_start:
            continue
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        lines_upper = [ln.strip().upper() for ln in text.splitlines() if ln.strip()]
        lines_upper_set = set(lines_upper)
        matched = []
        for ch_num in chapter_map:
            ch_name_up = chapter_map[ch_num]["chapter_name"].upper()
            if ch_name_up in lines_upper_set:
                matched.append(ch_num)
                continue
            for line in text.splitlines():
                m = _CH_TITLE_RE.match(line.strip())
                if m and m.group(1).strip().upper() == ch_name_up:
                    matched.append(ch_num)
                    break
        if matched:
            page_ch_matches[pg] = matched

    ch_start: dict[str, int] = {}
    for pg, chs in sorted(page_ch_matches.items()):
        if len(chs) == 1:
            ch_num = chs[0]
            if ch_num not in ch_start:
                ch_start[ch_num] = pg
    for pg, chs in sorted(page_ch_matches.items()):
        if len(chs) > 1:
            for ch_num in chs:
                if ch_num not in ch_start:
                    ch_start[ch_num] = pg

    sorted_chapters = sorted(ch_start.items(), key=lambda x: x[1])
    page_to_ch: dict[int, str] = {}
    for i, (ch_num, start_pg) in enumerate(sorted_chapters):
        end_pg = sorted_chapters[i + 1][1] if i + 1 < len(sorted_chapters) else 9999
        for p in range(start_pg, end_pg):
            page_to_ch[p] = ch_num

    ch_texts: dict[str, list[str]] = {}
    ch_pages: dict[str, list[int]] = {}
    for txt_file in sorted(txt_files, key=_page_num):
        pg     = _page_num(txt_file)
        ch_num = page_to_ch.get(pg)
        if not ch_num:
            continue
        ch_texts.setdefault(ch_num, []).append(txt_file.read_text(encoding="utf-8", errors="ignore"))
        ch_pages.setdefault(ch_num, []).append(pg)

    all_chunks, chunk_seq = [], 0
    for ch_num, texts in ch_texts.items():
        meta     = chapter_map[ch_num]
        combined = "\n".join(texts)
        pages    = ch_pages[ch_num]
        for segment in _split_into_chunks(combined, target, min_words, overlap):
            chunk_seq += 1
            imgs = []
            for pg in pages:
                imgs.extend(img_by_page.get(pg, []))
            all_chunks.append({
                "chunk_id":       f"{doc_id}_ch{ch_num}_{chunk_seq:04d}",
                "document_id":    doc_id,
                "chapter_number": ch_num,
                "chapter_name":   meta["chapter_name"],
                "topic_number":   None,
                "topic_name":     None,
                "content":        segment,
                "word_count":     len(_words(segment)),
                "page_start":     pages[0],
                "page_end":       pages[-1],
                "has_images":     bool(imgs),
                "image_refs":     imgs,
                "chunk_type":     "content",
                "mode":           "chapter",
            })
    return all_chunks


def _flat_chunks(
    txt_files: list[Path],
    images_index: dict,
    doc_id: str,
    target: int,
    min_words: int,
    overlap: float,
) -> list[dict]:
    img_by_page: dict[int, list[str]] = {}
    for img in images_index.get("images", []):
        pg = img["page_number"]
        img_by_page.setdefault(pg, []).append(img["image_id"])

    all_chunks, chunk_seq = [], 0
    for txt_file in sorted(txt_files, key=_page_num):
        pg   = _page_num(txt_file)
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        imgs = img_by_page.get(pg, [])
        for segment in _split_into_chunks(text, target, min_words, overlap):
            chunk_seq += 1
            all_chunks.append({
                "chunk_id":       f"{doc_id}_page{pg:04d}_{chunk_seq:04d}",
                "document_id":    doc_id,
                "chapter_number": None,
                "chapter_name":   None,
                "topic_number":   None,
                "topic_name":     None,
                "content":        segment,
                "word_count":     len(_words(segment)),
                "page_start":     pg,
                "page_end":       pg,
                "has_images":     bool(imgs),
                "image_refs":     imgs,
                "chunk_type":     "content",
                "mode":           "flat",
            })
    return all_chunks


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_document(
    txt_files: list[Path],
    toc: dict,
    images_index: dict,
    output_path: Path,
    doc_id: str,
    skip_pages: set[int] | None = None,
    content_list_path: Path | None = None,
) -> Path:
    """
    Main entry point. Uses MinerU content_list when available, else falls back
    to txt-file regex scanning. Writes chunks.json and returns output_path.
    """
    t0 = time.perf_counter()

    if skip_pages:
        txt_files = [f for f in txt_files if _page_num(f) not in skip_pages]
        log.info("  [Chunker] Skipping %d ToC pages.", len(skip_pages))

    topic_map   = _build_topic_map(toc)
    chapter_map = _build_chapter_map(toc)

    if content_list_path and content_list_path.exists():
        log.info("  [Chunker] Mode: mineru | %d topics | %d chapters",
                 len(topic_map), len(chapter_map))
        chunks = _mineru_chunks(
            content_list_path=content_list_path,
            topic_map=topic_map,
            chapter_map=chapter_map,
            images_index=images_index,
            doc_id=doc_id,
            target=CHUNK_TARGET_WORDS,
            min_words=CHUNK_MIN_WORDS,
            overlap=CHUNK_OVERLAP_RATIO,
            skip_pages=skip_pages,
        )
        mode = "mineru"
    else:
        mode = _detect_mode(txt_files, topic_map, chapter_map)
        log.info("  [Chunker] Mode: %s (fallback) | %d pages | %d topics",
                 mode, len(txt_files), len(topic_map))
        if mode == "structured":
            chunks = _structured_chunks(
                txt_files, topic_map, images_index, doc_id,
                CHUNK_TARGET_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_RATIO,
            )
        elif mode == "chapter":
            chunks = _chapter_chunks(
                txt_files, chapter_map, topic_map, images_index, doc_id,
                CHUNK_TARGET_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_RATIO,
                skip_pages=skip_pages,
            )
        else:
            chunks = _flat_chunks(
                txt_files, images_index, doc_id,
                CHUNK_TARGET_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_RATIO,
            )

    duration   = time.perf_counter() - t0
    total_words = sum(c["word_count"] for c in chunks)

    output_path.write_text(
        json.dumps({"total_chunks": len(chunks), "mode": mode, "chunks": chunks},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("  [Chunker] Done: %d chunks | %d total words | %.2fs",
             len(chunks), total_words, duration)
    log.info("  [Chunker] Saved -> %s", output_path.name)
    return output_path
