"""
chunker.py
Phase 5: Produce one chunk per (topic x page) pair for every book.

Strategy:
  - One chunk per page per topic that covers that page.
  - A boundary page (where topic N ends and topic N+1 starts) produces 2 chunks —
    one tagged with topic N metadata, one tagged with topic N+1 metadata,
    both carrying the full page text.
  - If a chapter has no topics, each page produces exactly 1 chunk tagged at
    chapter level only (topic_number=None, topic_name=None).

Topic PDF page mapping:
  - chapter_page_map.json gives confirmed_start_page and toc_start_page per chapter.
  - offset = confirmed_start_page - toc_chapter_start_page
  - topic_pdf_page = topic_toc_page + offset
  - A topic covers pages [topic_pdf_start, next_topic_pdf_start] inclusive
    (boundary page belongs to both).

Output: PDFInprogress/<book>/chunks.json
  List of chunk dicts, one per (topic x page) pair.
"""

import json
import logging
from pathlib import Path

from src.config import INPROGRESS_DIR

log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_page_text(txt_dir: Path, page_number: int) -> str:
    txt = txt_dir / f"page_{page_number:04d}.txt"
    if not txt.exists():
        return ""
    return txt.read_text(encoding="utf-8", errors="ignore").strip()


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _build_topic_ranges(
    topics: list[dict],
    chapter_confirmed_start: int,
    chapter_toc_start: int | None,
    chapter_end: int,
) -> list[dict]:
    """
    Map TOC topics to PDF page ranges.

    Returns list of:
      {topic_number, topic_name, pdf_start, pdf_end}

    pdf_end is inclusive and equals the next topic's pdf_start (boundary page
    belongs to both this topic and the next).
    """
    if not topics:
        return []

    # Compute per-chapter offset: how many pages to add to printed page numbers
    # to get PDF page indices.
    if chapter_toc_start is not None and chapter_toc_start > 0:
        offset = chapter_confirmed_start - chapter_toc_start
    else:
        offset = 0

    mapped = []
    for t in topics:
        toc_pg = _safe_int(t.get("start_page"))
        if toc_pg is None:
            continue
        pdf_pg = toc_pg + offset
        # Clamp to chapter bounds
        pdf_pg = max(chapter_confirmed_start, min(chapter_end, pdf_pg))
        mapped.append({
            "topic_number": t.get("topic_number"),
            "topic_name": (t.get("topic_name") or "").strip(),
            "pdf_start": pdf_pg,
        })

    if not mapped:
        return []

    # Sort by pdf_start
    mapped.sort(key=lambda x: x["pdf_start"])

    # Assign pdf_end: each topic ends where the next begins (boundary page shared)
    ranges = []
    for i, t in enumerate(mapped):
        if i + 1 < len(mapped):
            pdf_end = mapped[i + 1]["pdf_start"]  # inclusive boundary
        else:
            pdf_end = chapter_end
        ranges.append({
            "topic_number": t["topic_number"],
            "topic_name": t["topic_name"],
            "pdf_start": t["pdf_start"],
            "pdf_end": pdf_end,
        })

    return ranges


def chunk_book(work_dir: Path) -> dict:
    """
    Produce chunks.json for one book.
    Returns summary dict.
    """
    book_stem = work_dir.name
    map_path = work_dir / "chapter_page_map.json"
    toc_path = work_dir / "toc.json"
    txt_dir = work_dir / "txts"

    if not map_path.exists():
        log.warning("  [Chunker] %s — chapter_page_map.json missing, skipping.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no chapter_page_map.json"}

    if not toc_path.exists():
        log.warning("  [Chunker] %s — toc.json missing, skipping.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no toc.json"}

    chapter_map = _load_json(map_path)
    toc = _load_json(toc_path)

    manifest = {}
    if (work_dir / "manifest.json").exists():
        manifest = _load_json(work_dir / "manifest.json")

    subject = chapter_map.get("subject") or manifest.get("subject", "")
    class_number = chapter_map.get("class_number") or manifest.get("class_number", "")
    part = chapter_map.get("part") or manifest.get("part")

    # Build toc chapter lookup by chapter_name for topic retrieval
    toc_chapters: dict[str, list[dict]] = {}
    for tc in toc.get("chapters", []):
        name = (tc.get("chapter_name") or "").strip()
        toc_chapters[name] = tc.get("topics", [])

    aligned_chapters = chapter_map.get("chapters", [])
    if not aligned_chapters:
        log.warning("  [Chunker] %s — no aligned chapters.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no chapters"}

    chunks = []
    total_boundary_pages = 0

    for ch in aligned_chapters:
        ch_num = ch.get("chapter_number")
        ch_name = (ch.get("chapter_name") or "").strip()
        ch_start = _safe_int(ch.get("confirmed_start_page"))
        ch_end = _safe_int(ch.get("end_page"))
        ch_toc_start = _safe_int(ch.get("toc_start_page"))

        if ch_start is None or ch_end is None:
            log.warning("  [Chunker] %s Ch %s — missing page bounds, skipping.", book_stem, ch_num)
            continue

        # Get topics from toc.json for this chapter
        raw_topics = toc_chapters.get(ch_name, [])

        topic_ranges = _build_topic_ranges(
            raw_topics, ch_start, ch_toc_start, ch_end
        )

        log.debug(
            "  [Chunker] %s Ch %s '%s' pages %d-%d — %d topics",
            book_stem, ch_num, ch_name, ch_start, ch_end, len(topic_ranges),
        )

        for page_num in range(ch_start, ch_end + 1):
            text = _read_page_text(txt_dir, page_num)

            if not topic_ranges:
                # Chapter-level chunk only
                chunks.append({
                    "book_stem": book_stem,
                    "class_number": class_number,
                    "subject": subject,
                    "part": part,
                    "chapter_number": ch_num,
                    "chapter_name": ch_name,
                    "topic_number": None,
                    "topic_name": None,
                    "page_number": page_num,
                    "text": text,
                })
                continue

            # Find all topics that cover this page
            covering = [
                t for t in topic_ranges
                if t["pdf_start"] <= page_num <= t["pdf_end"]
            ]

            if not covering:
                # Fallback: assign to the last topic before this page
                before = [t for t in topic_ranges if t["pdf_start"] <= page_num]
                covering = [before[-1]] if before else [topic_ranges[0]]

            if len(covering) > 1:
                total_boundary_pages += 1
                log.debug(
                    "    page %d is a boundary page — %d topics: %s",
                    page_num,
                    len(covering),
                    [t["topic_number"] for t in covering],
                )

            for topic in covering:
                chunks.append({
                    "book_stem": book_stem,
                    "class_number": class_number,
                    "subject": subject,
                    "part": part,
                    "chapter_number": ch_num,
                    "chapter_name": ch_name,
                    "topic_number": topic["topic_number"],
                    "topic_name": topic["topic_name"],
                    "page_number": page_num,
                    "text": text,
                })

    out_path = work_dir / "chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "  [Chunker] %s - %d chunks from %d chapters (%d boundary pages) -> %s",
        book_stem, len(chunks), len(aligned_chapters), total_boundary_pages, out_path.name,
    )

    return {
        "book": book_stem,
        "status": "success",
        "chunks": len(chunks),
        "boundary_pages": total_boundary_pages,
    }


def chunk_all_books() -> list[dict]:
    book_dirs = sorted(
        d for d in INPROGRESS_DIR.iterdir()
        if d.is_dir()
        and (d / "chapter_page_map.json").exists()
        and (d / "toc.json").exists()
    )

    log.info("------------------------------------------------------------")
    log.info("CHUNKING  (%d books)", len(book_dirs))
    log.info("------------------------------------------------------------")

    results = []
    for book_dir in book_dirs:
        try:
            result = chunk_book(book_dir)
        except Exception as exc:
            log.error("  [Chunker] %s failed: %s", book_dir.name, exc, exc_info=True)
            result = {"book": book_dir.name, "status": "error", "error": str(exc)}
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    total_chunks = sum(r.get("chunks", 0) for r in results)

    log.info("------------------------------------------------------------")
    log.info(
        "CHUNKING DONE  --  ok=%d  skipped=%d  errors=%d  total_chunks=%d",
        ok, skipped, errors, total_chunks,
    )
    log.info("------------------------------------------------------------")
    return results
