"""
page_embedder.py
Embeds all pages for a processed book and stores them in Qdrant.

Each Qdrant point represents one page:
  id      : deterministic int from book_stem + page_number
  vector  : BGE-small embedding of the page text
  payload : subject, class_number, part, book_stem, page_number, txt_file
"""

import hashlib
import json
import logging
from pathlib import Path

from qdrant_client.models import PointStruct

from src.embedder import get_embedder
from src.config import INPROGRESS_DIR

log = logging.getLogger(__name__)
_BATCH_SIZE = 32
_MIN_CHARS  = 50    # skip near-empty pages


def _point_id(book_stem: str, page_number: int) -> int:
    """Deterministic stable int ID from book stem + page number."""
    raw = f"{book_stem}:{page_number}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:15], 16)


def _load_manifest(work_dir: Path) -> dict:
    p = work_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def embed_book(work_dir: Path) -> dict:
    """
    Embed all pages of one book and upsert into Qdrant.
    Returns a result dict with status, book, pages_embedded, pages_skipped.
    """
    book_stem = work_dir.name
    manifest  = _load_manifest(work_dir)
    txt_dir   = work_dir / "txts"

    subject      = manifest.get("subject", "Unknown")
    class_number = manifest.get("class_number", "Unknown")
    part         = manifest.get("part")
    pages_meta   = manifest.get("pages", [])

    log.info("  [PageEmbedder] %s — %d pages", book_stem, len(pages_meta))

    embedder = get_embedder()
    embedder.get_qdrant()   # ensure collection exists

    points_batch: list[PointStruct] = []
    embedded = 0
    skipped  = 0

    for meta in pages_meta:
        page_num = meta["page"]
        txt_path = txt_dir / meta["txt"]

        if not txt_path.exists():
            log.warning("    Page %d: txt missing — skipping.", page_num)
            skipped += 1
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < _MIN_CHARS:
            log.debug("    Page %d: too short (%d chars) — skipping.", page_num, len(text))
            skipped += 1
            continue

        point_id = _point_id(book_stem, page_num)
        vector   = embedder.embed_one(text)

        points_batch.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "book_stem":    book_stem,
                "subject":      subject,
                "class_number": class_number,
                "part":         part,
                "page_number":  page_num,
                "txt_file":     meta["txt"],
                "text_preview": text[:200],
            },
        ))
        embedded += 1

        if len(points_batch) >= _BATCH_SIZE:
            embedder.upsert_points(points_batch)
            log.info("    Upserted batch of %d (page %d)", len(points_batch), page_num)
            points_batch = []

    if points_batch:
        embedder.upsert_points(points_batch)
        log.info("    Upserted final batch of %d", len(points_batch))

    log.info("  [PageEmbedder] %s done — embedded=%d  skipped=%d", book_stem, embedded, skipped)
    return {
        "book":           book_stem,
        "status":         "success",
        "pages_embedded": embedded,
        "pages_skipped":  skipped,
    }


def embed_all_books() -> list[dict]:
    """Embed all books in PDFInprogress/ that have a manifest."""
    book_dirs = sorted(d for d in INPROGRESS_DIR.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    log.info("------------------------------------------------------------")
    log.info("PAGE EMBEDDING  (%d books)", len(book_dirs))
    log.info("------------------------------------------------------------")

    results = []
    for book_dir in book_dirs:
        try:
            result = embed_book(book_dir)
        except Exception as exc:
            log.error("  [PageEmbedder] %s failed: %s", book_dir.name, exc)
            result = {"book": book_dir.name, "status": "error", "error": str(exc)}
        results.append(result)

    ok     = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    total  = sum(r.get("pages_embedded", 0) for r in results)
    log.info("------------------------------------------------------------")
    log.info("EMBEDDING DONE  --  ok=%d  errors=%d  total_pages=%d", ok, errors, total)
    log.info("------------------------------------------------------------")
    return results
