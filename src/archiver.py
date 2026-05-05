"""
archiver.py
Phase 8: Archive all PDFInprogress assets into PostgreSQL and clean the working folder.

What it does per book:
  1. PDF bytes        -> textbook_books.pdf_data
  2. Image bytes      -> textbook_images.image_data
  3. fig_number       -> textbook_images.fig_number  (parsed from caption e.g. "Figure 1.2")
  4. JSON files       -> textbook_json_files  (manifest, toc, chapter_page_map, chunks)
  5. Delete PDFInprogress/<book_stem>/ after verified write

Run via:
  python main.py --archive [--book STEM]
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from src.config import INPROGRESS_DIR

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)
_SEP = "-" * 60

_FIG_RE = re.compile(
    r"(?:Figure|Fig\.?)\s*(\d+[\.\d]*(?:\([a-z]\))?)",
    re.IGNORECASE,
)

_JSON_FILES = [
    "manifest.json",
    "toc.json",
    "chapter_page_map.json",
    "chunks.json",
]


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "al_learning"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


def _extract_fig_number(caption: str | None) -> str | None:
    if not caption:
        return None
    m = _FIG_RE.search(caption)
    return m.group(1) if m else None


def archive_book(work_dir: Path) -> dict:
    """
    Archive one book's assets from PDFInprogress into PostgreSQL,
    then delete the working directory.
    """
    book_stem = work_dir.name
    log.info(_SEP)
    log.info("ARCHIVE  %s", book_stem)
    log.info(_SEP)

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM textbook_books WHERE book_stem = %s", (book_stem,))
            row = cur.fetchone()
        if not row:
            log.error("  [Archiver] %s not found in textbook_books — run --write-db first.", book_stem)
            return {"book": book_stem, "status": "error", "error": "book not in DB"}
        book_id = str(row["id"])

        # ── 1. PDF bytes ──────────────────────────────────────────────────────
        # Only store the original whole-book PDF if present at work_dir root.
        # Per-page splits (pdfs/page_*.pdf) are intermediate artifacts — skip them.
        pdf_files = [p for p in work_dir.glob("*.pdf") if not p.name.startswith("page_")]

        pdf_written = 0
        if pdf_files:
            pdf_path = pdf_files[0]
            log.info("  PDF: %s (%.1f MB)", pdf_path.name, pdf_path.stat().st_size / 1_048_576)
            pdf_bytes = pdf_path.read_bytes()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE textbook_books SET pdf_data = %s, updated_at = NOW() WHERE id = %s",
                        (psycopg2.Binary(pdf_bytes), book_id),
                    )
            pdf_written = 1
            log.info("  PDF stored in DB.")
        else:
            log.info("  No whole-book PDF at root of %s — skipping pdf_data (text already in DB).", work_dir)

        # ── 2. Images: bytes + fig_number ─────────────────────────────────────
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, image_path, caption FROM textbook_images WHERE book_id = %s",
                (book_id,),
            )
            image_rows = cur.fetchall()

        img_written = 0
        img_missing = 0
        updates = []
        for img in image_rows:
            img_path = work_dir / img["image_path"]
            if not img_path.exists():
                log.debug("  Image missing on disk: %s", img["image_path"])
                img_missing += 1
                continue
            img_bytes = img_path.read_bytes()
            fig_num = _extract_fig_number(img["caption"])
            updates.append((psycopg2.Binary(img_bytes), fig_num, str(img["id"])))

        if updates:
            with conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        """UPDATE textbook_images
                           SET image_data = data.img_bytes,
                               fig_number = data.fig_num
                           FROM (VALUES %s) AS data(img_bytes, fig_num, img_id)
                           WHERE id = data.img_id::UUID""",
                        updates,
                        template="(%s, %s, %s)",
                        page_size=50,
                    )
            img_written = len(updates)
        log.info("  Images stored: %d  missing: %d", img_written, img_missing)

        # ── 3. JSON files ─────────────────────────────────────────────────────
        json_written = 0
        for fname in _JSON_FILES:
            fpath = work_dir / fname
            if not fpath.exists():
                continue
            try:
                content = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("  Could not parse %s: %s", fname, exc)
                continue
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO textbook_json_files (book_id, book_stem, file_name, content)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (book_id, file_name) DO UPDATE
                               SET content = EXCLUDED.content""",
                        (book_id, book_stem, fname, json.dumps(content)),
                    )
            json_written += 1
            log.info("  Stored JSON: %s", fname)

        # ── 4. Verify then delete working directory ───────────────────────────
        # Sanity check: book row still readable
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM textbook_books WHERE id = %s", (book_id,))
            assert cur.fetchone(), "book row missing after archive"

        shutil.rmtree(work_dir)
        log.info("  Deleted working directory: %s", work_dir)

        result = {
            "book": book_stem,
            "status": "success",
            "pdf_written": pdf_written,
            "images_written": img_written,
            "images_missing": img_missing,
            "json_files_written": json_written,
        }
        log.info("  ARCHIVE DONE  %s", result)
        return result

    except Exception as exc:
        log.error("  [Archiver] %s failed: %s", book_stem, exc, exc_info=True)
        return {"book": book_stem, "status": "error", "error": str(exc)}
    finally:
        conn.close()


def archive_all_books() -> list[dict]:
    book_dirs = sorted(
        d for d in INPROGRESS_DIR.iterdir()
        if d.is_dir()
    )
    log.info(_SEP)
    log.info("ARCHIVE ALL  (%d books)", len(book_dirs))
    log.info(_SEP)

    results = []
    for book_dir in book_dirs:
        result = archive_book(book_dir)
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "success")
    log.info(_SEP)
    log.info("ARCHIVE ALL DONE  ok=%d / %d", ok, len(results))
    log.info(_SEP)

    # If all books archived, remove the now-empty PDFInprogress folder itself
    remaining = [d for d in INPROGRESS_DIR.iterdir()] if INPROGRESS_DIR.exists() else []
    if not remaining:
        INPROGRESS_DIR.rmdir()
        log.info("PDFInprogress/ removed (empty).")

    return results
