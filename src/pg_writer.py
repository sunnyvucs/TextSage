"""
pg_writer.py
Phase 6: Write chunks.json + image data into PostgreSQL.

Tables created (if not exist):
  textbook_books   — one row per book
  textbook_chunks  — one row per (topic x page) chunk
  textbook_images  — one row per image linked to a chunk's page

Run via:
  python main.py --write-db [--book STEM]
"""

import json
import logging
import os
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from src.config import INPROGRESS_DIR

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS textbook_books (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_stem       TEXT NOT NULL UNIQUE,
    class_number    TEXT,
    subject         TEXT,
    part            TEXT,
    total_chunks    INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS textbook_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id             UUID NOT NULL REFERENCES textbook_books(id) ON DELETE CASCADE,
    book_stem           TEXT NOT NULL,
    class_number        TEXT,
    subject             TEXT,
    part                TEXT,
    chapter_number      TEXT,
    chapter_name        TEXT,
    topic_number        TEXT,
    topic_name          TEXT,
    page_number         INTEGER NOT NULL,
    content             TEXT,
    qdrant_point_id     UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tc_book_id       ON textbook_chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_tc_book_stem     ON textbook_chunks(book_stem);
CREATE INDEX IF NOT EXISTS idx_tc_subject       ON textbook_chunks(subject);
CREATE INDEX IF NOT EXISTS idx_tc_chapter       ON textbook_chunks(book_id, chapter_number);
CREATE INDEX IF NOT EXISTS idx_tc_topic         ON textbook_chunks(book_id, topic_number);
CREATE INDEX IF NOT EXISTS idx_tc_page          ON textbook_chunks(book_id, page_number);
CREATE INDEX IF NOT EXISTS idx_tc_qdrant        ON textbook_chunks(qdrant_point_id);

CREATE TABLE IF NOT EXISTS textbook_images (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id         UUID NOT NULL REFERENCES textbook_books(id) ON DELETE CASCADE,
    book_stem       TEXT NOT NULL,
    page_number     INTEGER NOT NULL,
    image_path      TEXT NOT NULL,
    caption         TEXT,
    description     TEXT,
    bbox            JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ti_book_page ON textbook_images(book_id, page_number);
"""


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "al_learning"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


def ensure_schema():
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
        log.info("  [PgWriter] Schema ready.")
    finally:
        conn.close()


def _load_images(work_dir: Path) -> dict[int, list[dict]]:
    """
    Load images from MinerU content_list.json grouped by 1-based page number.
    Returns {} if no content_list exists (PyMuPDF books like English).
    """
    stem = work_dir.name
    cl_path = work_dir / "mineru_output" / stem / "txt" / f"{stem}_content_list.json"
    if not cl_path.exists():
        return {}

    try:
        blocks = json.loads(cl_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("  [PgWriter] Could not read content_list: %s", exc)
        return {}

    images_by_page: dict[int, list[dict]] = {}
    img_base = cl_path.parent  # .../mineru_output/<stem>/txt/

    for block in blocks:
        if block.get("type") != "image":
            continue
        page_num = block.get("page_idx", 0) + 1  # convert 0-based to 1-based
        img_path = block.get("img_path", "")
        # Store path relative to PDFInprogress/<stem>/
        abs_img = img_base / img_path
        rel_img = str(abs_img.relative_to(work_dir)) if abs_img.exists() else img_path

        captions = block.get("image_caption", [])
        caption = captions[0] if captions else None

        images_by_page.setdefault(page_num, []).append({
            "image_path": rel_img,
            "caption": caption,
            "description": None,
            "bbox": block.get("bbox"),
        })

    return images_by_page


def write_book(work_dir: Path) -> dict:
    """
    Write one book's chunks + images into PostgreSQL.
    Deletes existing rows for this book first (idempotent).
    """
    book_stem = work_dir.name
    chunks_path = work_dir / "chunks.json"

    if not chunks_path.exists():
        log.warning("  [PgWriter] %s — chunks.json missing, skipping.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no chunks.json"}

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        log.warning("  [PgWriter] %s — chunks.json is empty.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "empty chunks"}

    images_by_page = _load_images(work_dir)
    total_images = sum(len(v) for v in images_by_page.values())

    # Pull metadata from first chunk
    first = chunks[0]
    class_number = first.get("class_number")
    subject = first.get("subject")
    part = first.get("part")

    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Upsert book row
                cur.execute("""
                    INSERT INTO textbook_books (book_stem, class_number, subject, part, total_chunks)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (book_stem) DO UPDATE SET
                        class_number = EXCLUDED.class_number,
                        subject      = EXCLUDED.subject,
                        part         = EXCLUDED.part,
                        total_chunks = EXCLUDED.total_chunks,
                        updated_at   = NOW()
                    RETURNING id;
                """, (book_stem, class_number, subject, part, len(chunks)))
                book_id = cur.fetchone()[0]

                # Delete existing chunks + images for this book (re-run safe)
                cur.execute("DELETE FROM textbook_images WHERE book_id = %s;", (book_id,))
                cur.execute("DELETE FROM textbook_chunks WHERE book_id = %s;", (book_id,))

                # Bulk insert chunks
                chunk_rows = [
                    (
                        str(uuid.uuid4()),
                        str(book_id),
                        book_stem,
                        c.get("class_number"),
                        c.get("subject"),
                        c.get("part"),
                        str(c.get("chapter_number")) if c.get("chapter_number") else None,
                        c.get("chapter_name"),
                        c.get("topic_number"),
                        c.get("topic_name"),
                        c.get("page_number"),
                        c.get("text") or "",
                        None,  # qdrant_point_id — filled later by embedder
                    )
                    for c in chunks
                ]
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO textbook_chunks
                       (id, book_id, book_stem, class_number, subject, part,
                        chapter_number, chapter_name, topic_number, topic_name,
                        page_number, content, qdrant_point_id)
                       VALUES %s""",
                    chunk_rows,
                    page_size=500,
                )

                # Bulk insert images
                if images_by_page:
                    image_rows = []
                    for page_num, imgs in images_by_page.items():
                        for img in imgs:
                            image_rows.append((
                                str(uuid.uuid4()),
                                str(book_id),
                                book_stem,
                                page_num,
                                img["image_path"],
                                img["caption"],
                                img["description"],
                                json.dumps(img["bbox"]) if img["bbox"] else None,
                            ))
                    psycopg2.extras.execute_values(
                        cur,
                        """INSERT INTO textbook_images
                           (id, book_id, book_stem, page_number, image_path,
                            caption, description, bbox)
                           VALUES %s""",
                        image_rows,
                        page_size=500,
                    )

        log.info(
            "  [PgWriter] %s - %d chunks, %d images written (book_id=%s)",
            book_stem, len(chunks), total_images, book_id,
        )
        return {
            "book": book_stem,
            "status": "success",
            "chunks": len(chunks),
            "images": total_images,
            "book_id": str(book_id),
        }

    except Exception as exc:
        log.error("  [PgWriter] %s failed: %s", book_stem, exc, exc_info=True)
        return {"book": book_stem, "status": "error", "error": str(exc)}
    finally:
        conn.close()


def write_all_books() -> list[dict]:
    book_dirs = sorted(
        d for d in INPROGRESS_DIR.iterdir()
        if d.is_dir() and (d / "chunks.json").exists()
    )

    log.info("------------------------------------------------------------")
    log.info("DB WRITE  (%d books)", len(book_dirs))
    log.info("------------------------------------------------------------")

    ensure_schema()
    results = []
    for book_dir in book_dirs:
        try:
            result = write_book(book_dir)
        except Exception as exc:
            log.error("  [PgWriter] %s failed: %s", book_dir.name, exc, exc_info=True)
            result = {"book": book_dir.name, "status": "error", "error": str(exc)}
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "success")
    total_chunks = sum(r.get("chunks", 0) for r in results)
    total_images = sum(r.get("images", 0) for r in results)
    log.info("------------------------------------------------------------")
    log.info(
        "DB WRITE DONE  --  ok=%d  total_chunks=%d  total_images=%d",
        ok, total_chunks, total_images,
    )
    log.info("------------------------------------------------------------")
    return results
