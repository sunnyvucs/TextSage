"""
chunk_embedder.py
Phase 7: Embed textbook_chunks content into Qdrant and write qdrant_point_id back to PostgreSQL.

Collection: textbook_chunks  (separate from textbook_pages page-level collection)
Each point:
  id      : UUID (same as textbook_chunks.id)
  vector  : BGE-small embedding of chunk content
  payload : book_stem, subject, class_number, part,
            chapter_number, chapter_name, topic_number, topic_name,
            page_number, book_id, pg_id (== qdrant point id as str)

Run via:
  python main.py --embed-chunks [--book STEM]
"""

import logging
import os
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import QDRANT_HOST, QDRANT_PORT, EMBED_MODEL
from src.embedder import get_embedder

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

CHUNK_COLLECTION = "textbook_chunks"
_BATCH_SIZE = 64
_MIN_CHARS = 20


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "al_learning"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


def _ensure_chunk_collection(client: QdrantClient, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if CHUNK_COLLECTION in existing:
        info = client.get_collection(CHUNK_COLLECTION)
        existing_dim = info.config.params.vectors.size
        if existing_dim == dim:
            log.info("  [ChunkEmbedder] Qdrant collection '%s' exists (dim=%d)", CHUNK_COLLECTION, dim)
            return
        log.warning("  [ChunkEmbedder] Dimension mismatch (%d vs %d) — recreating collection.", existing_dim, dim)
        client.delete_collection(CHUNK_COLLECTION)

    client.create_collection(
        collection_name=CHUNK_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("  [ChunkEmbedder] Created Qdrant collection '%s' (dim=%d)", CHUNK_COLLECTION, dim)


def embed_chunks(book_stem: str | None = None) -> dict:
    """
    Embed all chunks (or one book's chunks) from PostgreSQL into Qdrant.
    Writes qdrant_point_id back to each row in textbook_chunks.
    Skips chunks with already-populated qdrant_point_id (incremental safe).
    """
    embedder = get_embedder()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_chunk_collection(qdrant, embedder._embed_dim)

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if book_stem:
                cur.execute(
                    "SELECT * FROM textbook_chunks WHERE book_stem = %s AND qdrant_point_id IS NULL ORDER BY page_number",
                    (book_stem,),
                )
            else:
                cur.execute(
                    "SELECT * FROM textbook_chunks WHERE qdrant_point_id IS NULL ORDER BY book_stem, page_number"
                )
            rows = cur.fetchall()

        if not rows:
            log.info("  [ChunkEmbedder] No unembedded chunks found%s.", f" for {book_stem}" if book_stem else "")
            return {"status": "success", "embedded": 0, "skipped": 0}

        log.info("  [ChunkEmbedder] %d chunks to embed%s.", len(rows), f" for {book_stem}" if book_stem else "")

        embedded = 0
        skipped = 0
        updates: list[tuple[str, str]] = []  # (qdrant_point_id, pg_chunk_id)

        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            texts = []
            valid_batch = []

            for row in batch:
                text = (row["content"] or "").strip()
                if len(text) < _MIN_CHARS:
                    skipped += 1
                    continue
                texts.append(text)
                valid_batch.append(row)

            if not texts:
                continue

            vectors = embedder.embed(texts)

            points = []
            for row, vector in zip(valid_batch, vectors):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "pg_chunk_id":    str(row["id"]),
                        "book_id":        str(row["book_id"]),
                        "book_stem":      row["book_stem"],
                        "class_number":   row["class_number"],
                        "subject":        row["subject"],
                        "part":           row["part"],
                        "chapter_number": row["chapter_number"],
                        "chapter_name":   row["chapter_name"],
                        "topic_number":   row["topic_number"],
                        "topic_name":     row["topic_name"],
                        "page_number":    row["page_number"],
                        "text_preview":   (row["content"] or "")[:200],
                    },
                ))
                updates.append((point_id, str(row["id"])))

            qdrant.upsert(collection_name=CHUNK_COLLECTION, points=points)
            embedded += len(points)
            log.info("  [ChunkEmbedder] Upserted batch %d-%d (%d points)", i + 1, i + len(batch), len(points))

        # Write qdrant_point_id back to PostgreSQL
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    "UPDATE textbook_chunks SET qdrant_point_id = data.qid::UUID FROM (VALUES %s) AS data(qid, cid) WHERE id = data.cid::UUID",
                    updates,
                    page_size=500,
                )
        log.info("  [ChunkEmbedder] Wrote %d qdrant_point_ids back to PostgreSQL.", len(updates))

        return {"status": "success", "embedded": embedded, "skipped": skipped}

    except Exception as exc:
        log.error("  [ChunkEmbedder] Failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}
    finally:
        conn.close()


def embed_all_chunks() -> dict:
    log.info("------------------------------------------------------------")
    log.info("CHUNK EMBEDDING")
    log.info("------------------------------------------------------------")
    result = embed_chunks(book_stem=None)
    log.info("------------------------------------------------------------")
    log.info(
        "CHUNK EMBEDDING DONE  --  embedded=%d  skipped=%d",
        result.get("embedded", 0), result.get("skipped", 0),
    )
    log.info("------------------------------------------------------------")
    return result
