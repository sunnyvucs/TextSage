"""Test the chat logic directly without HTTP."""
import os, json
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(".env"))

import psycopg2, psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config import QDRANT_HOST, QDRANT_PORT
from src.embedder import get_embedder

CHUNK_COLLECTION = "textbook_chunks"
TOP_K = 3

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST","localhost"),
        port=int(os.getenv("POSTGRES_PORT","5432")),
        dbname=os.getenv("POSTGRES_DB","al_learning"),
        user=os.getenv("POSTGRES_USER","postgres"),
        password=os.getenv("POSTGRES_PASSWORD",""),
        connect_timeout=10,
    )

question = "What is photosynthesis?"
subject  = "Biology"
class_number = "12"

print("1. Embedding question...")
embedder = get_embedder()
qv = embedder.embed_one(question)
print("   dim:", len(qv))

print("2. Searching Qdrant...")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
sr = qdrant.query_points(
    collection_name=CHUNK_COLLECTION,
    query=qv,
    query_filter=Filter(must=[
        FieldCondition(key="subject", match=MatchValue(value=subject)),
        FieldCondition(key="class_number", match=MatchValue(value=class_number)),
    ]),
    limit=TOP_K,
    with_payload=True,
)
results = sr.points
print(f"   {len(results)} results")
for r in results:
    print(f"   score={r.score:.3f} pg_chunk_id={r.payload.get('pg_chunk_id')}")

print("3. Fetching chunks from PostgreSQL...")
chunk_ids = [str(r.payload["pg_chunk_id"]) for r in results]
conn = get_conn()
with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute("SELECT id, chapter_name, topic_name, page_number, content FROM textbook_chunks WHERE id = ANY(%s::UUID[])", (chunk_ids,))
    chunks = cur.fetchall()
    print(f"   {len(chunks)} chunks fetched")
    for c in chunks:
        print(f"   page={c['page_number']} topic={c['topic_name']} content_len={len(c['content'] or '')}")

print("4. Fetching images...")
book_ids = list({str(c['id']) for c in chunks})  # wrong — should be book_id
# correct:
with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute("SELECT id, book_id, page_number, content FROM textbook_chunks WHERE id = ANY(%s::UUID[])", (chunk_ids,))
    full_chunks = cur.fetchall()
    b_ids = list({str(c['book_id']) for c in full_chunks})
    p_nums = list({c['page_number'] for c in full_chunks})
    cur.execute(
        "SELECT book_id, page_number, image_path, caption FROM textbook_images WHERE book_id = ANY(%s::UUID[]) AND page_number = ANY(%s::INT[])",
        (b_ids, p_nums)
    )
    imgs = cur.fetchall()
    print(f"   {len(imgs)} images found")

conn.close()
print("ALL OK")
