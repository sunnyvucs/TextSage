import psycopg2, psycopg2.extras, os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(".env"))

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST","localhost"), port=5432,
    dbname=os.getenv("POSTGRES_DB","al_learning"),
    user=os.getenv("POSTGRES_USER","postgres"),
    password=os.getenv("POSTGRES_PASSWORD","")
)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

print("=== Physics chunks pages 15-25 ===")
cur.execute("""SELECT page_number, topic_number, topic_name, LEFT(content,120) as preview
               FROM textbook_chunks WHERE book_stem='Physics_Part_I'
               AND page_number BETWEEN 15 AND 25 ORDER BY page_number""")
for r in cur.fetchall():
    print(f"  p={r['page_number']} topic={r['topic_number']} | {r['topic_name']} | {r['preview'][:80]}")

print()
print("=== Qdrant search results for 'electric charge' (Physics) ===")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.embedder import get_embedder

embedder = get_embedder()
qv = embedder.embed_one("explain electric charge")
qdrant = QdrantClient(host="localhost", port=6333)
sr = qdrant.query_points(
    collection_name="textbook_chunks",
    query=qv,
    query_filter=Filter(must=[
        FieldCondition(key="subject", match=MatchValue(value="Physics")),
        FieldCondition(key="class_number", match=MatchValue(value="12")),
    ]),
    limit=5, with_payload=True,
)
for r in sr.points:
    print(f"  score={r.score:.3f} page={r.payload.get('page_number')} topic={r.payload.get('topic_number')} | {r.payload.get('topic_name')}")

print()
print("=== Images for Physics pages 15-25 ===")
cur.execute("""SELECT page_number, image_path, caption FROM textbook_images
               WHERE book_stem='Physics_Part_I' AND page_number BETWEEN 15 AND 25
               ORDER BY page_number""")
for r in cur.fetchall():
    print(f"  p={r['page_number']} | {r['caption'] or '(no caption)'} | {r['image_path'][-50:]}")

conn.close()
