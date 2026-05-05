"""
api_server.py
FastAPI backend for the AITutor chat UI.

Endpoints:
  GET  /api/classes          -> list of available classes
  GET  /api/subjects?class=  -> subjects for a class
  POST /api/chat             -> RAG query + Ollama streaming response
  GET  /api/images/{book_stem}/{path:path} -> serve image files

Run:
  .\venv\Scripts\python.exe api_server.py
"""

import json
import logging
import os
import re
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest

from src.config import QDRANT_HOST, QDRANT_PORT, INPROGRESS_DIR
from src.embedder import get_embedder

# Matches "Fig. 1.2", "Figure 1.2(a)", "fig 2", etc.
_FIG_RE = re.compile(r"(?:Fig\.?|Figure)\s*(\d+[\.\d]*(?:\([a-z]\))?)", re.IGNORECASE)

load_dotenv(Path(".env"))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="AITutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b-local")
CHUNK_COLLECTION = "textbook_chunks"
TOP_K = 5
MAX_CHARS_PER_CHUNK = 1000  # truncate to keep total prompt within local model RAM


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "al_learning"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        connect_timeout=10,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/classes")
def get_classes():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT class_number FROM textbook_books WHERE class_number IS NOT NULL ORDER BY class_number")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


@app.get("/api/subjects")
def get_subjects(class_number: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT subject FROM textbook_books WHERE class_number = %s AND subject IS NOT NULL ORDER BY subject",
                (class_number,),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


class ChatRequest(BaseModel):
    question: str
    class_number: str
    subject: str
    history: list[dict] = []  # [{role, content}, ...]


def _build_search_query(question: str, history: list[dict]) -> str:
    """
    For short follow-up questions ("explain in detail", "give example", "what is that"),
    prepend the last user question from history so Qdrant searches on the actual topic.
    """
    q = question.strip()
    # Heuristic: if question is short and has no real topic noun, it's a follow-up
    if len(q.split()) <= 5 and history:
        last_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"),
            None,
        )
        if last_user:
            return f"{last_user} {q}"
    return q


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 1. Embed the question (with follow-up context if needed)
    embedder = get_embedder()
    search_query = _build_search_query(req.question, req.history)
    query_vector = embedder.embed_one(search_query)

    # 2. Search Qdrant with subject filter
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    search_result = qdrant.query_points(
        collection_name=CHUNK_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="subject", match=MatchValue(value=req.subject)),
                FieldCondition(key="class_number", match=MatchValue(value=req.class_number)),
            ]
        ),
        limit=TOP_K,
        with_payload=True,
    )
    results = search_result.points

    if not results:
        raise HTTPException(status_code=404, detail="No relevant content found for this subject.")

    # 3. Fetch full chunks from PostgreSQL
    chunk_ids = [str(r.payload["pg_chunk_id"]) for r in results]
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM textbook_chunks WHERE id = ANY(%s::UUID[])",
                (chunk_ids,),
            )
            chunks = {str(r["id"]): r for r in cur.fetchall()}

            # Extract all figure references mentioned in the matched chunk texts
            # e.g. "Fig. 1.2", "Figure 2.7(a)" -> ["1.2", "2.7"]
            fig_numbers: set[str] = set()
            for r in results:
                chunk = chunks.get(str(r.payload["pg_chunk_id"]))
                if chunk:
                    for m in _FIG_RE.finditer(chunk["content"] or ""):
                        # Normalize: strip sub-part like (a) for the lookup so "1.2" matches "Figure 1.2(a)"
                        fig_numbers.add(re.sub(r"\([a-z]\)$", "", m.group(1), flags=re.IGNORECASE))

            # Fetch images whose fig_number matches any referenced figure
            # Also fall back to exact page match for images with no caption (sub-parts like (a),(b))
            book_id_set = list({str(chunks[cid]["book_id"]) for cid in chunks})
            images_by_id: dict[str, dict] = {}

            if book_id_set and fig_numbers:
                cur.execute(
                    """SELECT id, book_id, book_stem, page_number, caption, fig_number,
                              image_data IS NOT NULL AS has_data
                       FROM textbook_images
                       WHERE book_id = ANY(%s::UUID[])
                         AND fig_number = ANY(%s::TEXT[])""",
                    (book_id_set, list(fig_numbers)),
                )
                for img in cur.fetchall():
                    images_by_id[str(img["id"])] = dict(img)

                # Also fetch sub-part images on the same pages:
                # captions like (a),(b),(i),(ii) etc. are parts of the same figure
                matched_pages = list({img["page_number"] for img in images_by_id.values()})
                if matched_pages:
                    cur.execute(
                        """SELECT id, book_id, book_stem, page_number, caption, fig_number,
                                  image_data IS NOT NULL AS has_data
                           FROM textbook_images
                           WHERE book_id = ANY(%s::UUID[])
                             AND page_number = ANY(%s::INT[])
                             AND (caption IS NULL OR caption ~ '^\\([a-z]+\\)$')""",
                        (book_id_set, matched_pages),
                    )
                    for img in cur.fetchall():
                        images_by_id[str(img["id"])] = dict(img)

            # If no fig refs found in text, fall back to top-1 chunk's page images
            if not images_by_id and results:
                top = chunks.get(str(results[0].payload["pg_chunk_id"]))
                if top:
                    cur.execute(
                        """SELECT id, book_id, book_stem, page_number, caption, fig_number,
                                  image_data IS NOT NULL AS has_data
                           FROM textbook_images
                           WHERE book_id = %s AND page_number = %s""",
                        (str(top["book_id"]), top["page_number"]),
                    )
                    for img in cur.fetchall():
                        images_by_id[str(img["id"])] = dict(img)

    finally:
        conn.close()

    # 4. Build context for LLM (ordered by score) + collect images
    context_blocks = []
    for r in results:
        chunk = chunks.get(str(r.payload["pg_chunk_id"]))
        if not chunk:
            continue
        ch_label = f"Chapter {chunk['chapter_number']}: {chunk['chapter_name']}" if chunk["chapter_number"] else chunk["chapter_name"] or ""
        topic_label = f"Topic {chunk['topic_number']}: {chunk['topic_name']}" if chunk["topic_number"] else ""
        header = " | ".join(filter(None, [ch_label, topic_label, f"Page {chunk['page_number']}"]))
        content = (chunk['content'] or "")[:MAX_CHARS_PER_CHUNK]
        context_blocks.append(f"[{header}]\n{content}")

    # Build image refs using DB UUIDs — frontend fetches /api/images/<id>
    image_refs = [
        {
            "id": img_id,
            "page_number": img["page_number"],
            "caption": img["caption"],
            "fig_number": img["fig_number"],
            "has_data": img["has_data"],
        }
        for img_id, img in images_by_id.items()
    ]

    context_text = "\n\n---\n\n".join(context_blocks)

    # 5. Build prompt
    # Strategy: embed the textbook passages directly in the user turn so the model
    # treats answering from them as the immediate task, not an abstract instruction.
    # Small local models (gemma) ignore system-prompt rules but do follow user-turn context.

    system_prompt = (
        f"You are a Class {req.class_number} {req.subject} tutor. "
        "Your only job is to explain the question using the textbook passages given to you by the student. "
        "Do not add any information that is not present in the passages."
    )

    # Build the user turn: passages first, then the question
    user_turn = (
        f"Here are the relevant passages from the Class {req.class_number} {req.subject} textbook:\n\n"
        f"{context_text}\n\n"
        "---\n"
        "Using ONLY the passages above (do not use outside knowledge), answer this question:\n"
        f"{req.question}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_turn})

    # 6. Stream from Ollama
    async def stream_response():
        # First yield image refs as a special JSON line
        if image_refs:
            yield f"data: {json.dumps({'images': image_refs})}\n\n"

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        err = json.loads(body).get("error", "Ollama error")
                        yield f"data: {json.dumps({'error': err})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if data.get("done"):
                                yield f"data: {json.dumps({'done': True})}\n\n"
                        except Exception:
                            continue
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# Serve images from PostgreSQL (primary) with filesystem fallback for pre-archive books
@app.get("/api/images/{image_id}")
def serve_image(image_id: str):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT image_data, image_path, book_stem FROM textbook_images WHERE id = %s",
                (image_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    image_data, image_path, book_stem = row

    # Primary: serve from DB bytes
    if image_data:
        ext = (image_path or "").rsplit(".", 1)[-1].lower()
        media = "image/png" if ext == "png" else "image/jpeg"
        return Response(content=bytes(image_data), media_type=media)

    # Fallback: serve from filesystem (books not yet archived)
    if image_path and book_stem:
        fs_path = INPROGRESS_DIR / book_stem / image_path
        if fs_path.exists():
            return FileResponse(str(fs_path))

    raise HTTPException(status_code=404, detail="Image data not available")


# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
