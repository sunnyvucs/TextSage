"""
embedder.py
Embedding with Redis cache and Qdrant storage.
Model is configured via EMBED_MODEL in .env (default: BAAI/bge-base-en-v1.5).

Public API:
  get_embedder()          -> Embedder singleton
  Embedder.embed(texts)   -> list of float vectors
  Embedder.embed_one(text)-> single float vector
"""

import hashlib
import json
import logging
from typing import Optional

import redis
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import (
    EMBED_MODEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    REDIS_HOST, REDIS_PORT, REDIS_DB,
)

log = logging.getLogger(__name__)

_REDIS_TTL = 60 * 60 * 24 * 7   # 7 days


class Embedder:
    def __init__(self):
        log.info("  [Embedder] Loading model: %s", EMBED_MODEL)
        self._model = SentenceTransformer(EMBED_MODEL)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(self._model, "get_sentence_embedding_dimension")
        self._embed_dim = get_dim()
        log.info("  [Embedder] Model loaded. Embedding dim: %d", self._embed_dim)
        self._redis: Optional[redis.Redis] = None
        self._redis_checked = False   # True after first connection attempt
        self._qdrant: Optional[QdrantClient] = None

    # ── Redis cache ───────────────────────────────────────────────────────────

    def _get_redis(self) -> Optional[redis.Redis]:
        if not self._redis_checked:
            self._redis_checked = True
            try:
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_timeout=2)
                r.ping()
                self._redis = r
                log.info("  [Embedder] Redis connected at %s:%s", REDIS_HOST, REDIS_PORT)
            except Exception as exc:
                log.warning("  [Embedder] Redis unavailable (%s) — cache disabled.", exc)
        return self._redis

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{EMBED_MODEL}:{text}".encode()).hexdigest()
        return f"emb:{h}"

    def _cache_get(self, text: str) -> Optional[list]:
        r = self._get_redis()
        if r is None:
            return None
        try:
            raw = r.get(self._cache_key(text))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _cache_set(self, text: str, vector: list) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.setex(self._cache_key(text), _REDIS_TTL, json.dumps(vector))
        except Exception:
            pass

    # ── Qdrant ────────────────────────────────────────────────────────────────

    def get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self._ensure_collection()
            log.info("  [Embedder] Qdrant connected at %s:%s", QDRANT_HOST, QDRANT_PORT)
        return self._qdrant

    def _ensure_collection(self) -> None:
        client = self._qdrant
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION in existing:
            # Check if existing collection has matching vector dimension
            info = client.get_collection(QDRANT_COLLECTION)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self._embed_dim:
                log.warning(
                    "  [Embedder] Collection '%s' has dim=%d but model needs dim=%d — "
                    "recreating collection (all vectors will be re-embedded).",
                    QDRANT_COLLECTION, existing_dim, self._embed_dim,
                )
                client.delete_collection(QDRANT_COLLECTION)
                existing = []
            else:
                log.info("  [Embedder] Qdrant collection exists: %s (dim=%d)", QDRANT_COLLECTION, existing_dim)
                return

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=self._embed_dim, distance=Distance.COSINE),
        )
        log.info("  [Embedder] Created Qdrant collection: %s (dim=%d)", QDRANT_COLLECTION, self._embed_dim)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_one(self, text: str) -> list[float]:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        self._cache_set(text, vector)
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            vectors = self._model.encode(
                uncached_texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
            for idx, text, vector in zip(uncached_indices, uncached_texts, vectors):
                results[idx] = vector
                self._cache_set(text, vector)

        return results

    def upsert_points(self, points: list[PointStruct]) -> None:
        """Upsert a batch of PointStructs into Qdrant."""
        client = self.get_qdrant()
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)


_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
