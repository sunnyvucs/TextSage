import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT_DIR = Path(r"D:\AI\AI Projects\PDFParserAI")
INPROGRESS_DIR = ROOT_DIR / "PDFInprogress"
FINAL_OUTPUT_DIR = ROOT_DIR / "FinalOutput"
CONFIG_DIR = ROOT_DIR / "config"

LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()  # "groq" | "ollama"

TEXT_EXTRACTOR = os.getenv("TEXT_EXTRACTOR", "pymupdf").lower()  # fallback/default only
PYMUPDF_SUBJECTS = {
    "english",
    "hindi",
    "sanskrit",
    "history",
    "geography",
    "political science",
    "sociology",
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-e4b-local")
OLLAMA_TIMEOUT = 120

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "textbook_pages")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "pdfparser")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

TOC_KEYWORDS_FILE = CONFIG_DIR / "toc_keywords.txt"
TOC_SEARCH_MAX_PAGE = 30
TOC_PAGES_TO_SEND = 6

CHUNK_TARGET_WORDS = 400
CHUNK_MIN_WORDS = 80
CHUNK_OVERLAP_RATIO = 0.15

MAX_WORKERS = 1 if TEXT_EXTRACTOR == "mineru" else 4
TEXT_EXTRACT_WORKERS = 8

for _d in [INPROGRESS_DIR, FINAL_OUTPUT_DIR, CONFIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
