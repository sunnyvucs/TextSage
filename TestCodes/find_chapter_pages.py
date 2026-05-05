"""
find_chapter_pages.py
Search all extracted page text files for a chapter name, list every page
where that text is found, and show the Qdrant embedding confidence score
for each page against the chapter query.

Usage:
    python find_chapter_pages.py
"""

import json
import sys
from pathlib import Path

import numpy as np

INPROGRESS_DIR = Path(r"D:\AI\AI Projects\PDFParserAI\PDFInprogress")

# Confidence thresholds for display
_HIGH   = 0.70
_MEDIUM = 0.55


def confidence_label(score: float) -> str:
    if score >= _HIGH:
        return "HIGH"
    if score >= _MEDIUM:
        return "MEDIUM"
    return "LOW"


def find_books_by_subject(subject_query: str) -> list[Path]:
    matches = []
    for book_dir in sorted(INPROGRESS_DIR.iterdir()):
        manifest = book_dir / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        subject = (data.get("subject") or "").lower()
        if subject_query.lower() in subject:
            matches.append(book_dir)
    return matches


def get_qdrant_scores(book_stem: str, query: str) -> dict[int, float]:
    """Return {page_number: similarity_score} for all pages in the book."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.embedder import get_embedder
    from src.config import QDRANT_COLLECTION

    embedder = get_embedder()
    client   = embedder.get_qdrant()
    qv       = np.array(embedder.embed_one(query))

    results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter={"must": [{"key": "book_stem", "match": {"value": book_stem}}]},
        with_vectors=True,
        limit=500,
    )

    return {
        p.payload["page_number"]: float(np.dot(qv, np.array(p.vector)))
        for p in results
    }


def search_pages(book_dir: Path, chapter_query: str) -> list[tuple[int, str]]:
    """Return (page_number, matched_line) for every page containing chapter_query."""
    txt_dir     = book_dir / "txts"
    query_lower = chapter_query.lower()
    results     = []

    for txt_file in sorted(txt_dir.glob("page_*.txt")):
        try:
            pg = int(txt_file.stem.replace("page_", ""))
        except ValueError:
            continue
        for line in txt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if query_lower in line.lower():
                results.append((pg, line.strip()))
                break

    return results


def main():
    print("=" * 70)
    print("  Chapter Page Finder  (with Qdrant confidence scores)")
    print("=" * 70)

    subject_input = input("\nEnter subject (e.g. Biology, Chemistry, Mathametics, Physics): ").strip()
    if not subject_input:
        print("No subject entered. Exiting.")
        sys.exit(1)

    chapter_input = input("Enter chapter name (or part of it): ").strip()
    if not chapter_input:
        print("No chapter name entered. Exiting.")
        sys.exit(1)

    books = find_books_by_subject(subject_input)
    if not books:
        print(f"\nNo books found for subject '{subject_input}'.")
        print("Available books:")
        for d in sorted(INPROGRESS_DIR.iterdir()):
            if (d / "manifest.json").exists():
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                print(f"  {d.name}  (subject: {m.get('subject', 'unknown')})")
        sys.exit(1)

    print(f"\nLoading embeddings and searching {len(books)} book(s) for '{chapter_input}'...\n")

    total_found = 0
    for book_dir in books:
        text_matches = search_pages(book_dir, chapter_input)
        print(f"Book: {book_dir.name}")

        if not text_matches:
            print("  No pages found containing that text.\n")
            continue

        # Get Qdrant scores for all matched pages
        print(f"  Fetching confidence scores from Qdrant...")
        try:
            scores = get_qdrant_scores(book_dir.name, chapter_input)
        except Exception as e:
            print(f"  Warning: could not fetch Qdrant scores ({e})")
            scores = {}

        print(f"\n  {'Page':>6}  {'Score':>7}  {'Confidence':>10}  Matched text")
        print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*40}")

        for pg, line in text_matches:
            score = scores.get(pg, None)
            if score is not None:
                label = confidence_label(score)
                score_str = f"{score:.4f}"
            else:
                label = "N/A"
                score_str = "   N/A"
            print(f"  {pg:>6}  {score_str:>7}  {label:>10}  {line[:60]}")

        # Also show top-5 Qdrant pages even if text not found there
        if scores:
            top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
            matched_pages = {pg for pg, _ in text_matches}
            extra = [(pg, s) for pg, s in top5 if pg not in matched_pages]
            if extra:
                print(f"\n  Top Qdrant matches NOT in text search:")
                print(f"  {'Page':>6}  {'Score':>7}  {'Confidence':>10}  First line of page")
                print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*40}")
                for pg, score in extra:
                    txt = book_dir / "txts" / f"page_{pg:04d}.txt"
                    first_line = txt.read_text(encoding="utf-8", errors="ignore").splitlines()[0][:60] if txt.exists() else "(no text)"
                    print(f"  {pg:>6}  {score:.4f}  {confidence_label(score):>10}  {first_line}")

        total_found += len(text_matches)
        print()

    print(f"Total pages found across all books: {total_found}")
    print(f"\nConfidence guide:  HIGH >= {_HIGH}   MEDIUM >= {_MEDIUM}   LOW < {_MEDIUM}")


if __name__ == "__main__":
    main()
