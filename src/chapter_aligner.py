"""
chapter_aligner.py
Phase 3: Find the true PDF page where each chapter begins.

Approach:
  Stage 1 - Qdrant similarity search using "Chapter N ChapterName" to get
            candidate pages across the full book.
  Stage 2 - Structural scoring reranks those candidates so chapter-opening
            pages beat mid-chapter exercise/summary pages.
  Stage 3 - The LLM is used only when the top structural candidate is not
            clearly better than the next-best option.

Output: PDFInprogress/<book>/chapter_page_map.json
"""

import json
import logging
import re
from pathlib import Path

import numpy as np

from src.chapter_start_features import score_page
from src.config import INPROGRESS_DIR, QDRANT_COLLECTION
from src.embedder import get_embedder
from src.llm_client import chat

log = logging.getLogger(__name__)

_TOP_K = 15
_TOC_WINDOW = 30
_MIN_SCORE = 0.25
_VECTOR_WEIGHT = 0.5
_BM25_WEIGHT = 0.5
_STRUCTURAL_WEIGHT = 0.85
_LLM_MARGIN = 0.18


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bm25_scores(query: str, txt_dir: Path, page_numbers: list[int]) -> dict[int, float]:
    """
    Compute BM25 keyword scores for query against all pages in txt_dir.
    Returns {page_number: normalized_score} where scores are in [0, 1].
    """
    from rank_bm25 import BM25Okapi

    query_tokens = query.lower().split()
    corpus = []
    pages = []

    for pg in page_numbers:
        txt = txt_dir / f"page_{pg:04d}.txt"
        if txt.exists():
            text = txt.read_text(encoding="utf-8", errors="ignore").lower()
            corpus.append(text.split())
            pages.append(pg)

    if not corpus:
        return {}

    bm25 = BM25Okapi(corpus)
    raw = bm25.get_scores(query_tokens)
    max_s = max(raw) if max(raw) > 0 else 1.0
    return {pg: float(s / max_s) for pg, s in zip(pages, raw)}


def _candidates(
    embedder,
    book_stem: str,
    query: str,
    total_pages: int,
    toc_start_page: int = 0,
    prev_confirmed_page: int = 0,
    txt_dir: Path | None = None,
) -> list[tuple[float, int]]:
    """
    Return up to _TOP_K (score, page_number) pairs using hybrid search:
      - vector similarity from Qdrant
      - BM25 keyword score from page text files

    Also force-includes a window of pages from the best available anchor so
    sparse heading pages are never missed even when their scores are low.
    """
    query_vector = embedder.embed_one(query)
    client = embedder.get_qdrant()

    results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter={"must": [{"key": "book_stem", "match": {"value": book_stem}}]},
        with_vectors=True,
        limit=total_pages + 2,
    )

    if not results:
        return []

    qv = np.array(query_vector)
    vec_scores = {}
    for point in results:
        pg = point.payload["page_number"]
        vec_scores[pg] = float(np.dot(qv, np.array(point.vector)))

    max_vec = max(vec_scores.values()) if vec_scores else 1.0
    vec_norm = {pg: s / max_vec for pg, s in vec_scores.items()}

    bm25_norm = {}
    if txt_dir and txt_dir.exists():
        bm25_norm = _bm25_scores(query, txt_dir, list(vec_scores.keys()))

    hybrid = {}
    for pg in vec_scores:
        v = vec_norm.get(pg, 0.0)
        b = bm25_norm.get(pg, 0.0)
        hybrid[pg] = _VECTOR_WEIGHT * v + _BM25_WEIGHT * b

    top_k = sorted(hybrid.items(), key=lambda x: -x[1])[:_TOP_K]
    candidate_pages = set(pg for pg, _ in top_k)

    if prev_confirmed_page > 0:
        window_start = prev_confirmed_page + 1
        window_end = min(total_pages, prev_confirmed_page + _TOC_WINDOW)
        log.debug(
            "      window anchor=prev_confirmed(%d), pages %d-%d",
            prev_confirmed_page, window_start, window_end,
        )
    elif toc_start_page > 0 and toc_start_page <= total_pages and toc_start_page <= (total_pages * 0.8):
        window_start = max(1, toc_start_page - 5)
        window_end = min(total_pages, toc_start_page + _TOC_WINDOW)
        log.debug(
            "      window anchor=toc_start(%d), pages %d-%d",
            toc_start_page, window_start, window_end,
        )
    else:
        window_start = 1
        window_end = min(total_pages, _TOC_WINDOW)
        log.debug("      window anchor=book_start, pages %d-%d", window_start, window_end)

    for pg in range(window_start, window_end + 1):
        if pg in hybrid:
            candidate_pages.add(pg)

    result = [(hybrid[pg], pg) for pg in candidate_pages if pg in hybrid]
    result.sort(key=lambda x: -x[0])
    return result


def _llm_pick(
    ch_num: str,
    ch_name: str,
    candidates: list[tuple[float, int]],
    txt_dir: Path,
    must_be_after: int = 0,
) -> int:
    """
    Ask the configured LLM which candidate page is the actual chapter start.
    Returns the chosen page number, or highest-scoring valid candidate on failure.
    """
    pages_sorted = sorted(set(p for _, p in candidates))
    valid_pages = [p for p in pages_sorted if p > must_be_after] if must_be_after > 0 else pages_sorted

    if not valid_pages:
        log.warning(
            "      Ch %s  no candidates after page %d - using first page after constraint.",
            ch_num, must_be_after,
        )
        all_pages = sorted(set(p for _, p in candidates))
        valid_pages = [p for p in all_pages if p > must_be_after]
        if not valid_pages:
            return candidates[0][1]

    max_pages_to_llm = 8
    snippet_chars = 300
    scored_valid = [(s, p) for s, p in candidates if p in valid_pages]
    scored_valid.sort(key=lambda x: -x[0])
    top_valid_pages = sorted(set(p for _, p in scored_valid[:max_pages_to_llm]))

    snippets = []
    for pg in top_valid_pages:
        txt = txt_dir / f"page_{pg:04d}.txt"
        if txt.exists():
            text = txt.read_text(encoding="utf-8", errors="ignore").strip()[:snippet_chars]
        else:
            text = "(no text)"
        snippets.append(f"PAGE {pg}:\n{text}")

    context = "\n\n---\n\n".join(snippets)
    after_constraint = (
        f"IMPORTANT: The previous chapter ends at page {must_be_after}, "
        f"so the answer MUST be a page number greater than {must_be_after}.\n\n"
        if must_be_after > 0 else ""
    )

    prompt = (
        f'I am looking for the START page of Chapter {ch_num}: "{ch_name}" in a school textbook.\n\n'
        f"The START page is the FIRST page of the chapter - the one that has the chapter "
        f"title or number as a large heading, OR a motivational quote followed by an "
        f"introduction, OR learning objectives ('After studying this unit...'). "
        f"It is always BEFORE any section content, exercises, or worked examples.\n\n"
        f"REJECT any page that starts with: EXERCISE, Example, Solution, Table of Contents, "
        f"Foreword, Preface, Acknowledgement, Summary, or any numbered problem list. "
        f"Those are mid-chapter or front-matter pages, NOT the chapter start.\n\n"
        f"{after_constraint}"
        f"{context}\n\n"
        f'Which page number is the START of Chapter {ch_num}: "{ch_name}"?\n'
        f"Reply with ONLY the page number as a single integer, nothing else."
    )

    try:
        raw = chat(prompt)
        m = re.search(r"\d+", raw)
        if m:
            picked = int(m.group())
            if picked in top_valid_pages:
                return picked
            if picked in valid_pages:
                return picked
            log.warning("      LLM returned page %d not in candidates - using best valid match.", picked)
    except Exception as exc:
        log.warning("      LLM call failed: %s - using best valid match.", exc)

    for score, pg in candidates:
        if pg in valid_pages:
            return pg
    return valid_pages[0]


def _rerank_candidates(
    candidates: list[tuple[float, int]],
    txt_dir: Path,
    ch_num: str,
    ch_name: str,
) -> list[tuple[float, int, dict[str, float]]]:
    reranked = []
    for hybrid_score, pg in candidates:
        txt_path = txt_dir / f"page_{pg:04d}.txt"
        structural = score_page(txt_path, ch_num, ch_name) if txt_path.exists() else None
        structural_score = structural.score if structural else 0.0
        total = hybrid_score + (_STRUCTURAL_WEIGHT * structural_score)
        signals = dict(structural.signals) if structural else {}
        signals["hybrid_score"] = round(hybrid_score, 4)
        signals["total_score"] = round(total, 4)
        reranked.append((total, pg, signals))
    reranked.sort(key=lambda x: -x[0])
    return reranked


def align_book(work_dir: Path) -> dict:
    """
    Align chapters for one book. Writes chapter_page_map.json into work_dir.
    """
    book_stem = work_dir.name
    toc_path = work_dir / "toc.json"
    manifest = _load_json(work_dir / "manifest.json")
    total_pages = len(manifest.get("pages", []))
    txt_dir = work_dir / "txts"

    if not toc_path.exists():
        log.warning("  [Aligner] %s - toc.json missing, skipping.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no toc.json"}

    toc = _load_json(toc_path)
    chapters = toc.get("chapters", [])
    if not chapters:
        log.warning("  [Aligner] %s - toc.json has no chapters.", book_stem)
        return {"book": book_stem, "status": "skipped", "reason": "no chapters"}

    def _to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    valid_starts = [(c, _to_int(c["start_page"])) for c in chapters if c.get("start_page") is not None]
    valid_starts = [(c, s) for c, s in valid_starts if s is not None]  # drop Roman numerals / non-numeric
    in_range = [(c, s) for c, s in valid_starts if s <= total_pages]
    out_of_range = [(c, s) for c, s in valid_starts if s > total_pages]
    if in_range and out_of_range:
        min_oor = min(s for _, s in out_of_range)
        page_offset = min_oor - 1
        min_in_range = min(s for _, s in in_range)
        is_companion = min_in_range <= (total_pages // 4)
        if is_companion:
            log.warning(
                "  [Aligner] %s - dropping %d companion-volume in-range chapters "
                "(min start_page=%d); applying offset -%d to %d out-of-range chapters",
                book_stem, len(in_range), min_in_range, page_offset, len(out_of_range),
            )
            chapters = [c for c, _ in out_of_range]
        else:
            log.info(
                "  [Aligner] %s - keeping %d genuine in-range chapters (min start_page=%d); "
                "applying offset -%d to %d out-of-range chapters",
                book_stem, len(in_range), min_in_range, page_offset, len(out_of_range),
            )
            chapters = [c for c, _ in in_range] + [c for c, _ in out_of_range]
        for c, s in out_of_range:
            c["start_page"] = s - page_offset

    log.info("  [Aligner] %s - aligning %d chapters  (total_pages=%d)", book_stem, len(chapters), total_pages)
    embedder = get_embedder()
    aligned = []
    prev_confirmed_page = 0

    _SKIP_CHAPTER_NAMES = (
        "foreword", "preface", "acknowledgement", "acknowledgements",
        "a note for the teacher", "note for the teacher", "rationalisation",
        "rationalisation of content", "textbook development committee",
        "the constitution of india",
    )

    for ch in chapters:
        ch_num = ch.get("chapter_number", "?")
        ch_name = (ch.get("chapter_name") or "").strip()
        if not ch_name:
            log.warning("    Ch %s  chapter_name is null/empty - skipping.", ch_num)
            continue

        if ch_name.lower() in _SKIP_CHAPTER_NAMES:
            log.info("    Ch %s  skipping front-matter chapter '%s'.", ch_num, ch_name)
            continue

        ch_num_str = str(ch_num).strip()
        if ch_num_str.isdigit():
            query = f"Chapter {ch_num_str} {ch_name}"
        else:
            query = f"CHAPTER {ch_num_str} {ch_name}"

        try:
            toc_pg = int(ch.get("start_page") or 0)
        except (ValueError, TypeError):
            toc_pg = 0  # Roman numerals or non-numeric TOC page — treat as unknown
        candidates = _candidates(
            embedder,
            book_stem,
            query,
            total_pages,
            toc_start_page=toc_pg,
            prev_confirmed_page=prev_confirmed_page,
            txt_dir=txt_dir,
        )

        if not candidates:
            log.warning("    Ch %s  no Qdrant results - skipping.", ch_num)
            continue

        top_score = candidates[0][0]
        if top_score < _MIN_SCORE:
            log.warning("    Ch %s  low top score=%.3f - LLM may struggle.", ch_num, top_score)

        reranked = _rerank_candidates(candidates, txt_dir, ch_num_str, ch_name)
        if not reranked:
            log.warning("    Ch %s  no reranked candidates - skipping.", ch_num)
            continue

        best_score, best_page, best_signals = reranked[0]
        second_score = reranked[1][0] if len(reranked) > 1 else -999.0
        score_gap = best_score - second_score
        filtered_candidates = [(score, pg) for score, pg, _ in reranked]

        if score_gap >= _LLM_MARGIN and best_signals.get("negative_marker", 0.0) == 0.0:
            confirmed = best_page
            log.info(
                "    Ch %s  structural pick=%d  total=%.3f  gap=%.3f  %s",
                ch_num, confirmed, best_score, score_gap, ch_name,
            )
        else:
            confirmed = _llm_pick(ch_num_str, ch_name, filtered_candidates, txt_dir)
            log.info(
                "    Ch %s  llm pick=%d  total=%.3f  gap=%.3f  %s",
                ch_num, confirmed, best_score, score_gap, ch_name,
            )

        prev_confirmed_page = confirmed
        aligned.append({
            "chapter_number": ch_num,
            "chapter_name": ch_name,
            "toc_start_page": ch.get("start_page"),
            "confirmed_start_page": confirmed,
            "end_page": None,
            "confidence": round(best_score, 3),
            "signals": best_signals,
            "_candidates": filtered_candidates,
        })

    if not aligned:
        return {"book": book_stem, "status": "skipped", "reason": "no alignable chapters"}

    for i in range(1, len(aligned)):
        prev_page = aligned[i - 1]["confirmed_start_page"]
        curr_page = aligned[i]["confirmed_start_page"]
        if curr_page <= prev_page:
            ch = aligned[i]
            log.warning(
                "    Ch %s  confirmed=%d <= prev_confirmed=%d - re-asking LLM with constraint >%d",
                ch["chapter_number"], curr_page, prev_page, prev_page,
            )
            corrected = _llm_pick(
                str(ch["chapter_number"]),
                ch["chapter_name"],
                aligned[i]["_candidates"],
                txt_dir,
                must_be_after=prev_page,
            )
            log.warning("    Ch %s  corrected=%d", ch["chapter_number"], corrected)
            aligned[i]["confirmed_start_page"] = corrected

    for i, entry in enumerate(aligned):
        if i + 1 < len(aligned):
            entry["end_page"] = aligned[i + 1]["confirmed_start_page"] - 1
        else:
            entry["end_page"] = total_pages

    for entry in aligned:
        entry.pop("_candidates", None)

    out = {
        "book_stem": book_stem,
        "subject": manifest.get("subject"),
        "class_number": manifest.get("class_number"),
        "part": manifest.get("part"),
        "chapters": aligned,
    }

    out_path = work_dir / "chapter_page_map.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("  [Aligner] %s - written %s", book_stem, out_path.name)

    return {"book": book_stem, "status": "success", "chapters_aligned": len(aligned)}


def align_all_books() -> list[dict]:
    """Align chapters for all books in PDFInprogress/ that have toc.json + manifest.json."""
    book_dirs = sorted(
        d for d in INPROGRESS_DIR.iterdir()
        if d.is_dir() and (d / "manifest.json").exists() and (d / "toc.json").exists()
    )

    log.info("------------------------------------------------------------")
    log.info("CHAPTER ALIGNMENT  (%d books)", len(book_dirs))
    log.info("------------------------------------------------------------")

    results = []
    for book_dir in book_dirs:
        try:
            result = align_book(book_dir)
        except Exception as exc:
            log.error("  [Aligner] %s failed: %s", book_dir.name, exc, exc_info=True)
            result = {"book": book_dir.name, "status": "error", "error": str(exc)}
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    log.info("------------------------------------------------------------")
    log.info("ALIGNMENT DONE  --  ok=%d  skipped=%d  errors=%d", ok, skipped, errors)
    log.info("------------------------------------------------------------")
    return results
