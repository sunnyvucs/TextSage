"""
main.py
Phase 1: Extract PDFs one at a time.

  python main.py <folder>

Runs MinerU extraction, subject identification, and ToC parsing.
Output written to PDFInprogress/<stem>/ for each PDF.

Chunking, alignment, and vector storage are separate pipeline stages.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
from src.resume_check import check_and_prompt, ResumeChoice
from src.pipeline import extract_pdf, retoc_book
from src.manifest_validator import validate_all_manifests
from src.page_embedder import embed_all_books
from src.chapter_aligner import align_all_books
from src.chunker import chunk_all_books, chunk_book
from src.pg_writer import write_all_books, write_book
from src.chunk_embedder import embed_all_chunks, embed_chunks
from src.archiver import archive_all_books, archive_book

setup_logging()
log = logging.getLogger(__name__)

_SEP  = "═" * 60
_SEP2 = "─" * 60


def collect_pdfs(folder: Path) -> list[Path]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        log.error("No PDF files found in: %s", folder)
    return pdfs


def print_summary(results: list[dict], total_elapsed: float) -> None:
    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] != "success"]

    log.info(_SEP)
    log.info("SUMMARY")
    log.info(_SEP)
    log.info("  Total   : %d", len(results))
    log.info("  Success : %d", len(success))
    log.info("  Failed  : %d", len(failed))
    log.info("  Elapsed : %.2fs", total_elapsed)
    log.info(_SEP2)

    for r in success:
        log.info("  ✔ %s  (%.2fs)", Path(r["source"]).name, r.get("duration_seconds") or 0)
    for r in failed:
        log.error("  ✘ %s  —  %s", Path(r["source"]).name, r.get("error", "unknown"))

    log.info(_SEP)


def run(pdf_folder: Path) -> None:
    pdfs = collect_pdfs(pdf_folder)
    if not pdfs:
        return

    log.info(_SEP)
    log.info("PDF EXTRACT PIPELINE")
    log.info("  Folder  : %s", pdf_folder)
    log.info("  PDFs    : %d", len(pdfs))
    log.info("  Started : %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info(_SEP)

    wall_start = time.perf_counter()
    results = []

    for i, pdf in enumerate(pdfs, 1):
        log.info(_SEP2)
        log.info("EXTRACT  %d / %d  :  %s", i, len(pdfs), pdf.name)
        log.info(_SEP2)

        choice = check_and_prompt(pdf)
        if choice == ResumeChoice.CHUNK_ONLY:
            log.info("  MinerU output already exists — skipping extract for %s.", pdf.name)
            results.append({
                "source": str(pdf),
                "status": "success",
                "duration_seconds": 0,
            })
            continue

        result = extract_pdf(pdf, choice)
        results.append(result)

        if result["status"] != "success":
            log.error("  Extract failed for %s — continuing with next file.", pdf.name)

    print_summary(results, time.perf_counter() - wall_start)

    log.info("")
    log.info("Running manifest validation pass...")
    validate_all_manifests()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Extract Pipeline")
    parser.add_argument("folder", nargs="?", help="Folder containing PDF files")
    parser.add_argument("--validate-only", action="store_true", help="Only run manifest validation")
    parser.add_argument("--embed-only", action="store_true", help="Only run page embedding into Qdrant")
    parser.add_argument("--align-only", action="store_true", help="Only run chapter alignment")
    parser.add_argument("--chunk-only", action="store_true", help="Only run chunking")
    parser.add_argument("--write-db", action="store_true", help="Write chunks + images into PostgreSQL")
    parser.add_argument("--embed-chunks", action="store_true", help="Embed chunk content into Qdrant and write point IDs back to PostgreSQL")
    parser.add_argument("--archive", action="store_true", help="Archive PDFs, images, JSONs into PostgreSQL and clean PDFInprogress folder")
    parser.add_argument("--book", metavar="BOOK_STEM", help="Limit --embed-only / --align-only / --chunk-only / --write-db / --embed-chunks / --archive to one book (e.g. Physics_Part_I)")
    parser.add_argument("--retoc", metavar="BOOK_STEM", help="Re-run ToC extraction for one book (e.g. Physics_Part_I)")
    args = parser.parse_args()

    if args.validate_only:
        validate_all_manifests()
        sys.exit(0)

    if args.embed_only:
        if args.book:
            from src.config import INPROGRESS_DIR
            from src.page_embedder import embed_book
            work_dir = INPROGRESS_DIR / args.book
            if not work_dir.exists():
                log.error("Book directory not found: %s", work_dir)
                sys.exit(1)
            embed_book(work_dir)
        else:
            embed_all_books()
        sys.exit(0)

    if args.align_only:
        if args.book:
            from src.config import INPROGRESS_DIR
            from src.chapter_aligner import align_book
            work_dir = INPROGRESS_DIR / args.book
            if not work_dir.exists():
                log.error("Book directory not found: %s", work_dir)
                sys.exit(1)
            align_book(work_dir)
        else:
            align_all_books()
        sys.exit(0)

    if args.chunk_only:
        if args.book:
            from src.config import INPROGRESS_DIR
            work_dir = INPROGRESS_DIR / args.book
            if not work_dir.exists():
                log.error("Book directory not found: %s", work_dir)
                sys.exit(1)
            chunk_book(work_dir)
        else:
            chunk_all_books()
        sys.exit(0)

    if args.write_db:
        from src.pg_writer import ensure_schema
        if args.book:
            from src.config import INPROGRESS_DIR
            work_dir = INPROGRESS_DIR / args.book
            if not work_dir.exists():
                log.error("Book directory not found: %s", work_dir)
                sys.exit(1)
            ensure_schema()
            result = write_book(work_dir)
            log.info("Result: %s", result)
        else:
            results = write_all_books()
            ok = sum(1 for r in results if r["status"] == "success")
            log.info("Done: %d / %d books written successfully.", ok, len(results))
        sys.exit(0)

    if args.embed_chunks:
        result = embed_chunks(book_stem=args.book) if args.book else embed_all_chunks()
        log.info("Result: embedded=%d  skipped=%d", result.get("embedded", 0), result.get("skipped", 0))
        sys.exit(0 if result["status"] == "success" else 1)

    if args.archive:
        if args.book:
            from src.config import INPROGRESS_DIR
            work_dir = INPROGRESS_DIR / args.book
            if not work_dir.exists():
                log.error("Book directory not found: %s", work_dir)
                sys.exit(1)
            result = archive_book(work_dir)
            log.info("Result: %s", result)
            sys.exit(0 if result["status"] == "success" else 1)
        else:
            results = archive_all_books()
            ok = sum(1 for r in results if r["status"] == "success")
            log.info("Done: %d / %d books archived successfully.", ok, len(results))
            sys.exit(0 if ok == len(results) else 1)

    if args.retoc:
        result = retoc_book(args.retoc)
        sys.exit(0 if result["status"] == "success" else 1)

    if args.folder:
        folder_input = args.folder
    else:
        folder_input = input("Enter the folder path containing PDFs: ").strip().strip('"')

    pdf_folder = Path(folder_input)
    if not pdf_folder.exists() or not pdf_folder.is_dir():
        log.error("Path does not exist or is not a directory: %s", pdf_folder)
        sys.exit(1)

    run(pdf_folder)
