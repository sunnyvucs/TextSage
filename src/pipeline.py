"""
pipeline.py
Phase 1: Extract the PDF into per-page text, detect subject, and parse ToC.
"""

import logging
import time
from pathlib import Path

from src.config import INPROGRESS_DIR, PYMUPDF_SUBJECTS, TEXT_EXTRACT_WORKERS
from src.manifest_manager import create_manifest, update_subject
from src.pdf_splitter import split_pdf
from src.resume_check import ResumeChoice
from src.subject_identifier import identify_subject
from src.text_extractor import extract_text_all
from src.timer import StepTimer
from src.toc_detector import find_toc_pages
from src.toc_extractor import extract_toc

log = logging.getLogger(__name__)
_SEP = "═" * 60


def _load_existing_pages(pdf_dir: Path, txt_dir: Path) -> tuple[list[Path], list[Path]]:
    page_pdfs = sorted(pdf_dir.glob("page_*.pdf"))
    txt_files = sorted(txt_dir.glob("page_*.txt"))
    return page_pdfs, txt_files


def _content_list_path(work_dir: Path, stem: str) -> Path:
    return work_dir / "mineru_output" / stem / "txt" / f"{stem}_content_list.json"


def _choose_extractor(subject: str) -> str:
    if (subject or "").strip().lower() in PYMUPDF_SUBJECTS:
        return "pymupdf"
    return "mineru"


def _extract_preview_pages(page_pdfs: list[Path], txt_dir: Path) -> list[Path]:
    preview_pages = page_pdfs[: min(5, len(page_pdfs))]
    if not preview_pages:
        return []
    return extract_text_all(preview_pages, txt_dir, workers=min(len(preview_pages), 4))


def extract_pdf(source_pdf: Path, resume_choice: ResumeChoice = ResumeChoice.SKIP) -> dict:
    """
    Phase 1:
      split PDF -> extract preview pages -> identify subject -> choose extractor
      -> extract full text -> write manifest -> detect/parse ToC
    """
    result = {
        "source": str(source_pdf),
        "status": "failed",
        "work_dir": None,
        "pages": 0,
        "error": None,
        "duration_seconds": None,
    }

    t0 = time.perf_counter()
    pdf_name = source_pdf.name
    work_dir = INPROGRESS_DIR / source_pdf.stem
    pdf_dir = work_dir / "pdfs"
    txt_dir = work_dir / "txts"

    log.info(_SEP)
    log.info("EXTRACT START  ▶  %s  (at %s)", pdf_name, time.strftime("%H:%M:%S"))
    log.info(_SEP)

    try:
        content_list_path: Path | None = None
        page_pdfs: list[Path] = []
        txt_files: list[Path] = []

        if resume_choice == ResumeChoice.RESUME:
            log.info("  Resuming — reloading existing split pages.")
            page_pdfs, txt_files = _load_existing_pages(pdf_dir, txt_dir)
            log.info("  Loaded %d page PDFs, %d TXT files.", len(page_pdfs), len(txt_files))

            missing = [p for p in page_pdfs if not (txt_dir / (p.stem + ".txt")).exists()]
            if missing:
                log.info("  Re-extracting %d missing TXT files with PyMuPDF...", len(missing))
                with StepTimer("Step 4 · Re-extract missing TXT files", pdf_name):
                    extract_text_all(missing, txt_dir, TEXT_EXTRACT_WORKERS)
                page_pdfs, txt_files = _load_existing_pages(pdf_dir, txt_dir)

            cl = _content_list_path(work_dir, source_pdf.stem)
            if cl.exists():
                content_list_path = cl
                log.info("  Restored MinerU content_list: %s", cl.name)
        else:
            with StepTimer("Step 1 · Prepare working directory", pdf_name):
                work_dir.mkdir(parents=True, exist_ok=True)
                pdf_dir.mkdir(parents=True, exist_ok=True)
                txt_dir.mkdir(parents=True, exist_ok=True)
                log.info("  Working dir: %s", work_dir)

            with StepTimer("Step 2 · Split PDF into per-page PDFs", pdf_name):
                page_pdfs = split_pdf(source_pdf, pdf_dir)
                log.info("  Split into %d page PDFs.", len(page_pdfs))

            with StepTimer("Step 3 · Extract preview pages for subject ID", pdf_name):
                _extract_preview_pages(page_pdfs, txt_dir)
                txt_files = sorted(txt_dir.glob("page_*.txt"))

        with StepTimer("Step 4 · Identify subject via Groq vision", pdf_name):
            subject, part, class_number = identify_subject(pdf_dir)
            log.info("  Subject: '%s'  |  Part: %s  |  Class: '%s'", subject, part, class_number)

        extractor = _choose_extractor(subject)
        log.info("  Chosen extractor for subject '%s': %s", subject, extractor)

        if resume_choice != ResumeChoice.RESUME:
            if extractor == "mineru":
                with StepTimer("Step 5 · MinerU extraction", pdf_name):
                    from src.mineru_extractor import extract_text_mineru

                    log.info("  Running MinerU on full PDF...")
                    txt_files, content_list_path = extract_text_mineru(
                        source_pdf, txt_dir, work_dir / "mineru_output"
                    )
                    log.info("  MinerU done: %d txt files.", len(txt_files))
                    log.info("  MinerU complete — GPU now free for Ollama.")
            else:
                missing = [p for p in page_pdfs if not (txt_dir / (p.stem + ".txt")).exists()]
                with StepTimer("Step 5 · Extract all pages text (parallel)", pdf_name):
                    log.info("  Extracting %d remaining pages with %d threads...", len(missing), TEXT_EXTRACT_WORKERS)
                    if missing:
                        extract_text_all(missing, txt_dir, TEXT_EXTRACT_WORKERS)
                    txt_files = sorted(txt_dir.glob("page_*.txt"))
                    log.info("  Text extraction complete: %d txt files.", len(txt_files))

        with StepTimer("Step 6 · Create manifest.json", pdf_name):
            manifest_path = create_manifest(
                source_pdf, page_pdfs, txt_files, work_dir, extractor=extractor
            )
            update_subject(manifest_path, subject, class_number, part=part)
            log.info("  Manifest saved: %s", manifest_path.name)

        with StepTimer("Step 7 · Detect Table of Contents", pdf_name):
            toc_txt_files = find_toc_pages(txt_files)
            if toc_txt_files:
                log.info("  ToC found at: %s  (%d page(s))", toc_txt_files[0].name, len(toc_txt_files))
            else:
                log.warning("  No Table of Contents found.")

        toc_json_path: Path | None = None
        if toc_txt_files:
            with StepTimer("Step 8 · Parse ToC via Groq vision", pdf_name):
                toc_json_path = work_dir / "toc.json"
                toc_page_pdfs = [
                    pdf_dir / (t.stem + ".pdf")
                    for t in toc_txt_files
                    if (pdf_dir / (t.stem + ".pdf")).exists()
                ]
                extract_toc(toc_txt_files, toc_json_path, toc_page_pdfs=toc_page_pdfs)
                log.info("  toc.json saved.")
        else:
            log.info("  Step 8 skipped — no ToC.")

        total = time.perf_counter() - t0
        log.info(_SEP)
        log.info("EXTRACT DONE  ✔  %s  |  %.2fs", pdf_name, total)
        log.info("  Pages    : %d", len(page_pdfs))
        log.info("  Subject  : %s  |  Class: %s", subject, class_number)
        log.info("  Extractor: %s", extractor)
        if content_list_path:
            log.info("  content_list.json : %s", content_list_path)
        log.info(_SEP)

        result.update({
            "status": "success",
            "work_dir": str(work_dir),
            "pages": len(page_pdfs),
            "duration_seconds": round(total, 2),
        })

    except Exception as exc:
        total = time.perf_counter() - t0
        log.error(_SEP)
        log.error("EXTRACT FAILED  ✘  %s  |  %.2fs  —  %s", pdf_name, total, exc, exc_info=True)
        log.error(_SEP)
        result["error"] = str(exc)
        result["duration_seconds"] = round(total, 2)

    return result


def retoc_book(book_stem: str) -> dict:
    """
    Re-run ToC extraction for an already-processed book using Groq vision.
    Overwrites the existing toc.json.
    """
    work_dir = INPROGRESS_DIR / book_stem
    if not work_dir.exists():
        log.error("Book directory not found: %s", work_dir)
        return {"book": book_stem, "status": "error", "error": "directory not found"}

    txt_dir = work_dir / "txts"
    pdf_dir = work_dir / "pdfs"
    txt_files = sorted(txt_dir.glob("page_*.txt"))

    if not txt_files:
        log.error("No txt files found in %s", txt_dir)
        return {"book": book_stem, "status": "error", "error": "no txt files"}

    log.info("------------------------------------------------------------")
    log.info("RETOC  %s", book_stem)
    log.info("------------------------------------------------------------")

    toc_txt_files = find_toc_pages(txt_files)
    if not toc_txt_files:
        log.error("  No ToC pages detected in %s", book_stem)
        return {"book": book_stem, "status": "error", "error": "no ToC pages detected"}

    log.info("  ToC pages: %s", [t.name for t in toc_txt_files])

    toc_page_pdfs = [
        pdf_dir / (t.stem + ".pdf")
        for t in toc_txt_files
        if (pdf_dir / (t.stem + ".pdf")).exists()
    ]
    log.info("  Page PDFs for vision: %d", len(toc_page_pdfs))

    toc_json_path = work_dir / "toc.json"
    try:
        extract_toc(toc_txt_files, toc_json_path, toc_page_pdfs=toc_page_pdfs)
        log.info("  toc.json updated.")
        return {"book": book_stem, "status": "success"}
    except Exception as exc:
        log.error("  retoc failed: %s", exc, exc_info=True)
        return {"book": book_stem, "status": "error", "error": str(exc)}
