"""
pipeline.py
Two decoupled pipeline phases for a single PDF:

  extract_pdf()  — Steps 1-7: split, MinerU extraction, subject ID, ToC.
                   Writes to PDFInprogress/<stem>/. Stops before chunking.

  chunk_pdf()    — Steps 8b-8e: images, chunking, knowledge base, FinalOutput.
                   Reads from existing PDFInprogress/<stem>/ MinerU output.

main.py orchestrates which phase runs and in what order.
"""

import json
import logging
import re
import time
from pathlib import Path

from src.config import INPROGRESS_DIR, TEXT_EXTRACT_WORKERS, TEXT_EXTRACTOR
from src.pdf_splitter import split_pdf
from src.text_extractor import extract_text, extract_text_all
from src.manifest_manager import create_manifest, update_subject
from src.subject_identifier import identify_subject
from src.toc_detector import find_toc_pages
from src.toc_extractor import extract_toc
from src.image_extractor import extract_images
from src.output_organizer import organize_output
from src.resume_check import ResumeChoice
from src.timer import StepTimer

log = logging.getLogger(__name__)
_SEP = "═" * 60


def _load_existing_pages(pdf_dir: Path, txt_dir: Path) -> tuple[list[Path], list[Path]]:
    page_pdfs = sorted(pdf_dir.glob("page_*.pdf"))
    txt_files = sorted(txt_dir.glob("page_*.txt"))
    return page_pdfs, txt_files


def _content_list_path(work_dir: Path, stem: str) -> Path:
    return work_dir / "mineru_output" / stem / "txt" / f"{stem}_content_list.json"


# ── Phase 1: Extract ─────────────────────────────────────────────────────────

def extract_pdf(source_pdf: Path, resume_choice: ResumeChoice = ResumeChoice.SKIP) -> dict:
    """
    Steps 1-7: split PDF, run MinerU, identify subject, detect + parse ToC.
    Writes all output to PDFInprogress/<stem>/.
    Does NOT do chunking or FinalOutput.

    Returns result dict with keys: source, status, work_dir, pages, error, duration_seconds
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
    pdf_dir  = work_dir / "pdfs"
    txt_dir  = work_dir / "txts"

    log.info(_SEP)
    log.info("EXTRACT START  ▶  %s  (at %s)", pdf_name, time.strftime("%H:%M:%S"))
    log.info(_SEP)

    try:
        content_list_path: Path | None = None

        if resume_choice == ResumeChoice.RESUME:
            log.info("  Resuming — reloading existing split pages.")
            page_pdfs, txt_files = _load_existing_pages(pdf_dir, txt_dir)
            log.info("  Loaded %d page PDFs, %d TXT files.", len(page_pdfs), len(txt_files))

            missing = [p for p in page_pdfs if not (txt_dir / (p.stem + ".txt")).exists()]
            if missing:
                log.info("  Re-extracting %d missing TXT files...", len(missing))
                with StepTimer("Step 4 · Re-extract missing TXT files", pdf_name):
                    extract_text_all(missing, txt_dir, TEXT_EXTRACT_WORKERS)
                page_pdfs, txt_files = _load_existing_pages(pdf_dir, txt_dir)

            cover_txt = txt_dir / "page_0001.txt"
            cl = _content_list_path(work_dir, source_pdf.stem)
            if cl.exists():
                content_list_path = cl
                log.info("  Restored MinerU content_list: %s", cl.name)

        else:
            # ── Step 1 ───────────────────────────────────────────────────────
            with StepTimer("Step 1 · Prepare working directory", pdf_name):
                work_dir.mkdir(parents=True, exist_ok=True)
                pdf_dir.mkdir(parents=True, exist_ok=True)
                txt_dir.mkdir(parents=True, exist_ok=True)
                log.info("  Working dir: %s", work_dir)

            # ── Step 2 ───────────────────────────────────────────────────────
            with StepTimer("Step 2 · Split PDF into per-page PDFs", pdf_name):
                page_pdfs = split_pdf(source_pdf, pdf_dir)
                log.info("  Split into %d page PDFs.", len(page_pdfs))

            # ── Steps 3-4 ────────────────────────────────────────────────────
            if TEXT_EXTRACTOR == "mineru":
                with StepTimer("Step 3-4 · MinerU extraction", pdf_name):
                    from src.mineru_extractor import extract_text_mineru
                    log.info("  Running MinerU on full PDF...")
                    txt_files, content_list_path = extract_text_mineru(
                        source_pdf, txt_dir, work_dir / "mineru_output"
                    )
                    log.info("  MinerU done: %d txt files.", len(txt_files))
                    cover_txt = txt_files[0] if txt_files else txt_dir / "page_0001.txt"
            else:
                with StepTimer("Step 3 · Extract cover page text", pdf_name):
                    cover_txt = extract_text(page_pdfs[0], txt_dir)
                with StepTimer("Step 4 · Extract all pages text (parallel)", pdf_name):
                    log.info("  Extracting %d pages with %d threads...", len(page_pdfs), TEXT_EXTRACT_WORKERS)
                    txt_files = extract_text_all(page_pdfs, txt_dir, TEXT_EXTRACT_WORKERS)
                    log.info("  Text extraction complete: %d txt files.", len(txt_files))

        # ── Step 4b: identify subject ─────────────────────────────────────────
        with StepTimer("Step 4b · Identify subject via LLM", pdf_name):
            extra = [txt_files[i] for i in (1, 2) if i < len(txt_files)]
            subject, class_number = identify_subject(cover_txt, extra_pages=extra)
            log.info("  Subject: '%s'  |  Class: '%s'", subject, class_number)

        # ── Step 5: manifest ──────────────────────────────────────────────────
        with StepTimer("Step 5 · Create manifest.json", pdf_name):
            manifest_path = create_manifest(source_pdf, page_pdfs, txt_files, work_dir)
            update_subject(manifest_path, subject, class_number)
            log.info("  Manifest saved: %s", manifest_path.name)

        # ── Step 6: detect ToC ────────────────────────────────────────────────
        with StepTimer("Step 6 · Detect Table of Contents", pdf_name):
            toc_txt_files = find_toc_pages(txt_files)
            if toc_txt_files:
                log.info("  ToC found at: %s  (%d page(s))", toc_txt_files[0].name, len(toc_txt_files))
            else:
                log.warning("  No Table of Contents found.")

        # ── Step 7: extract ToC structure ────────────────────────────────────
        toc_json_path: Path | None = None
        if toc_txt_files:
            with StepTimer("Step 7 · Parse ToC via LLM", pdf_name):
                toc_json_path = work_dir / "toc.json"
                extract_toc(toc_txt_files, toc_json_path)
                log.info("  toc.json saved.")
        else:
            log.info("  Step 7 skipped — no ToC.")

        total = time.perf_counter() - t0
        log.info(_SEP)
        log.info("EXTRACT DONE  ✔  %s  |  %.2fs", pdf_name, total)
        log.info("  Pages   : %d", len(page_pdfs))
        log.info("  Subject : %s  |  Class: %s", subject, class_number)
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


# ── Phase 2: Chunk ───────────────────────────────────────────────────────────

def chunk_pdf(source_pdf: Path) -> dict:
    """
    Steps 8b-8e: build image index, chunk, knowledge base, copy to FinalOutput.
    Reads from existing PDFInprogress/<stem>/ — requires extract_pdf() to have run first.

    Returns result dict with keys: source, status, output_dir, error, duration_seconds
    """
    from src.chunker import chunk_document
    from src.knowledge_base_builder import build_knowledge_base

    result = {
        "source": str(source_pdf),
        "status": "failed",
        "output_dir": None,
        "error": None,
        "duration_seconds": None,
    }

    t0 = time.perf_counter()
    pdf_name = source_pdf.name
    work_dir = INPROGRESS_DIR / source_pdf.stem
    pdf_dir  = work_dir / "pdfs"
    txt_dir  = work_dir / "txts"

    log.info(_SEP)
    log.info("CHUNK START  ▶  %s  (at %s)", pdf_name, time.strftime("%H:%M:%S"))
    log.info(_SEP)

    try:
        # ── Load existing extraction output ───────────────────────────────────
        page_pdfs, txt_files = _load_existing_pages(pdf_dir, txt_dir)
        if not page_pdfs:
            raise FileNotFoundError(f"No page PDFs found in {pdf_dir}. Run extract phase first.")
        log.info("  Loaded %d page PDFs, %d TXT files.", len(page_pdfs), len(txt_files))

        cl = _content_list_path(work_dir, source_pdf.stem)
        content_list_path = cl if cl.exists() else None
        if content_list_path:
            log.info("  MinerU content_list.json found.")
        else:
            log.warning("  No content_list.json — chunker will use txt fallback.")

        # ── Load manifest (subject/class) ─────────────────────────────────────
        manifest_path = work_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {work_dir}. Run extract phase first.")
        _m = json.loads(manifest_path.read_text(encoding="utf-8"))
        subject      = _m.get("subject", "Unknown")
        class_number = _m.get("class_number", "Unknown")
        log.info("  Subject: %s  |  Class: %s", subject, class_number)

        # ── Load toc.json ─────────────────────────────────────────────────────
        toc_json_path = work_dir / "toc.json"
        toc_data: dict = {}
        toc_page_nums: set[int] = set()
        if toc_json_path.exists():
            toc_data = json.loads(toc_json_path.read_text(encoding="utf-8"))
            log.info("  toc.json loaded: %d chapters.", len(toc_data.get("chapters", [])))
        else:
            log.warning("  No toc.json found — chunking without chapter structure.")

        # Infer ToC page numbers from txt filenames recorded in toc (or skip)
        # We re-detect from txt files since we don't store them explicitly
        toc_txt_files = find_toc_pages(txt_files)
        if toc_txt_files:
            toc_page_nums = {
                int(re.search(r"(\d+)", f.stem).group(1))
                for f in toc_txt_files if re.search(r"(\d+)", f.stem)
            }

        # ── Step 8b: image index ──────────────────────────────────────────────
        images_index_path: Path | None = None
        if TEXT_EXTRACTOR == "mineru" and content_list_path:
            with StepTimer("Step 8b · Build image index from MinerU output", pdf_name):
                from src.mineru_image_builder import build_images_index
                images_index_path = build_images_index(
                    content_list_path=content_list_path,
                    mineru_output_dir=work_dir / "mineru_output",
                    images_dest_dir=work_dir / "images",
                    txt_files=txt_files,
                    toc=toc_data,
                    doc_id=source_pdf.stem,
                )
        else:
            with StepTimer("Step 8b · Extract images (parallel)", pdf_name):
                images_index_path = extract_images(
                    page_pdfs=page_pdfs,
                    output_dir=work_dir,
                    toc=toc_data,
                    workers=TEXT_EXTRACT_WORKERS,
                )

        # ── Step 8c: chunk ────────────────────────────────────────────────────
        images_data: dict = {}
        if images_index_path and images_index_path.exists():
            images_data = json.loads(images_index_path.read_text(encoding="utf-8"))

        with StepTimer("Step 8c · Chunk document", pdf_name):
            chunks_path = work_dir / "chunks.json"
            chunk_document(
                txt_files=txt_files,
                toc=toc_data,
                images_index=images_data,
                output_path=chunks_path,
                doc_id=source_pdf.stem,
                skip_pages=toc_page_nums or None,
                content_list_path=content_list_path,
            )

        # ── Step 8d: knowledge base ───────────────────────────────────────────
        with StepTimer("Step 8d · Build knowledge_base.json", pdf_name):
            kb_path = work_dir / "knowledge_base.json"
            build_knowledge_base(
                chunks_path=chunks_path,
                images_index_path=images_index_path,
                output_path=kb_path,
                doc_id=source_pdf.stem,
                source_pdf_name=source_pdf.name,
            )

        # ── Step 8e: organise FinalOutput ─────────────────────────────────────
        with StepTimer("Step 8e · Copy to FinalOutput", pdf_name):
            dest = organize_output(
                work_dir=work_dir,
                subject=subject,
                class_number=class_number,
                page_pdfs=page_pdfs,
                txt_files=txt_files,
                manifest_path=manifest_path,
                toc_json_path=toc_json_path if toc_json_path.exists() else None,
                images_index_path=images_index_path,
                chunks_path=chunks_path,
                kb_path=kb_path,
                folder_name=source_pdf.stem,
            )
            log.info("  Output folder: %s", dest)

        total = time.perf_counter() - t0
        log.info(_SEP)
        log.info("CHUNK DONE  ✔  %s  |  %.2fs", pdf_name, total)
        log.info("  Output : %s", dest)
        log.info(_SEP)

        result.update({
            "status": "success",
            "output_dir": str(dest),
            "duration_seconds": round(total, 2),
        })

    except Exception as exc:
        total = time.perf_counter() - t0
        log.error(_SEP)
        log.error("CHUNK FAILED  ✘  %s  |  %.2fs  —  %s", pdf_name, total, exc, exc_info=True)
        log.error(_SEP)
        result["error"] = str(exc)
        result["duration_seconds"] = round(total, 2)

    return result


# ── Legacy entry point (kept for backwards compatibility) ─────────────────────

def process_pdf(source_pdf: Path, resume_choice: ResumeChoice = ResumeChoice.SKIP, phase: str = "full") -> dict:
    """Full pipeline: extract then chunk in one call."""
    if phase == "extract":
        return extract_pdf(source_pdf, resume_choice)
    if phase == "chunk":
        return chunk_pdf(source_pdf)

    # phase == "full": run extract then chunk
    r = extract_pdf(source_pdf, resume_choice)
    if r["status"] != "success":
        return r
    return chunk_pdf(source_pdf)
