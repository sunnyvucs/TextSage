"""
resume_check.py
Before processing a PDF, checks if it was already split in a previous run.
Asks the user whether to resume from book identification + ToC extraction,
or restart everything from scratch.
"""

import shutil
import logging
from pathlib import Path
from src.config import INPROGRESS_DIR, FINAL_OUTPUT_DIR

log = logging.getLogger(__name__)


def _find_output_folder(pdf_stem: str) -> Path | None:
    """Return the existing FinalOutput subject folder for this PDF stem, if any."""
    for folder in FINAL_OUTPUT_DIR.iterdir():
        if folder.is_dir():
            manifest = folder / "manifest.json"
            if manifest.exists():
                import json
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if Path(data.get("source_pdf", "")).stem == pdf_stem:
                        return folder
                except Exception:
                    pass
    return None


class ResumeChoice:
    RESUME     = "resume"      # skip splitting, redo identification + ToC
    RESTART    = "restart"     # delete everything and start fresh
    SKIP       = "skip"        # this PDF was not previously processed — proceed normally
    CHUNK_ONLY = "chunk_only"  # MinerU extraction done — run chunking + KB build only


def _has_mineru_output(source_pdf: Path) -> bool:
    """Check if MinerU content_list.json exists for this PDF."""
    work_dir = INPROGRESS_DIR / source_pdf.stem
    cl = work_dir / "mineru_output" / source_pdf.stem / "txt" / f"{source_pdf.stem}_content_list.json"
    return cl.exists()


def check_and_prompt(source_pdf: Path) -> ResumeChoice:
    """
    Check if source_pdf has already been split into PDFInprogress.
    If yes, prompt the user for what to do.
    Returns a ResumeChoice value.
    """
    work_dir = INPROGRESS_DIR / source_pdf.stem
    pdf_dir  = work_dir / "pdfs"

    if not pdf_dir.exists() or not any(pdf_dir.glob("page_*.pdf")):
        return ResumeChoice.SKIP

    page_count = len(list(pdf_dir.glob("page_*.pdf")))
    has_mineru = _has_mineru_output(source_pdf)

    log.info("─" * 60)
    log.info("⚠  Previously processed PDF detected: %s", source_pdf.name)
    log.info("   %d page PDFs already exist in: %s", page_count, work_dir)
    if has_mineru:
        log.info("   MinerU content_list.json is present.")
    log.info("─" * 60)
    print()
    print(f"  '{source_pdf.name}' was already split into {page_count} pages.")
    if has_mineru:
        print("  MinerU extraction output found.")
    print()
    print("  Choose an option:")
    print("  [1] Resume     — skip splitting, redo book identification + ToC extraction only")
    print("  [2] Restart    — delete everything and start from scratch")
    if has_mineru:
        print("  [3] Chunk only — MinerU output exists, run chunking + KB build only")
    print()

    valid = ["1", "2", "3"] if has_mineru else ["1", "2"]
    while True:
        choice = input(f"  Enter {'/'.join(valid)}: ").strip()
        if choice == "1":
            log.info("User chose RESUME for: %s", source_pdf.name)
            _clear_output_only(source_pdf.stem)
            return ResumeChoice.RESUME
        elif choice == "2":
            log.info("User chose RESTART for: %s", source_pdf.name)
            _clear_all(source_pdf.stem)
            return ResumeChoice.RESTART
        elif choice == "3" and has_mineru:
            log.info("User chose CHUNK_ONLY for: %s", source_pdf.name)
            _clear_output_only(source_pdf.stem)
            return ResumeChoice.CHUNK_ONLY
        else:
            print(f"  Invalid input. Please enter {' or '.join(valid)}.")


def _clear_output_only(pdf_stem: str) -> None:
    """Remove only the FinalOutput subject folder for this PDF."""
    output_folder = _find_output_folder(pdf_stem)
    if output_folder and output_folder.exists():
        shutil.rmtree(output_folder)
        log.info("Cleared output folder: %s", output_folder)
    else:
        log.info("No existing output folder found for: %s", pdf_stem)


def _clear_all(pdf_stem: str) -> None:
    """Remove both the PDFInprogress work folder and the FinalOutput subject folder."""
    work_dir = INPROGRESS_DIR / pdf_stem
    if work_dir.exists():
        shutil.rmtree(work_dir)
        log.info("Cleared inprogress folder: %s", work_dir)

    _clear_output_only(pdf_stem)
