"""
manifest_manager.py
Creates and updates the per-PDF manifest JSON that links every page PDF to its TXT file
and stores metadata like subject and class number.
"""

import json
from pathlib import Path


def create_manifest(
    source_pdf: Path,
    page_pdfs: list[Path],
    txt_files: list[Path],
    work_dir: Path,
    extractor: str | None = None,
) -> Path:
    """
    Build the initial manifest JSON for a PDF.
    page_pdfs and txt_files must be in the same order (index = page - 1).

    Returns the path to the saved manifest.json.
    """
    pages = []
    for i, (pdf, txt) in enumerate(zip(page_pdfs, txt_files), start=1):
        pages.append({
            "page": i,
            "pdf": pdf.name,
            "txt": txt.name,
        })

    manifest = {
        "source_pdf": source_pdf.name,
        "subject": None,
        "part": None,
        "class_number": None,
        "extractor": extractor,
        "pages": pages,
    }

    manifest_path = work_dir / "manifest.json"
    _save(manifest, manifest_path)
    return manifest_path


def update_subject(manifest_path: Path, subject: str, class_number: str, part: str | None = None) -> None:
    """Patch subject, part, and class_number into an existing manifest."""
    manifest = _load(manifest_path)
    manifest["subject"] = subject
    manifest["part"] = part
    manifest["class_number"] = class_number
    _save(manifest, manifest_path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
