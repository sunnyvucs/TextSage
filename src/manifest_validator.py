"""
manifest_validator.py
Validates all manifests in PDFInprogress/ and patches any with missing or
Unknown subject/part/class_number using Groq vision on the page images.
Writes a validation_report.json to each book's work directory.
"""

import json
import logging
import time
from pathlib import Path

from src.config import INPROGRESS_DIR
from src.subject_identifier import identify_subject

log = logging.getLogger(__name__)
_SEP = "-" * 60


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_BAD_VALUES = {"UNKNOWN", "NONE", "NULL", ""}
_ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}


def _is_bad(value) -> bool:
    return value is None or str(value).strip().upper() in _BAD_VALUES


def _is_roman(value) -> bool:
    return value is not None and str(value).strip().upper() in _ROMAN


def _needs_fix(manifest: dict) -> bool:
    return (
        _is_bad(manifest.get("subject"))
        or _is_bad(manifest.get("class_number"))
        or _is_roman(manifest.get("class_number"))
        or _is_roman(manifest.get("part"))
    )


def _write_report(work_dir: Path, report: dict) -> None:
    path = work_dir / "validation_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_all_manifests() -> list[dict]:
    """
    Scan every book in PDFInprogress/, check manifest quality,
    patch missing fields via Groq vision, and write per-book validation_report.json.
    Returns a list of result dicts (one per book).
    """
    results = []
    book_dirs = sorted(d for d in INPROGRESS_DIR.iterdir() if d.is_dir())

    if not book_dirs:
        log.info("  [Validator] No books found in %s", INPROGRESS_DIR)
        return results

    log.info(_SEP)
    log.info("MANIFEST VALIDATION  (%d books)", len(book_dirs))
    log.info(_SEP)

    for book_dir in book_dirs:
        manifest_path = book_dir / "manifest.json"
        pdf_dir = book_dir / "pdfs"

        if not manifest_path.exists():
            log.warning("  [%s] No manifest.json — skipping.", book_dir.name)
            continue

        manifest = _load_manifest(manifest_path)
        result = {
            "book": book_dir.name,
            "patched": False,
            "subject": manifest.get("subject"),
            "part": manifest.get("part"),
            "class_number": manifest.get("class_number"),
            "pages": len(manifest.get("pages", [])),
        }

        if not _needs_fix(manifest):
            log.info("  [%s] OK  subject=%s  part=%s  class=%s",
                     book_dir.name, manifest["subject"], manifest.get("part"), manifest["class_number"])
            result["status"] = "ok"
            _write_report(book_dir, result)
            results.append(result)
            continue

        log.info("  [%s] Needs fix (bad/Roman values) — running Groq vision...", book_dir.name)

        if not pdf_dir.exists() or not any(pdf_dir.glob("page_*.pdf")):
            log.warning("  [%s] No page PDFs found — cannot fix.", book_dir.name)
            result["status"] = "no_pdfs"
            _write_report(book_dir, result)
            results.append(result)
            continue

        try:
            t0 = time.perf_counter()
            subject, part, class_number = identify_subject(pdf_dir)
            elapsed = time.perf_counter() - t0

            # Overwrite all identity fields with normalized values from Groq vision
            manifest["subject"] = subject
            manifest["class_number"] = class_number
            manifest["part"] = part

            _save_manifest(manifest, manifest_path)

            result.update({
                "status": "patched",
                "patched": True,
                "subject": manifest["subject"],
                "part": manifest.get("part"),
                "class_number": manifest["class_number"],
                "fix_duration_seconds": round(elapsed, 2),
            })
            log.info("  [%s] Patched => subject=%s  part=%s  class=%s  (%.2fs)",
                     book_dir.name, manifest["subject"], manifest.get("part"), manifest["class_number"], elapsed)

        except Exception as exc:
            log.error("  [%s] Fix failed: %s", book_dir.name, exc)
            result["status"] = "error"
            result["error"] = str(exc)

        _write_report(book_dir, result)
        results.append(result)

    log.info(_SEP)
    ok      = sum(1 for r in results if r.get("status") == "ok")
    patched = sum(1 for r in results if r.get("status") == "patched")
    errors  = sum(1 for r in results if r.get("status") == "error")
    log.info("VALIDATION DONE  —  ok=%d  patched=%d  errors=%d", ok, patched, errors)
    log.info(_SEP)

    return results
