"""
chunk_pdfs.py
Standalone chunking script. Reads existing MinerU output from PDFInprogress/
and writes chunks + knowledge_base + FinalOutput.

Usage:
  python chunk_pdfs.py                      # chunk all PDFs in PDFInprogress/
  python chunk_pdfs.py <stem> [<stem> ...]  # chunk specific books by folder name

Does not touch main.py or the extraction phase at all.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
from src.config import INPROGRESS_DIR
from src.pipeline import chunk_pdf
from src.resume_check import _has_mineru_output

setup_logging()
log = logging.getLogger(__name__)

_SEP  = "═" * 60
_SEP2 = "─" * 60


def find_targets(stems: list[str]) -> list[Path]:
    """Return list of source PDF paths to chunk, resolved from PDFInprogress stems."""
    targets = []
    for stem in stems:
        work_dir = INPROGRESS_DIR / stem
        if not work_dir.exists():
            log.error("  Not found in PDFInprogress: %s", stem)
            continue
        # Find the original PDF name from manifest
        manifest = work_dir / "manifest.json"
        if manifest.exists():
            import json
            m = json.loads(manifest.read_text(encoding="utf-8"))
            pdf_name = Path(m.get("source_pdf", stem + ".pdf")).name
        else:
            pdf_name = stem + ".pdf"
        # We only need the Path object — chunk_pdf uses work_dir derived from stem
        # Construct a fake path with the right stem so chunk_pdf works correctly
        targets.append(Path(pdf_name).with_suffix(".pdf"))
    return targets


def run(stems: list[str] | None = None) -> None:
    if stems:
        # Build Path objects that have the right .stem for chunk_pdf()
        targets = []
        for stem in stems:
            work_dir = INPROGRESS_DIR / stem
            if not work_dir.exists():
                log.error("PDFInprogress/%s does not exist.", stem)
                continue
            import json
            manifest = work_dir / "manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text(encoding="utf-8"))
                pdf_name = m.get("source_pdf", stem + ".pdf")
                targets.append(Path(pdf_name))
            else:
                targets.append(Path(stem + ".pdf"))
    else:
        # Auto-discover all stems in PDFInprogress that have MinerU output
        targets = []
        for work_dir in sorted(INPROGRESS_DIR.iterdir()):
            if not work_dir.is_dir():
                continue
            manifest = work_dir / "manifest.json"
            if not manifest.exists():
                continue
            import json
            m = json.loads(manifest.read_text(encoding="utf-8"))
            pdf_path = Path(m.get("source_pdf", work_dir.name + ".pdf"))
            targets.append(pdf_path)

    if not targets:
        log.error("No targets found to chunk.")
        return

    log.info(_SEP)
    log.info("CHUNK-ONLY RUN  —  %d book(s)", len(targets))
    log.info("  Started: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info(_SEP)

    results = []
    wall_start = time.perf_counter()

    for i, pdf_path in enumerate(targets, 1):
        log.info(_SEP2)
        log.info("  %d / %d  :  %s", i, len(targets), pdf_path.name)
        log.info(_SEP2)

        if not _has_mineru_output(pdf_path):
            log.warning("  No MinerU output for %s — skipping.", pdf_path.name)
            results.append({"source": str(pdf_path), "status": "skipped"})
            continue

        result = chunk_pdf(pdf_path)
        results.append(result)

    total = time.perf_counter() - wall_start
    log.info(_SEP)
    log.info("DONE  —  %.2fs", total)
    for r in results:
        icon = "✔" if r["status"] == "success" else ("⚠" if r["status"] == "skipped" else "✘")
        msg  = r.get("output_dir") or r.get("error") or r["status"]
        log.info("  %s  %s  →  %s", icon, Path(r["source"]).name, msg)
    log.info(_SEP)


if __name__ == "__main__":
    stems = sys.argv[1:] if len(sys.argv) > 1 else None
    run(stems)
