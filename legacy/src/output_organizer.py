"""
output_organizer.py
Copies all processed files into a clean, organised FinalOutput folder:

  FinalOutput/<Subject>_Class<N>/
  ├── knowledge_base.json     ← main RAG file
  ├── toc.json
  ├── manifest.json
  ├── pages/                  ← all per-page PDFs + TXTs
  ├── images/                 ← all extracted images
  └── chunks/                 ← one JSON per chapter
"""

import json
import re
import shutil
from pathlib import Path
from src.config import FINAL_OUTPUT_DIR


def _sanitize(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "Unknown"


def build_subject_folder(subject: str, class_number: str) -> Path:
    folder_name = f"{_sanitize(subject)}_Class{_sanitize(class_number)}"
    folder = FINAL_OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_chapter_chunks(kb_path: Path, chunks_dir: Path) -> None:
    """Split knowledge_base chunks into per-chapter JSON files."""
    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    by_chapter: dict[str, list[dict]] = {}

    for chunk in kb.get("chunks", []):
        ch_num = chunk.get("chapter_number") or "00"
        by_chapter.setdefault(ch_num, []).append(chunk)

    for ch_num, chunks in sorted(by_chapter.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        # Get chapter name from first chunk
        ch_name = chunks[0].get("chapter_name") or "Unknown"
        safe_name = re.sub(r"[^\w\s-]", "", ch_name).strip().replace(" ", "_")[:40]
        fname = f"chapter_{int(ch_num):02d}_{safe_name}.json" if ch_num.isdigit() else f"chapter_00_front_matter.json"
        out = {
            "chapter_number": ch_num,
            "chapter_name": ch_name,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
        (chunks_dir / fname).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def organize_output(
    work_dir: Path,
    subject: str,
    class_number: str,
    page_pdfs: list[Path],
    txt_files: list[Path],
    manifest_path: Path,
    toc_json_path: Path | None,
    images_index_path: Path | None = None,
    chunks_path: Path | None = None,
    kb_path: Path | None = None,
    folder_name: str | None = None,
) -> Path:
    if folder_name:
        dest = FINAL_OUTPUT_DIR / _sanitize(folder_name)
        dest.mkdir(parents=True, exist_ok=True)
    else:
        dest = build_subject_folder(subject, class_number)

    # ── pages/ subfolder ─────────────────────────────────────────────────────
    pages_dir = dest / "pages"
    pages_dir.mkdir(exist_ok=True)
    for pdf in page_pdfs:
        shutil.copy2(pdf, pages_dir / pdf.name)
    for txt in txt_files:
        shutil.copy2(txt, pages_dir / txt.name)

    # ── root JSON files ───────────────────────────────────────────────────────
    shutil.copy2(manifest_path, dest / manifest_path.name)

    if toc_json_path and toc_json_path.exists():
        shutil.copy2(toc_json_path, dest / toc_json_path.name)

    if kb_path and kb_path.exists():
        shutil.copy2(kb_path, dest / kb_path.name)

    # ── images/ subfolder ────────────────────────────────────────────────────
    if images_index_path and images_index_path.exists():
        src_images_dir = work_dir / "images"
        dest_images_dir = dest / "images"
        if src_images_dir.exists():
            if dest_images_dir.exists():
                shutil.rmtree(dest_images_dir)
            shutil.copytree(src_images_dir, dest_images_dir)

    # ── chunks/ subfolder — one JSON per chapter ─────────────────────────────
    if kb_path and kb_path.exists():
        chunks_dir = dest / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        chunks_dir.mkdir()
        _write_chapter_chunks(kb_path, chunks_dir)

    return dest
