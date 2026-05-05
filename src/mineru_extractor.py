"""
mineru_extractor.py
Runs MinerU on the full PDF, writes per-page .txt files, and preserves the
full MinerU output (including images) for downstream use.
Equations are preserved as inline $...$ or block $$...$$ LaTeX.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# Dedicated MinerU venv with CUDA torch — isolated from the project venv
_MINERU_PYTHON = Path(r"D:\AI\AI Projects\MinerU\MinerU_venv\Scripts\python.exe")


def _run_mineru(source_pdf: Path, output_dir: Path) -> Path:
    """Run MinerU CLI on source_pdf, return path to the content_list.json."""
    python = _MINERU_PYTHON if _MINERU_PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python), "-c",
        (
            f"import sys; sys.argv = ["
            f"'mineru', '-p', {str(source_pdf)!r}, '-o', {str(output_dir)!r},"
            f"'-b', 'pipeline', '-m', 'txt', '-l', 'en'"
            f"]; from mineru.cli.client import main; main()"
        ),
    ]
    import sys as _sys
    # Stream stdout+stderr live so progress is visible in the terminal
    stderr_lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=_sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    for line in proc.stderr:
        stderr_lines.append(line)
        print(line, end="", file=_sys.stderr, flush=True)
    proc.wait()
    if proc.returncode != 0:
        stderr_tail = "".join(stderr_lines[-20:])
        raise RuntimeError(
            f"MinerU failed with exit code {proc.returncode}"
            + (f": {stderr_tail.strip()[-500:]}" if stderr_tail else "")
        )

    content_list = output_dir / source_pdf.stem / "txt" / f"{source_pdf.stem}_content_list.json"
    if not content_list.exists():
        raise FileNotFoundError(f"MinerU output not found: {content_list}")
    return content_list


def _blocks_to_text(blocks: list[dict]) -> str:
    """Convert a page's content blocks to plain text with LaTeX equations."""
    lines = []
    for block in blocks:
        btype = block.get("type", "")
        if btype in ("text", "title"):
            lines.append(block.get("text", "").strip())
        elif btype == "equation":
            eq = block.get("text", "").strip()
            if eq:
                # MinerU already wraps block equations in $$, don't double-wrap
                if not eq.startswith("$$"):
                    eq = f"$${eq}$$"
                lines.append(eq)
        # skip: image, footer, header
    return "\n\n".join(l for l in lines if l)


def extract_text_mineru(
    source_pdf: Path,
    txt_dir: Path,
    mineru_output_dir: Path,
) -> tuple[list[Path], Path]:
    """
    Run MinerU on source_pdf. Saves full MinerU output to mineru_output_dir
    (images + content_list.json preserved for downstream use).
    Writes one .txt per page in txt_dir (page_0001.txt, ...).

    Returns (list of txt paths sorted by page, path to content_list.json).
    """
    txt_dir.mkdir(parents=True, exist_ok=True)
    mineru_output_dir.mkdir(parents=True, exist_ok=True)

    content_list_path = _run_mineru(source_pdf, mineru_output_dir)
    blocks = json.loads(content_list_path.read_text(encoding="utf-8"))

    # Group blocks by page index
    pages: dict[int, list[dict]] = {}
    for block in blocks:
        idx = block.get("page_idx", 0)
        pages.setdefault(idx, []).append(block)

    # Write one .txt per page
    total_pages = len(pages)
    txt_files: list[Path] = []
    for i, page_idx in enumerate(sorted(pages.keys()), 1):
        txt_path = txt_dir / f"page_{page_idx + 1:04d}.txt"
        text = _blocks_to_text(pages[page_idx])
        txt_path.write_text(text, encoding="utf-8")
        txt_files.append(txt_path)
        if i % 50 == 0 or i == total_pages:
            print(f"  [MinerU] Writing txt files: {i}/{total_pages} pages", flush=True)

    return txt_files, content_list_path
