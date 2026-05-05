"""
mineru_image_builder.py
Builds images_index.json from MinerU's content_list.json output.
Copies MinerU's extracted images to work_dir/images/.
Links each image to its chapter/topic using a page→topic map built
from scanned .txt files.
"""

import json
import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Matches: "1.3 COULOMB'S LAW", "1.14 Applications of Gauss's Law"
_TOPIC_RE = re.compile(r"^(\d+)\.(\d+)\s+(\S.*)", re.MULTILINE)

# Matches: "Chapter One ...", "Chapter Two ..." etc.
_CHAPTER_WORD_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_CHAPTER_RE = re.compile(r"Chapter\s+(\w+)\s+(.+)", re.IGNORECASE)

# Matches figure captions: "FIGURE 1.3", "Fig. 2.1"
_CAPTION_FIG_RE = re.compile(r"\b(?:FIGURE|Figure|Fig\.?)\s+(\d+)\.(\d+)", re.IGNORECASE)


def _page_num(txt_file: Path) -> int:
    m = re.search(r"(\d+)", txt_file.stem)
    return int(m.group(1)) if m else 0


def build_page_to_topic_map(txt_files: list[Path], toc: dict) -> dict[int, dict]:
    """
    Scan all .txt files for chapter/topic heading patterns.
    Returns {page_number: {chapter_number, chapter_name, topic_number, topic_name}}.
    Carries forward last seen metadata to un-headed pages.
    """
    # Build lookup from toc
    topic_lookup: dict[str, dict] = {}
    chapter_lookup: dict[str, dict] = {}
    for ch in toc.get("chapters", []):
        ch_num = str(ch["chapter_number"])
        chapter_lookup[ch_num] = {
            "chapter_number": ch_num,
            "chapter_name": ch["chapter_name"],
            "topic_number": None,
            "topic_name": None,
        }
        for t in ch.get("topics", []):
            t_num = str(t["topic_number"])
            topic_lookup[t_num] = {
                "chapter_number": ch_num,
                "chapter_name": ch["chapter_name"],
                "topic_number": t_num,
                "topic_name": t["topic_name"],
            }

    page_map: dict[int, dict] = {}
    carried: dict | None = None

    for txt_file in sorted(txt_files, key=_page_num):
        pg = _page_num(txt_file)
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        found: dict | None = None

        # Check for topic heading (most specific)
        for m in _TOPIC_RE.finditer(text):
            t_num = f"{m.group(1)}.{m.group(2)}"
            if t_num in topic_lookup:
                found = topic_lookup[t_num]
                break

        # Check for chapter heading if no topic found
        if not found:
            m = _CHAPTER_RE.search(text)
            if m:
                word = m.group(1).lower()
                ch_num = _CHAPTER_WORD_MAP.get(word, word)
                if ch_num in chapter_lookup:
                    found = chapter_lookup[ch_num]

        if found:
            carried = found
            page_map[pg] = found
        elif carried:
            page_map[pg] = carried

    return page_map


def build_images_index(
    content_list_path: Path,
    mineru_output_dir: Path,
    images_dest_dir: Path,
    txt_files: list[Path],
    toc: dict,
    doc_id: str,
) -> Path:
    """
    Read MinerU content_list.json, copy images to images_dest_dir,
    and write images_index.json with full metadata.
    Returns path to images_index.json.
    """
    images_dest_dir.mkdir(parents=True, exist_ok=True)

    blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
    mineru_images_dir = content_list_path.parent / "images"

    # Build page→topic map
    page_to_topic = build_page_to_topic_map(txt_files, toc)

    image_entries: list[dict] = []
    img_seq: dict[int, int] = {}  # page → count for naming

    for block in blocks:
        if block.get("type") != "image":
            continue

        page_idx = block.get("page_idx", 0)
        page_num = page_idx + 1  # 1-based

        # Source image from MinerU output
        src_img_rel = block.get("img_path", "")
        src_img = mineru_images_dir / Path(src_img_rel).name
        if not src_img.exists():
            continue

        # Rename to human-readable: page_0024_img_001.jpg
        img_seq[page_num] = img_seq.get(page_num, 0) + 1
        ext = src_img.suffix
        img_id = f"page_{page_num:04d}_img_{img_seq[page_num]:03d}"
        dest_img = images_dest_dir / f"{img_id}{ext}"
        shutil.copy2(src_img, dest_img)

        # Extract caption text
        captions = block.get("image_caption", [])
        footnotes = block.get("image_footnote", [])
        caption_text = " ".join(c for c in captions + footnotes if c and c.strip())

        # Try to get figure number from caption for better topic mapping
        topic_meta = page_to_topic.get(page_num, {})
        fig_m = _CAPTION_FIG_RE.search(caption_text)
        if fig_m:
            # Caption has e.g. "FIGURE 1.3" — use chapter from caption
            fig_ch = fig_m.group(1)
            fig_topic_key = f"{fig_m.group(1)}.{fig_m.group(2)}"
            # Build lookup inline
            for ch in toc.get("chapters", []):
                if str(ch["chapter_number"]) == fig_ch:
                    for t in ch.get("topics", []):
                        if str(t["topic_number"]) == fig_topic_key:
                            topic_meta = {
                                "chapter_number": str(ch["chapter_number"]),
                                "chapter_name": ch["chapter_name"],
                                "topic_number": fig_topic_key,
                                "topic_name": t["topic_name"],
                            }
                            break

        image_entries.append({
            "image_id":       img_id,
            "document_id":    doc_id,
            "file":           f"images/{img_id}{ext}",
            "page_number":    page_num,   # kept for chunker compatibility
            "page":           page_num,
            "bbox":           block.get("bbox", []),
            "caption":        caption_text,
            "chapter_number": topic_meta.get("chapter_number"),
            "chapter_name":   topic_meta.get("chapter_name"),
            "topic_number":   topic_meta.get("topic_number"),
            "topic_name":     topic_meta.get("topic_name"),
            "source":         "mineru",
        })

    index = {
        "total_images": len(image_entries),
        "images": image_entries,
    }
    index_path = images_dest_dir.parent / "images_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("  [Images] %d images extracted and indexed.", len(image_entries))
    return index_path
