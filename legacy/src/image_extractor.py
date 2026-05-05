"""
image_extractor.py
Extracts figures from PDFs by finding figure captions and rendering the
page region above each caption. Captures vector drawings, embedded images,
and text labels exactly as they appear — no raw image extraction.

Mapping priority:
  1. Caption number (e.g. "FIGURE 3.2" → chapter 3, look up topic in toc.json)
  2. Topic heading above the figure on the page (Y-position)
  3. Carry-forward last known topic from previous pages
  4. Page number only
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

log = logging.getLogger(__name__)

# Matches: FIGURE 1.3, Fig. 2.1, Figure 3.4(a), FIG 1.1
_CAPTION_RE = re.compile(
    r"\b(?:FIGURE|Figure|Fig\.?|FIG\.?)\s+(\d+)\.(\d+)", re.IGNORECASE
)

# Matches topic headings like "3.1  INTRODUCTION"
_TOPIC_HEADING_RE = re.compile(r"^(\d+)\.(\d+)\s{2,}\S")

# Horizontal padding on each side
_CROP_H_PADDING = 20
# Minimum rendered image size to keep
_MIN_PIXELS = 5000
# Max gap (points) to look upward for a preceding text block boundary
_MAX_TEXT_SEARCH_UP = 500
# Minimum gap between caption top and crop top (always capture at least this much)
_MIN_CROP_HEIGHT = 80
# Page header margin to skip (points from top) — avoids capturing page header text
_PAGE_HEADER_MARGIN = 40


def _build_lookups(toc: dict) -> tuple[dict, dict]:
    topic_lookup = {}
    chapter_lookup = {}
    for ch in toc.get("chapters", []):
        ch_num = str(ch["chapter_number"])
        chapter_lookup[ch_num] = ch
        for t in ch.get("topics", []):
            t_num = str(t["topic_number"])
            topic_lookup[t_num] = {
                "chapter_number": ch_num,
                "chapter_name":   ch["chapter_name"],
                "topic_number":   t_num,
                "topic_name":     t["topic_name"],
            }
    return topic_lookup, chapter_lookup


def _topic_above(blocks_sorted: list, fig_y0: float, topic_lookup: dict, chapter_lookup: dict) -> dict | None:
    """Return metadata for the closest topic heading above fig_y0."""
    last = None
    for b in blocks_sorted:
        if b[1] >= fig_y0:
            break
        m = _TOPIC_HEADING_RE.match(b[4].strip())
        if m:
            t_num = f"{m.group(1)}.{m.group(2)}"
            ch_num = m.group(1)
            if t_num in topic_lookup:
                last = {**topic_lookup[t_num], "mapping_method": "y_position"}
            elif ch_num in chapter_lookup:
                ch = chapter_lookup[ch_num]
                last = {
                    "chapter_number": ch_num,
                    "chapter_name":   ch["chapter_name"],
                    "topic_number":   None,
                    "topic_name":     None,
                    "mapping_method": "y_position_chapter_only",
                }
    return last


def _process_page(
    page_pdf: Path,
    page_num: int,
    images_dir: Path,
    topic_lookup: dict,
    chapter_lookup: dict,
) -> list[dict]:
    """
    Find all figure captions on the page, render the region above each,
    and return metadata entries.
    """
    results = []
    try:
        doc = fitz.open(str(page_pdf))
        page = doc[0]
        blocks_sorted = sorted(page.get_text("blocks"), key=lambda b: b[1])

        seen_figs = set()
        fig_index = 0

        for b in blocks_sorted:
            x0, y0, x1, y1, text, *_ = b
            m = _CAPTION_RE.search(text)
            if not m:
                continue

            ch_str   = m.group(1)
            t_suffix = m.group(2)
            fig_key  = f"{ch_str}.{t_suffix}"

            if fig_key in seen_figs:
                continue
            seen_figs.add(fig_key)

            # ── Smart top boundary: find nearest text block above caption ────
            # Walk blocks from bottom upward to find the closest non-caption
            # text block above this figure's caption. That block's bottom edge
            # becomes the crop top, so we capture the entire figure region.
            crop_top = _PAGE_HEADER_MARGIN  # default: near top of page
            for tb in reversed(blocks_sorted):
                tb_y1 = tb[3]  # bottom of this text block
                tb_y0 = tb[1]  # top of this text block
                if tb_y1 >= y0:
                    # This block is at or below caption — skip
                    continue
                if y0 - tb_y1 > _MAX_TEXT_SEARCH_UP:
                    # Too far above — stop searching, use page top
                    break
                # Skip if this block IS the caption itself (same text)
                if tb[4].strip() == text.strip():
                    continue
                # Skip topic/chapter headings (they're labels, not content above fig)
                block_text = tb[4].strip()
                if _TOPIC_HEADING_RE.match(block_text):
                    # Still use this as boundary — heading separates sections
                    crop_top = max(_PAGE_HEADER_MARGIN, tb_y1 + 2)
                    break
                # Regular text block found above the figure — use its bottom as boundary
                crop_top = max(_PAGE_HEADER_MARGIN, tb_y1 + 2)
                break

            # If caption is in bottom half and crop_top ended up being page top,
            # that's fine — we want the full figure. Ensure min height.
            crop_bottom = min(page.rect.height, y1 + 5)
            if crop_bottom - crop_top < _MIN_CROP_HEIGHT:
                crop_top = max(_PAGE_HEADER_MARGIN, crop_bottom - _MIN_CROP_HEIGHT)

            # Use full page width — figures typically span the entire column/page
            crop = fitz.Rect(
                _CROP_H_PADDING,
                crop_top,
                page.rect.width - _CROP_H_PADDING,
                crop_bottom,
            )
            pix = page.get_pixmap(dpi=150, clip=crop)

            if pix.width * pix.height < _MIN_PIXELS:
                continue

            fig_index += 1
            img_id   = f"page_{page_num:04d}_fig_{fig_index:03d}"
            img_file = images_dir / f"{img_id}.png"
            pix.save(str(img_file))

            # ── Metadata mapping ─────────────────────────────────────────────
            full_t_num = f"{ch_str}.{t_suffix}"
            if full_t_num in topic_lookup:
                meta = {**topic_lookup[full_t_num], "mapping_method": "caption"}
            elif ch_str in chapter_lookup:
                ch = chapter_lookup[ch_str]
                # Find the topic that's active at caption Y position
                y_meta = _topic_above(blocks_sorted, y0, topic_lookup, chapter_lookup)
                if y_meta:
                    meta = y_meta
                    meta["mapping_method"] = "caption_chapter+y_position"
                else:
                    meta = {
                        "chapter_number": ch_str,
                        "chapter_name":   ch["chapter_name"],
                        "topic_number":   None,
                        "topic_name":     None,
                        "mapping_method": "caption_chapter_only",
                    }
            else:
                y_meta = _topic_above(blocks_sorted, y0, topic_lookup, chapter_lookup)
                meta = y_meta or {
                    "chapter_number": None,
                    "chapter_name":   None,
                    "topic_number":   None,
                    "topic_name":     None,
                    "mapping_method": "page_only",
                }

            results.append({
                "image_id":    img_id,
                "figure_ref":  f"Fig {fig_key}",
                "caption":     text.strip()[:120],
                "page_number": page_num,
                "image_file":  f"images/{img_id}.png",
                "position": {
                    "x0": round(crop.x0, 1),
                    "y0": round(crop.y0, 1),
                    "x1": round(crop.x1, 1),
                    "y1": round(crop.y1, 1),
                },
                **meta,
            })

        doc.close()
    except Exception as exc:
        log.warning("  [Images] Failed on page_%04d: %s", page_num, exc)

    return results


def extract_images(
    page_pdfs: list[Path],
    output_dir: Path,
    toc: dict,
    workers: int = 8,
) -> Path:
    """
    Extract all figures from all page PDFs in parallel.
    Saves rendered crops to output_dir/images/ and writes images_index.json.
    Returns path to images_index.json.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    topic_lookup, chapter_lookup = _build_lookups(toc)

    log.info("  [Images] Scanning %d pages for figures (parallel, %d threads)...",
             len(page_pdfs), workers)
    t0 = time.perf_counter()

    raw: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_process_page, pdf, idx + 1, images_dir, topic_lookup, chapter_lookup): idx
            for idx, pdf in enumerate(page_pdfs)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            raw[idx] = future.result()

    # Apply carry-forward sequentially across pages
    all_images = []
    carried: dict | None = None
    for idx in range(len(page_pdfs)):
        for entry in raw.get(idx, []):
            if entry["mapping_method"] == "page_only" and carried:
                entry.update({**carried, "mapping_method": "carry_forward"})
            if entry.get("topic_number"):
                carried = {k: entry[k] for k in
                           ("chapter_number", "chapter_name", "topic_number", "topic_name")}
            all_images.append(entry)

    duration = time.perf_counter() - t0

    index_path = output_dir / "images_index.json"
    index_path.write_text(
        json.dumps({"total_images": len(all_images), "images": all_images},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    methods = {}
    for img in all_images:
        m = img["mapping_method"]
        methods[m] = methods.get(m, 0) + 1

    log.info("  [Images] Done: %d figures extracted in %.2fs", len(all_images), duration)
    log.info("  [Images] Mapping: %s", methods)
    log.info("  [Images] Index → images_index.json")
    return index_path
