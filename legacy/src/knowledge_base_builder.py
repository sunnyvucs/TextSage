"""
knowledge_base_builder.py
Merges chunks.json + images_index.json into a single knowledge_base.json.
Cross-links chunks ↔ images by page range.
Extracts equations from chunk content into a separate list per chunk.
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Matches block equations: $$ ... $$
_BLOCK_EQ_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Matches inline equations: $ ... $  (not $$)
_INLINE_EQ_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _extract_equations(text: str) -> tuple[str, list[str]]:
    """
    Pull block and inline equations out of text.
    Returns (clean_text_with_placeholders_removed, list_of_equation_strings).
    """
    equations = []

    def replace_block(m):
        eq = m.group(0).strip()
        equations.append(eq)
        return ""

    def replace_inline(m):
        eq = m.group(0).strip()
        equations.append(eq)
        return ""

    clean = _BLOCK_EQ_RE.sub(replace_block, text)
    clean = _INLINE_EQ_RE.sub(replace_inline, clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, equations


def build_knowledge_base(
    chunks_path: Path,
    images_index_path: Path,
    output_path: Path,
    doc_id: str,
    source_pdf_name: str,
) -> Path:
    """
    Merge chunks and images into knowledge_base.json.
    - Each chunk gets: equations extracted, image_refs populated by page overlap
    - Each image gets: linked chunk_ids
    Returns path to knowledge_base.json.
    """
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    images_data = json.loads(images_index_path.read_text(encoding="utf-8")) \
        if images_index_path and images_index_path.exists() else {"images": []}

    images = images_data.get("images", [])

    # Build image lookup by page (handles both "page" and "page_number" keys)
    img_by_page: dict[int, list[dict]] = {}
    for img in images:
        pg = img.get("page") or img.get("page_number") or 0
        img_by_page.setdefault(pg, []).append(img)

    # Image ID → entry (for back-linking)
    img_by_id: dict[str, dict] = {img["image_id"]: img for img in images}
    # Track which chunks reference each image
    img_chunk_links: dict[str, list[str]] = {img["image_id"]: [] for img in images}

    enriched_chunks = []
    for chunk in chunks_data.get("chunks", []):
        # Extract equations from content
        clean_content, equations = _extract_equations(chunk.get("content", ""))

        # Find images that fall within this chunk's page range
        page_start = chunk.get("page_start") or 0
        page_end = chunk.get("page_end") or page_start
        image_refs = []
        for pg in range(page_start, page_end + 1):
            for img in img_by_page.get(pg, []):
                if img["image_id"] not in image_refs:
                    image_refs.append(img["image_id"])
                    img_chunk_links[img["image_id"]].append(chunk["chunk_id"])

        enriched_chunks.append({
            **chunk,
            "content":    clean_content,
            "equations":  equations,
            "has_equations": bool(equations),
            "image_refs": image_refs,
            "has_images": bool(image_refs),
        })

    # Add chunk_ids back-link to each image
    enriched_images = []
    for img in images:
        enriched_images.append({
            **img,
            "chunk_ids": img_chunk_links.get(img["image_id"], []),
        })

    knowledge_base = {
        "document_id":   doc_id,
        "source_file":   source_pdf_name,
        "total_chunks":  len(enriched_chunks),
        "total_images":  len(enriched_images),
        "chunks":        enriched_chunks,
        "images":        enriched_images,
    }

    output_path.write_text(
        json.dumps(knowledge_base, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_eq = sum(len(c["equations"]) for c in enriched_chunks)
    log.info("  [KB] knowledge_base.json: %d chunks, %d images, %d equations",
             len(enriched_chunks), len(enriched_images), total_eq)
    return output_path
