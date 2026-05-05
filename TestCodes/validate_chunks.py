import json
from collections import Counter
from pathlib import Path

INPROGRESS = Path("PDFInprogress")

for book_dir in sorted(INPROGRESS.iterdir()):
    chunks_path = book_dir / "chunks.json"
    map_path = book_dir / "chapter_page_map.json"
    if not chunks_path.exists() or not map_path.exists():
        continue

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    ch_map = json.loads(map_path.read_text(encoding="utf-8"))

    total_pages = sum(
        c["end_page"] - c["confirmed_start_page"] + 1
        for c in ch_map["chapters"]
    )
    empty_text   = sum(1 for c in chunks if not c["text"].strip())
    has_topic    = sum(1 for c in chunks if c["topic_number"])
    no_topic     = sum(1 for c in chunks if not c["topic_number"])

    # Every page in every chapter must have >= 1 chunk
    expected_pages = set()
    for ch in ch_map["chapters"]:
        for p in range(ch["confirmed_start_page"], ch["end_page"] + 1):
            expected_pages.add(p)
    chunked_pages = set(c["page_number"] for c in chunks)
    missing_pages = sorted(expected_pages - chunked_pages)
    extra_pages   = sorted(chunked_pages - expected_pages)

    # Boundary pages = same (chapter, page) appears > 1 chunk
    page_counts = Counter((c["chapter_name"], c["page_number"]) for c in chunks)
    boundary = sum(1 for v in page_counts.values() if v > 1)

    # Sample: first 3 chunks of Physics to eyeball metadata
    sample = chunks[:3] if chunks else []

    print(f"=== {book_dir.name} ===")
    print(f"  Pages in chapters : {total_pages}")
    print(f"  Total chunks      : {len(chunks)}")
    print(f"  With topic        : {has_topic}")
    print(f"  Without topic     : {no_topic}")
    print(f"  Empty text pages  : {empty_text}")
    print(f"  Missing pages     : {len(missing_pages)}", missing_pages[:5] if missing_pages else "")
    print(f"  Extra pages       : {len(extra_pages)}", extra_pages[:5] if extra_pages else "")
    print(f"  Boundary pages    : {boundary}")
    print()

# Deep eyeball: Physics first chapter, first 3 boundary pages
print("=== PHYSICS SAMPLE CHUNKS (first boundary page) ===")
phys = json.loads((INPROGRESS / "Physics_Part_I" / "chunks.json").read_text(encoding="utf-8"))
from collections import Counter as C
pc = C((c["chapter_name"], c["page_number"]) for c in phys)
boundary_keys = {k for k, v in pc.items() if v > 1}
shown = 0
for c in phys:
    key = (c["chapter_name"], c["page_number"])
    if key in boundary_keys:
        print(f"  page={c['page_number']} ch={c['chapter_number']} topic={c['topic_number']} | {c['topic_name']}")
        shown += 1
    if shown >= 6:
        break
