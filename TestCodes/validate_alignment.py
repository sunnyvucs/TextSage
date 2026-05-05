import json
from pathlib import Path

for book in sorted(Path("PDFInprogress").iterdir()):
    mp = book / "chapter_page_map.json"
    if not mp.exists():
        print(book.name, "-- MISSING chapter_page_map.json")
        continue
    d = json.loads(mp.read_text(encoding="utf-8"))
    chs = d["chapters"]
    manifest = json.loads((book / "manifest.json").read_text(encoding="utf-8"))
    total_pages = len(manifest["pages"])

    print()
    print(f"{'='*70}")
    print(f"{book.name}  ({len(chs)} chapters, {total_pages} total pages)")
    print(f"{'='*70}")

    issues = []
    prev_end = 0
    for i, c in enumerate(chs):
        s    = c["confirmed_start_page"]
        e    = c["end_page"]
        name = c["chapter_name"][:45]
        ch   = str(c["chapter_number"])
        flags = []
        if s <= prev_end and i > 0:
            flags.append("OVERLAP")
        if e is not None and e < s:
            flags.append("BAD_END")
        if s < 1 or s > total_pages:
            flags.append("OUT_OF_RANGE")
        if e is not None and e > total_pages:
            flags.append("END_EXCEEDS")
        gap = s - prev_end - 1 if i > 0 else 0
        flag_str = " *** " + ",".join(flags) if flags else ""
        print(f"  Ch {ch:>5}  pages {s:>4}-{e:>4}  gap={gap:>3}{flag_str}  {name}")
        if flags:
            issues.append(f"Ch {ch}: {','.join(flags)}")
        prev_end = e if e else s

    if issues:
        print(f"  ISSUES: {'; '.join(issues)}")
    else:
        print("  OK - no issues detected")

print()
