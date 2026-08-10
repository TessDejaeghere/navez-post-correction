"""
Split the combined GLiNER2 pre-annotations for `NER_manual_batch2` into
one shared IAA file and four per-annotator files ready for Label Studio.

Each pre-annotated task carries `data.assignment` (set by
`build_manual_ner_batch.py`); we simply group on that field.

Output (into `data/labelstudio_upload/NER_manual_batch2/`):

    packs/
      NER_manual_batch2_iaa10.json           <- 10 tasks, upload to every project
      NER_manual_batch2_A1.json         <- IAA-10 + A1's unique share
      NER_manual_batch2_A2.json
      NER_manual_batch2_A3.json
      NER_manual_batch2_A4.json

Every per-annotator file already includes the 10 shared IAA tasks, so you can
create one Label Studio project per annotator with a single import.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR    = ROOT / "data" / "labelstudio_upload" / "NER_manual_batch2"
PREANNOT_DIR = ROOT / "data" / "preannots_GLiNER2" / "NER_manual_batch2"
COMBINED     = PREANNOT_DIR / "gliner_preannots.json"

ANNOTATORS = ["A1", "A2", "A3", "A4"]


def main() -> None:
    tasks = json.loads(COMBINED.read_text(encoding="utf-8"))
    print(f"loaded {len(tasks)} pre-annotated tasks from {COMBINED.name}")

    by_assign: dict[str, list] = defaultdict(list)
    for t in tasks:
        a = ((t.get("data") or {}).get("assignment") or "").strip()
        if not a:
            raise SystemExit(f"task missing `data.assignment`: "
                             f"{t.get('data',{}).get('letter_id')}")
        by_assign[a].append(t)

    print("counts by assignment:")
    for k, v in sorted(by_assign.items()):
        print(f"  {k:>8}: {len(v)}")

    iaa = by_assign.pop("IAA", [])
    if len(iaa) != 10:
        print(f"warning: expected 10 IAA tasks, got {len(iaa)}")

    out_dir = BATCH_DIR / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared IAA-only file (upload to every project OR as a standalone project)
    iaa_file = out_dir / "NER_manual_batch2_iaa10.json"
    iaa_file.write_text(json.dumps(iaa, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {iaa_file.relative_to(ROOT)}  ({len(iaa)} tasks)")

    # Per-annotator files: their unique share + the 10 IAA tasks
    for who in ANNOTATORS:
        pack = by_assign.get(who, []) + iaa
        f = out_dir / f"NER_manual_batch2_{who}.json"
        f.write_text(json.dumps(pack, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"wrote {f.relative_to(ROOT)}  "
              f"({len(pack)} tasks = {len(pack)-len(iaa)} unique + {len(iaa)} IAA)")

    # Sanity: no leftover assignments
    leftover = set(by_assign) - set(ANNOTATORS)
    if leftover:
        print(f"warning: unexpected assignment values: {sorted(leftover)}")


if __name__ == "__main__":
    main()
