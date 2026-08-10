"""
Cosmetic clean-up of the raw NER-import task texts in
`data/labelstudio_upload/NER_manual_batch2/tasks/` before running GLiNER2:

  1. Detach uncertainty / gap markers glued to word chars on either side:
        `raturet[?]`      -> `raturet [?]`
        `[XXX]lie`        -> `[XXX] lie`
     Otherwise Label Studio's word-select double-click grabs the marker
     into the span.

  2. Strip trailing whitespace at end of each line (Windows/Unix leftovers).

  3. Collapse runs of 3+ blank lines to a single blank line (paragraph
     separator). Line breaks are otherwise preserved — annotators use them
     visually.

Original texts are preserved by copying the whole `tasks/` folder to
`tasks_raw_uncleaned/` on first run.

Usage (from repo root):

    python NER_pipeline/clean_task_texts.py            # in-place
    python NER_pipeline/clean_task_texts.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT      = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "data" / "labelstudio_upload" / "NER_manual_batch2"
TASKS_DIR = BATCH_DIR / "tasks"
BACKUP_DIR = BATCH_DIR / "tasks_raw_uncleaned"

# Markers we treat as gap / uncertainty tokens. Deliberately conservative:
# `[?]`, `[XXX]`, `[!]`, `[…]`, `[...]`.
MARKER_ALT = r"XXX|\?|!|…|\.\.\."

# `\w`-before-marker  →  insert space
RE_GLUED_LEFT  = re.compile(rf"(\w)(\[(?:{MARKER_ALT})\])")
# marker-before-\w  →  insert space
RE_GLUED_RIGHT = re.compile(rf"(\[(?:{MARKER_ALT})\])(\w)")

RE_TRAILING_WS  = re.compile(r"[ \t]+\n")
RE_TRIPLE_BLANK = re.compile(r"\n{3,}")


def clean(text: str) -> tuple[str, dict]:
    """Return (cleaned_text, counts_of_edits)."""
    counts = {"glued_left": 0, "glued_right": 0, "trailing_ws": 0, "blank_runs": 0}

    def _left_sub(m):
        counts["glued_left"] += 1
        return f"{m.group(1)} {m.group(2)}"

    def _right_sub(m):
        counts["glued_right"] += 1
        return f"{m.group(1)} {m.group(2)}"

    out = RE_GLUED_LEFT.sub(_left_sub, text)
    out = RE_GLUED_RIGHT.sub(_right_sub, out)

    out, tws_n = RE_TRAILING_WS.subn("\n", out)
    counts["trailing_ws"] = tws_n

    out, blk_n = RE_TRIPLE_BLANK.subn("\n\n", out)
    counts["blank_runs"] = blk_n

    return out, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; don't write anything.")
    args = ap.parse_args()

    files = sorted(TASKS_DIR.glob("*_ls_import.json"))
    if not files:
        raise SystemExit(f"no *_ls_import.json in {TASKS_DIR}")

    # Backup once, before any writes.
    if not args.dry_run and not BACKUP_DIR.exists():
        shutil.copytree(TASKS_DIR, BACKUP_DIR)
        print(f"backed up originals -> {BACKUP_DIR.relative_to(ROOT)}")

    totals   = {"glued_left": 0, "glued_right": 0,
                "trailing_ws": 0, "blank_runs": 0}
    changed  = 0
    unchanged = 0

    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        task    = payload[0]
        text    = task["data"]["text"]

        cleaned, counts = clean(text)
        if cleaned == text:
            unchanged += 1
            continue

        for k in totals:
            totals[k] += counts[k]
        changed += 1

        if not args.dry_run:
            task["data"]["text"] = cleaned
            f.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"tasks scanned      : {len(files)}")
    print(f"tasks unchanged    : {unchanged}")
    print(f"tasks changed      : {changed}")
    print(f"total edits by kind:")
    for k, v in totals.items():
        print(f"  {k:<14}: {v}")
    if args.dry_run:
        print("\n[dry-run] no files written.")


if __name__ == "__main__":
    main()
