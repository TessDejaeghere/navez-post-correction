"""
Select the N most-recently annotated HTR corrections not yet uploaded for NER,
and emit Label Studio NER import tasks for them.

Given
  * a Label Studio HTR-correction export JSON (new / intermediate export), and
  * a folder of previously-uploaded NER import files
    (``data/labelstudio_upload/NER_from_HTR_corrections/<name>/`` produced by
    ``build_ner_tasks_from_corrections.py``),

this script drops any ``ms_key`` already present in the upload folder, sorts
the remaining tasks by the most recent annotation ``updated_at`` (descending),
takes the top N, and writes per-letter + combined NER import JSONs to a fresh
output folder.

Usage (from repo root)::

    python NER_pipeline/select_recent_corrections_for_ner.py \
        data/labelstudio_output/HTR_107_output_A3.json \
        --already-uploaded-dir data/labelstudio_upload/NER_from_HTR_corrections/HTR_corrections_A3 \
        --top-n 50 \
        --output-dir data/labelstudio_upload/NER_from_HTR_corrections/HTR_corrections_A3_batch2_top50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the extraction / task-building logic from the existing pipeline script
# so the schema stays identical to earlier NER batches.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ner_tasks_from_corrections import (  # noqa: E402
    build_task,
    pick_corrected_text,
    safe_filename,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="Label Studio HTR-correction export JSON.")
    p.add_argument("--already-uploaded-dir", type=Path, required=True,
                   help="Folder with previously-uploaded NER tasks "
                        "(one *_ls_import.json per letter).")
    p.add_argument("--top-n", type=int, default=50,
                   help="How many most-recent unseen tasks to emit (default 50).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the new NER import files.")
    return p.parse_args()


def already_uploaded_shelfmarks(folder: Path) -> set[str]:
    """Read the `data.letter_id` field from each *_ls_import.json in *folder*.

    Reading the JSON is safer than reversing the filename `/` -> `_` mangling.
    """
    seen: set[str] = set()
    if not folder.exists():
        return seen
    for f in folder.glob("*_ls_import.json"):
        try:
            tasks = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(tasks, list):
            tasks = [tasks]
        for t in tasks:
            lid = (t.get("data") or {}).get("letter_id")
            if lid:
                seen.add(lid)
    return seen


def latest_annotation_timestamp(task: dict) -> str:
    """ISO timestamp of the most recent non-cancelled annotation, else ''."""
    anns = task.get("annotations") or []
    non_c = [a for a in anns if not a.get("was_cancelled")]
    if not non_c:
        return ""
    return max((a.get("updated_at") or a.get("created_at") or "") for a in non_c)


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    export = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(export, list):
        print("ERROR: expected a Label Studio export JSON array.", file=sys.stderr)
        return 1

    seen = already_uploaded_shelfmarks(args.already_uploaded_dir)
    print(f"already-uploaded shelfmarks in {args.already_uploaded_dir.name}: {len(seen)}")

    # Extract (task, ms_key, corrected_text, ts) for every candidate.
    candidates: list[tuple[dict, str, str, str]] = []
    counts = {"corrected": 0, "cancelled": 0, "unannotated": 0,
              "no_textarea": 0, "missing_id": 0, "already_uploaded": 0}

    for task in export:
        data = task.get("data") or {}
        ms_key = data.get("ms_key") or data.get("letter_id")
        if not ms_key:
            counts["missing_id"] += 1
            continue
        if ms_key in seen:
            counts["already_uploaded"] += 1
            continue
        text, status = pick_corrected_text(task)
        counts[status] = counts.get(status, 0) + 1
        if status != "corrected" or not text:
            continue
        candidates.append((task, ms_key, text, latest_annotation_timestamp(task)))

    print(f"total tasks in export           : {len(export)}")
    print(f"  skipped (already uploaded)    : {counts['already_uploaded']}")
    print(f"  skipped (missing shelfmark)   : {counts['missing_id']}")
    print(f"  skipped (cancelled)           : {counts.get('cancelled', 0)}")
    print(f"  skipped (unannotated)         : {counts.get('unannotated', 0)}")
    print(f"  skipped (no transcription TA) : {counts.get('no_textarea', 0)}")
    print(f"eligible unseen corrected tasks : {len(candidates)}")

    candidates.sort(key=lambda x: x[3], reverse=True)
    picked = candidates[: args.top_n]
    print(f"picking top {len(picked)} by most-recent annotation timestamp\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict] = []
    for task, ms_key, text, ts in picked:
        ner_task = build_task(ms_key, text, task.get("data") or {})
        combined.append(ner_task)
        per_letter = args.output_dir / f"{safe_filename(ms_key)}_ls_import.json"
        per_letter.write_text(json.dumps([ner_task], ensure_ascii=False, indent=2),
                              encoding="utf-8")

    stem = args.input.stem
    combined_path = args.output_dir / f"{stem}_top{len(picked)}_ner_import.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    manifest = args.output_dir / f"{stem}_top{len(picked)}_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        f.write("rank\tms_key\tlatest_annotation_ts\tdate\tsender\treceiver\n")
        for i, (task, ms_key, _text, ts) in enumerate(picked, 1):
            d = task.get("data") or {}
            f.write(f"{i}\t{ms_key}\t{ts}\t{d.get('date','')}\t"
                    f"{d.get('source','')}\t{d.get('target','')}\n")

    print(f"wrote {len(combined)} NER tasks to {args.output_dir}")
    print(f"  combined file : {combined_path.name}")
    print(f"  manifest      : {manifest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
