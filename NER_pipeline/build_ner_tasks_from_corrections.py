"""
Turn a finished HTR-correction Label Studio export into Label Studio NER
import tasks (text-only, no predictions).

Input  : a Label Studio export JSON of the HTR-correction project, e.g.
         ``data/labelstudio_upload/HTR_corrections_finished/HTR_corrections_A3.json``
Output : per-letter ``<letter_id>_ls_import.json`` files plus a combined
         ``<stem>_ner_import.json`` array, written into an output folder.
         Default output folder:
         ``data/labelstudio_upload/NER_from_HTR_corrections/<input-stem>/``.

The output schema matches ``data/labelstudio_upload/GLINER_batch_1/*.json``:

    {"data": {"letter_id", "text", "meta_date",
              "meta_sender", "meta_receiver", "image_url"}}

so it can be (a) imported directly into the NER Label Studio project, or
(b) handed to ``NER_pipeline/gliner_predict.py --input-dir <out-folder>``
for zero-shot preannotation first.

Usage (from repo root):

    python NER_pipeline/build_ner_tasks_from_corrections.py \
        data/labelstudio_upload/HTR_corrections_finished/HTR_corrections_A3.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path,
                   help="Label Studio export JSON of the HTR-correction project.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write the NER import files. "
                        "Default: data/labelstudio_upload/NER_from_HTR_corrections/<stem>/")
    p.add_argument("--include-skipped", action="store_true",
                   help="Also emit tasks that were cancelled/skipped in HTR "
                        "(falls back to the original claude_text).")
    p.add_argument("--include-unannotated", action="store_true",
                   help="Also emit tasks with zero annotations "
                        "(falls back to claude_text).")
    return p.parse_args()


def safe_filename(letter_id: str) -> str:
    """``Ms. II 70/1/131`` -> ``Ms. II 70_1_131`` (matches existing GLINER files)."""
    return letter_id.replace("/", "_")


def pick_corrected_text(task: dict) -> tuple[str | None, str]:
    """Return (corrected_text, status) for the task.

    Status is one of:
      - 'corrected'   : transcription TextArea found, used.
      - 'cancelled'   : annotation was cancelled by the annotator.
      - 'unannotated' : no annotations at all.
      - 'no_textarea' : annotation exists but no `transcription` result.
    """
    anns = task.get("annotations") or []
    if not anns:
        return None, "unannotated"

    # Pick the most recent non-cancelled annotation; if all cancelled,
    # pick the most recent cancelled one.
    non_cancelled = [a for a in anns if not a.get("was_cancelled")]
    chosen = (non_cancelled or anns)[-1]
    if chosen.get("was_cancelled"):
        return None, "cancelled"

    for r in chosen.get("result", []):
        if r.get("from_name") == "transcription":
            val = r.get("value", {})
            txt = val.get("text")
            if isinstance(txt, list):
                txt = txt[0] if txt else None
            if isinstance(txt, str) and txt.strip():
                return txt, "corrected"
    return None, "no_textarea"


def build_task(letter_id: str, text: str, src_data: dict) -> dict:
    return {
        "data": {
            "letter_id": letter_id,
            "text": text,
            "meta_date": src_data.get("date", "") or "",
            "meta_sender": src_data.get("source", "") or "",
            "meta_receiver": src_data.get("target", "") or "",
            "image_url": "",
            # Preserve the nodegoat numeric ID for traceability back to the
            # source HTR-correction task.
            "nodegoat_id": str(src_data.get("letter_id", "") or ""),
        }
    }


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (
        Path("data/labelstudio_upload/NER_from_HTR_corrections") / args.input.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as f:
        export = json.load(f)
    if not isinstance(export, list):
        print("ERROR: expected a Label Studio export JSON array.", file=sys.stderr)
        return 1

    combined: list[dict] = []
    counts = {"corrected": 0, "cancelled": 0, "unannotated": 0,
              "no_textarea": 0, "missing_id": 0, "missing_text": 0,
              "fallback_claude": 0}

    for task in export:
        data = task.get("data") or {}
        # The HTR-correction export stores the shelfmark ('Ms. II 70/1/131')
        # under ``ms_key`` and the numeric nodegoat id under ``letter_id``.
        # The NER batches use the shelfmark as ``letter_id``, so map ms_key
        # onto it (falling back to letter_id for older exports).
        letter_id = data.get("ms_key") or data.get("letter_id")
        if not letter_id:
            counts["missing_id"] += 1
            continue

        text, status = pick_corrected_text(task)
        counts[status] += 1

        if text is None:
            allow_fallback = (
                (status == "cancelled" and args.include_skipped) or
                (status == "unannotated" and args.include_unannotated)
            )
            if not allow_fallback:
                continue
            text = data.get("claude_text") or ""
            if not text.strip():
                counts["missing_text"] += 1
                continue
            counts["fallback_claude"] += 1

        ner_task = build_task(letter_id, text, data)
        combined.append(ner_task)

        per_letter_path = out_dir / f"{safe_filename(letter_id)}_ls_import.json"
        with per_letter_path.open("w", encoding="utf-8") as f:
            json.dump([ner_task], f, ensure_ascii=False, indent=2)

    combined_path = out_dir / f"{args.input.stem}_ner_import.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(combined)} NER tasks to {out_dir}")
    print(f"  combined file: {combined_path}")
    print("  status breakdown:")
    for k, v in counts.items():
        if v:
            print(f"    {k:>16}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
