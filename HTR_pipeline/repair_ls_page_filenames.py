#!/usr/bin/env python3
"""Patch existing Label Studio task JSONs to refresh `page_filenames_str` /
`n_pages` after the fix to `page_filenames_for_letter()` (the old version
looked only in the now-defunct `data/brieven_te_HTRen/_pages_*`).

This script does NOT re-run grammar checks or GLiNER. It only updates the two
page-related fields per task, using the same lookup the builder now uses
(including the on-demand zip extraction fallback).

Default targets: every
`data/labelstudio_upload/sample_priority_HTR_to_correct/labelstudio_sample_priority*.json`
(including any auto- or annotator-named variants), so a single run fixes the
source files and the per-annotator splits in one go.

Usage:
    python HTR_pipeline/repair_ls_page_filenames.py
    python HTR_pipeline/repair_ls_page_filenames.py path/to/file1.json path/to/file2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Reuse the helper that already does the right thing (PAGES_ROOTS + zip fallback).
sys.path.insert(0, str(REPO / "HTR_pipeline"))
from build_labelstudio_correction_tasks import page_filenames_for_letter  # noqa: E402

DEFAULT_GLOB_DIR = REPO / "data" / "labelstudio_upload" / "sample_priority_HTR_to_correct"
DEFAULT_GLOB     = "labelstudio_sample_priority*.json"


def patch_file(path: Path) -> tuple[int, int, int]:
    """Returns (n_tasks, n_repaired, n_still_missing)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data if isinstance(data, list) else data.get("tasks", [])
    repaired = 0
    still_missing = 0
    for t in tasks:
        d = t.get("data") if isinstance(t, dict) else None
        if not d:
            continue
        kbr   = (d.get("letter_id") or "").strip()
        batch = (d.get("htr_opus_batch") or "").strip()
        if not kbr:
            continue
        page_filenames = page_filenames_for_letter(kbr, batch)
        old_str = d.get("page_filenames_str", "")
        new_str = (", ".join(page_filenames) if page_filenames
                   else "(no scans found locally)")
        if page_filenames:
            if old_str != new_str or d.get("n_pages", -1) != len(page_filenames):
                d["page_filenames_str"] = new_str
                d["n_pages"]            = len(page_filenames)
                repaired += 1
        else:
            still_missing += 1
            # Make sure the placeholder is at least present/consistent.
            d["page_filenames_str"] = new_str
            d["n_pages"]            = 0
    if isinstance(data, list):
        path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        data["tasks"] = tasks
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(tasks), repaired, still_missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path,
                    help="Specific JSON files to patch. If omitted, defaults to "
                         f"{DEFAULT_GLOB_DIR.relative_to(REPO)}/{DEFAULT_GLOB}")
    args = ap.parse_args()

    if args.files:
        targets = [p.resolve() for p in args.files]
    else:
        targets = sorted(DEFAULT_GLOB_DIR.glob(DEFAULT_GLOB))
        if not targets:
            print(f"No files matched {DEFAULT_GLOB_DIR}/{DEFAULT_GLOB}")
            return 1

    print(f"Patching {len(targets)} file(s):")
    for p in targets:
        if not p.exists():
            print(f"  [skip missing] {p}")
            continue
        n, r, m = patch_file(p)
        print(f"  {p.relative_to(REPO)}: {n} tasks  "
              f"repaired={r}  still_missing={m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
