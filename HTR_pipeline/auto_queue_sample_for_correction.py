#!/usr/bin/env python3
"""Auto-queue freshly HTR'd sample letters for Label Studio correction.

Run AFTER `build_letters_overview.py` (so the overview reflects the new HTR
outputs) and BEFORE `sample_pipeline_status.py` (so the status table picks up
the new addendum file).

Logic:
  1. Load the sample Ms_key set from data/sample_letters/Sample_Letters.zip.
  2. Find sample keys that have an HTR_OPUS_path in letters_overview.csv and
     are NOT manually transcribed.
  3. Subtract any keys already present in
     data/labelstudio_upload/sample_priority_HTR_to_correct/labelstudio_sample_priority*.json.
  4. If the remainder is non-empty, write the keys to a temp text file and
     invoke `build_labelstudio_correction_tasks.py --priority-ids <tmp>
     --out labelstudio_sample_priority_auto_<UTC_ts>.json`.

Idempotent: if all eligible sample letters are already queued, prints a note
and exits 0 without writing anything.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ZIP_PATH       = REPO / "data" / "sample_letters" / "Sample_Letters.zip"
OVERVIEW_CSV   = REPO / "data" / "letters_overview.csv"
PRIORITY_DIR   = (REPO / "data" / "labelstudio_upload"
                  / "sample_priority_HTR_to_correct")
PRIORITY_GLOB  = "labelstudio_sample_priority*.json"
TMP_IDS_TXT    = REPO / "data" / "sample_letters" / "_auto_queue_priority_ids.txt"
BUILDER        = REPO / "HTR_pipeline" / "build_labelstudio_correction_tasks.py"


def normalize_key(s: str) -> str:
    """Canonical 'Ms. II 70_<tome>_<rest>' (underscore-separated)."""
    s = (s or "").strip()
    if not s:
        return s
    for ext in (".txt", ".TXT", ".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff",
                ".json", "_ls_import"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^Ms\.?\s*II\s*70\s*(.*)$", s, re.IGNORECASE)
    if m:
        rest = re.sub(r"\s+", "_", m.group(1).strip())
        rest = re.sub(r"_+", "_", rest).strip("_")
        s = f"Ms. II 70_{rest}" if rest else "Ms. II 70"
    return s.rstrip(".")


def load_sample_keys() -> set[str]:
    keys: set[str] = set()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            k = normalize_key(Path(name).stem)
            if k and k.startswith("Ms. II 70_"):
                keys.add(k)
    return keys


def load_overview_htr_keys() -> set[str]:
    """Sample-eligible keys from overview: have Opus output and are not manual."""
    eligible: set[str] = set()
    if not OVERVIEW_CSV.exists():
        return eligible
    with OVERVIEW_CSV.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("manually_transcribed") or "").upper() == "TRUE":
                continue
            if not (r.get("HTR_OPUS_path") or "").strip():
                continue
            k = normalize_key(r.get("Ms_key", ""))
            if k:
                eligible.add(k)
    return eligible


def load_already_queued_keys() -> tuple[set[str], list[Path]]:
    queued: set[str] = set()
    files: list[Path] = []
    if not PRIORITY_DIR.exists():
        return queued, files
    for p in sorted(PRIORITY_DIR.glob(PRIORITY_GLOB)):
        files.append(p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  (skip) {p.name}: {exc}")
            continue
        items = data if isinstance(data, list) else data.get("tasks", [])
        for task in items:
            d = task.get("data", {}) if isinstance(task, dict) else {}
            for field in ("ms_key", "Ms_key", "letter_id", "name", "key", "id"):
                v = d.get(field)
                if v:
                    queued.add(normalize_key(str(v)))
                    break
    return queued, files


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"  [auto-queue] missing {ZIP_PATH.relative_to(REPO)}; skipping.")
        return 0
    if not OVERVIEW_CSV.exists():
        print(f"  [auto-queue] missing {OVERVIEW_CSV.relative_to(REPO)}; skipping.")
        return 0
    if not BUILDER.exists():
        print(f"  [auto-queue] missing {BUILDER.relative_to(REPO)}; skipping.")
        return 0

    sample = load_sample_keys()
    overview_eligible = load_overview_htr_keys()
    queued, files = load_already_queued_keys()

    sample_htr = sample & overview_eligible
    new_keys = sorted(sample_htr - queued)

    print(f"  sample letters             : {len(sample)}")
    print(f"  of those, HTR'd by Opus    : {len(sample_htr)}")
    print(f"  already in LS priority     : {len(sample_htr & queued)} "
          f"(across {len(files)} file(s))")
    print(f"  NEW to queue this run      : {len(new_keys)}")

    if not new_keys:
        print("  [auto-queue] nothing new to queue.")
        return 0

    PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
    TMP_IDS_TXT.parent.mkdir(parents=True, exist_ok=True)
    TMP_IDS_TXT.write_text("\n".join(new_keys) + "\n", encoding="utf-8")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = PRIORITY_DIR / f"labelstudio_sample_priority_auto_{ts}.json"

    print(f"  [auto-queue] invoking builder -> {out_path.relative_to(REPO)}")
    cmd = [
        sys.executable, str(BUILDER),
        "--priority-ids", str(TMP_IDS_TXT),
        "--out",          str(out_path),
    ]
    rc = subprocess.run(cmd, cwd=REPO, check=False).returncode
    if rc != 0:
        print(f"  [auto-queue] builder exited with code {rc}")
        return rc
    if out_path.exists():
        try:
            tasks = json.loads(out_path.read_text(encoding="utf-8"))
            n_tasks = len(tasks) if isinstance(tasks, list) else len(tasks.get("tasks", []))
        except Exception:
            n_tasks = -1
        print(f"  [auto-queue] wrote {n_tasks} LS task(s) to "
              f"{out_path.relative_to(REPO)}")
    else:
        print(f"  [auto-queue] WARNING: builder returned 0 but {out_path.name} "
              "was not created (no HTR-eligible rows?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
