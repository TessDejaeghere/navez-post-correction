#!/usr/bin/env python3
"""
Build a master overview of letters going to Label Studio for correction.

A letter is in scope if at least one of these is true:
  - it has a manual transcription   .txt file in `data/manually_transcribed_letters/`
  - it was re-transcribed by HTR_OPUS in `data/htr_opus_outputs/<batch>/`

Linking is done via the nodegoat export `data/nodegoat_export_2106.csv`:
  - Ms_key  ('Name'  column)        -> manual transcription stems
  - IDN     ('[Archive] IDN' col)   -> KBR ID used by HTR_OPUS file names

`HTR_corrected` is auto-set to TRUE when:
  - the letter has a manual transcription, OR
  - it appears (with a non-cancelled, non-empty `transcription` result) in any
    Label Studio export under `data/labelstudio_upload/HTR_corrections_finished/`.
  Any TRUE flips from a previous CSV are preserved across rebuilds.

Output (always rewritten):
  data/letters_overview.csv

Run again any time a new HTR_OPUS batch is added or the manual set changes.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NODEGOAT_CSV  = REPO / "data" / "nodegoat_export_2106.csv"
MANUAL_DIR    = REPO / "data" / "manually_transcribed_letters"
HTR_OPUS_ROOT = REPO / "data" / "htr_opus_outputs"
HTR_FINISHED_DIR = REPO / "data" / "labelstudio_upload" / "HTR_corrections_finished"
OVERVIEW_CSV  = REPO / "data" / "letters_overview.csv"
UNMATCHED_CSV = REPO / "data" / "letters_unmatched_opus.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stem_to_ms_key(stem: str) -> str:
    """`Ms. II 70_1_100`      -> `Ms. II 70/1/100`
    `Ms. II 70_6_sup._79bis` -> `Ms. II 70/6/sup./79bis`
    """
    prefix = "Ms. II 70"
    if not stem.startswith(prefix):
        return stem
    rest = stem[len(prefix):].lstrip("_")
    return f"{prefix}/{rest.replace('_', '/')}"


def load_nodegoat(path: Path) -> list[dict[str, str]]:
    """Read all rows. CSV has two 'Name' columns — keep only the first."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Dedupe header by suffixing repeats: Name -> Name, Name -> Name_2
        seen: dict[str, int] = {}
        unique_header: list[str] = []
        for col in header:
            seen[col] = seen.get(col, 0) + 1
            unique_header.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
        rows = []
        for row in reader:
            if len(row) < len(unique_header):
                row += [""] * (len(unique_header) - len(row))
            rows.append(dict(zip(unique_header, row)))
    return rows


def find_manual_ms_keys(manual_dir: Path) -> set[str]:
    return {stem_to_ms_key(p.stem) for p in manual_dir.glob("*.txt")}


def find_opus_outputs(htr_root: Path) -> dict[str, list[tuple[str, Path]]]:
    """{KBR_id: [(batch_name, txt_path), ...]} — list because the same letter
    could in principle be re-run in multiple batches."""
    out: dict[str, list[tuple[str, Path]]] = {}
    if not htr_root.exists():
        return out
    for batch_dir in sorted(p for p in htr_root.iterdir() if p.is_dir()):
        for txt in sorted(batch_dir.glob("*.txt")):
            kbr_id = txt.stem
            if re.fullmatch(r"\d+", kbr_id):  # skip usage.jsonl etc.
                out.setdefault(kbr_id, []).append((batch_dir.name, txt))
    return out


def find_finished_htr_ms_keys(finished_dir: Path) -> set[str]:
    """Ms_keys whose task in any HTR_corrections_finished export has a
    non-cancelled annotation with a non-empty `transcription` TextArea result.
    """
    out: set[str] = set()
    if not finished_dir.exists():
        return out
    for p in sorted(finished_dir.glob("HTR_corrections_*.json")):
        try:
            export = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(export, list):
            continue
        for task in export:
            d = task.get("data") or {}
            ms_key = (d.get("ms_key") or d.get("Ms_key") or "").strip()
            if not ms_key:
                continue
            for ann in task.get("annotations") or []:
                if ann.get("was_cancelled"):
                    continue
                for r in ann.get("result", []) or []:
                    if r.get("from_name") != "transcription":
                        continue
                    txt = (r.get("value") or {}).get("text")
                    if isinstance(txt, list):
                        txt = txt[0] if txt else None
                    if isinstance(txt, str) and txt.strip():
                        out.add(ms_key)
                        break
                else:
                    continue
                break
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    assert NODEGOAT_CSV.exists(), f"Missing nodegoat CSV: {NODEGOAT_CSV}"
    assert MANUAL_DIR.exists(),    f"Missing manual transcriptions dir: {MANUAL_DIR}"

    nodegoat = load_nodegoat(NODEGOAT_CSV)
    manual_ms_keys = find_manual_ms_keys(MANUAL_DIR)
    opus_by_kbr    = find_opus_outputs(HTR_OPUS_ROOT)
    finished_htr_ms_keys = find_finished_htr_ms_keys(HTR_FINISHED_DIR)

    # Preserve HTR_corrected flips from the previous CSV so manual updates
    # (e.g. "this letter has been corrected in Label Studio") survive a rebuild.
    prev_corrected: set[tuple[str, str]] = set()  # keys: (Ms_key, IDN)
    if OVERVIEW_CSV.exists():
        with OVERVIEW_CSV.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("HTR_corrected", "").strip().upper() == "TRUE":
                    prev_corrected.add((row.get("Ms_key", ""), row.get("IDN", "")))

    columns = [
        "Ms_key",
        "nodegoat_ID",
        "Object_ID",
        "Source",     # sender
        "Target",     # receiver
        "Date",
        "Archive",
        "IDN",        # KBR id
        "manually_transcribed",
        "HTR_corrected",        # TRUE once the HTR has been human-reviewed
        "re_HTR",
        "HTR_OPUS_batch",
        "HTR_OPUS_path",
    ]

    rows_out: list[dict[str, str]] = []
    matched_kbr_ids: set[str] = set()

    for ng in nodegoat:
        ms_key = ng.get("Name", "").strip()
        idn    = ng.get("[Archive] IDN", "").strip()
        in_manual = ms_key in manual_ms_keys
        opus_hits = opus_by_kbr.get(idn, []) if idn else []
        if not (in_manual or opus_hits):
            continue
        if opus_hits:
            matched_kbr_ids.add(idn)
        # Default: a manual transcription counts as already corrected.
        # Otherwise honour either a finished HTR-correction export or whatever
        # was set in the previous CSV (manual flips survive rebuilds).
        corrected = (
            in_manual
            or ms_key in finished_htr_ms_keys
            or ((ms_key, idn) in prev_corrected)
        )
        rows_out.append({
            "Ms_key":               ms_key,
            "nodegoat_ID":          ng.get("nodegoat ID", ""),
            "Object_ID":            ng.get("Object ID", ""),
            "Source":               ng.get("Source", ""),
            "Target":               ng.get("Target", ""),
            "Date":                 ng.get("Date", ""),
            "Archive":              ng.get("[Archive] Name Archive", ""),
            "IDN":                  idn,
            "manually_transcribed": "TRUE" if in_manual else "FALSE",
            "HTR_corrected":        "TRUE" if corrected else "FALSE",
            "re_HTR":               "TRUE" if opus_hits else "FALSE",
            "HTR_OPUS_batch":       "; ".join(b for b, _ in opus_hits),
            "HTR_OPUS_path":        "; ".join(str(p.relative_to(REPO)) for _, p in opus_hits),
        })

    # Sort by Ms_key for stable diffs
    rows_out.sort(key=lambda r: r["Ms_key"])

    # Surface any Opus output that didn't match a nodegoat IDN (data-quality flag)
    unmatched_kbr = sorted(set(opus_by_kbr) - matched_kbr_ids)
    for kbr_id in unmatched_kbr:
        for batch, path in opus_by_kbr[kbr_id]:
            corrected = ("", kbr_id) in prev_corrected
            rows_out.append({
                "Ms_key":               "",
                "nodegoat_ID":          "",
                "Object_ID":            "",
                "Source":               "",
                "Target":               "",
                "Date":                 "",
                "Archive":              "",
                "IDN":                  kbr_id,
                "manually_transcribed": "FALSE",
                "HTR_corrected":        "TRUE" if corrected else "FALSE",
                "re_HTR":               "TRUE",
                "HTR_OPUS_batch":       batch,
                "HTR_OPUS_path":        str(path.relative_to(REPO)),
            })

    with OVERVIEW_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)

    # --- Separate CSV for Opus letters with no nodegoat counterpart ----------
    # These are HTR'd but not yet entered in the database — flagged so the
    # historian can add the missing metadata.
    unmatched_cols = [
        "IDN", "HTR_OPUS_batch", "HTR_OPUS_path",
        "n_chars", "preview", "txt_mtime",
    ]
    unmatched_rows: list[dict[str, str]] = []
    for kbr_id in unmatched_kbr:
        for batch, path in opus_by_kbr[kbr_id]:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            preview = " ".join(text.split())[:200]
            mtime = ""
            try:
                from datetime import datetime
                mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                pass
            unmatched_rows.append({
                "IDN":            kbr_id,
                "HTR_OPUS_batch": batch,
                "HTR_OPUS_path":  str(path.relative_to(REPO)),
                "n_chars":        str(len(text)),
                "preview":        preview,
                "txt_mtime":      mtime,
            })

    with UNMATCHED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unmatched_cols)
        writer.writeheader()
        writer.writerows(unmatched_rows)

    # --- Report -------------------------------------------------------------
    n_total  = len(rows_out)
    n_manual = sum(1 for r in rows_out if r["manually_transcribed"] == "TRUE")
    n_opus   = sum(1 for r in rows_out if r["re_HTR"] == "TRUE")
    n_both   = sum(1 for r in rows_out if r["re_HTR"] == "TRUE" and r["manually_transcribed"] == "TRUE")
    n_corr   = sum(1 for r in rows_out if r["HTR_corrected"] == "TRUE")
    n_orphan = len(unmatched_kbr)

    print(f"Wrote {OVERVIEW_CSV.relative_to(REPO)}  ({n_total} rows)")
    print(f"  manually transcribed       : {n_manual}")
    print(f"  HTR corrected (incl. manual): {n_corr}")
    print(f"  re-HTR'd by Opus           : {n_opus}")
    print(f"  in BOTH sets               : {n_both}")
    print(f"  HTR_OPUS without IDN match : {n_orphan}")
    if n_orphan:
        print("    ", ", ".join(unmatched_kbr))
    print(f"Wrote {UNMATCHED_CSV.relative_to(REPO)}  ({len(unmatched_rows)} rows)")


if __name__ == "__main__":
    main()
