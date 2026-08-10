#!/usr/bin/env python3
"""
Build Label Studio correction tasks for Opus-HTR'd letters.

For each letter that has been HTR'd by Opus and is NOT already in the manual
set, write one Label Studio task with:

  data:
    - claude_text       (the .txt produced by run_htr_opus.py)
    - image_links_html  (one button per page, linking to Google Drive)
    - letter_id, ms_key, n_pages, source, target, date, htr_opus_batch

  predictions[0].result:
    - 'Person Name'        spans (GLiNER multi v2.1)
    - 'Suspicious French'  spans (pygrammalecte)
    - 'Format Anomaly'     spans (chars outside the 19c-FR whitelist)
    - one 'textarea' pre-fill so the right-pane editor opens with the same text

Output: data/labelstudio_tasks_opus_correction.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# GLiNER2 prints emoji during model load; on Windows the default cp1252
# console will crash on those. Reconfigure stdout/stderr to UTF-8 early.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]

OVERVIEW_CSV   = REPO / "data" / "letters_overview.csv"
# Where extracted KBR scans live. Old letters were processed out of
# data/brieven_te_HTRen/ (now defunct); current HTR runs extract into
# data/tomes_KBR/_pages_*. We accept either, plus a fallback that extracts the
# matching zip on demand.
PAGES_ROOTS    = [
    REPO / "data" / "tomes_KBR",
    REPO / "data" / "brieven_te_HTRen",
]
ZIP_ROOTS      = [
    REPO / "data" / "tomes_KBR",
    REPO / "data" / "brieven_te_HTRen",
]
# Per-batch text files listing JPEG filenames (one per line). Used as a
# lightweight fallback when neither extracted _pages_<batch>/ nor <batch>.zip
# exist locally (e.g. when the scans live only in Google Drive). Extracted
# from data/jpeg_lists_archive.zip.
JPEG_LISTS_ROOT = REPO / "data" / "jpeg_lists"
DRIVE_MAP_JSON = REPO / "config" / "drive_file_ids.json"
TASKS_OUT      = REPO / "data" / "labelstudio_tasks_opus_correction.json"
PRIORITY_OUT   = (REPO / "data" / "labelstudio_upload" / "sample_priority"
                  / "labelstudio_sample_priority.json")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1U7jIuhzGyfyoShMGdJ6oTRTkx9GqCv2o?usp=sharing"


# ---------------------------------------------------------------------------
# Anomaly detectors
# ---------------------------------------------------------------------------

# Whitelist of chars that are 'normal' in 19c French letters + Claude's markers
# ([PS:], [Address:], [...], etc). Anything else gets flagged as Format Anomaly.
ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n\t\r"
    ".,;:!?\"'()[]{}-_/\\*&@#%+=<>|"
    "àâäæçèéêëîïñôöœùûüÿ"
    "ÀÂÄÆÇÈÉÊËÎÏÑÔÖŒÙÛÜŸ"
    "«»‹›\u201c\u201d\u2018\u2019\u201a\u201e`´"     # curly quotes
    "§°†‡€£¢¤¥•…"
    "–—‒"                                              # en/em/figure dashes
)


def find_weird_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) for each contiguous run of chars NOT in the whitelist."""
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] not in ALLOWED_CHARS:
            j = i
            while j < n and text[j] not in ALLOWED_CHARS:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def find_grammar_spans(text: str, include_spelling: bool = False
                       ) -> list[tuple[int, int, str]]:
    """Return (abs_start, abs_end, message) for each Grammalecte issue.

    We bypass `pygrammalecte` (which has a known invalid-JSON bug when some
    paragraphs have no errors) and call the underlying `grammalecte` package
    directly, paragraph by paragraph.

    By default we keep only *grammar* errors. Spelling errors are almost all
    false positives on 19c letters (proper names, archaic forms like `etoit`,
    place names, etc.) so they would drown the genuine signal.
    """
    try:
        import grammalecte
    except ImportError:
        print("  (grammalecte not installed; skipping Suspicious French)", file=sys.stderr)
        return []

    gc = grammalecte.GrammarChecker("fr")
    gc.gce.setOptions({"html": True, "latex": True, "apos": False})

    spans: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.split("\n"):
        if line.strip():
            try:
                grammar_errs, spelling_errs = gc.getParagraphErrors(line)
            except Exception as e:  # noqa: BLE001
                print(f"  grammalecte failed on a paragraph: {type(e).__name__}: {e}",
                      file=sys.stderr)
                grammar_errs, spelling_errs = (), ()
            errs = list(grammar_errs)
            if include_spelling:
                errs += list(spelling_errs)
            for err in errs:
                s = offset + int(err["nStart"])
                e = offset + int(err["nEnd"])
                if e > s:
                    spans.append((s, e, err.get("sMessage") or err.get("sValue", "")))
        offset += len(line) + 1  # +1 for the '\n'
    return spans


def find_person_spans(text: str, gliner_model, threshold: float = 0.55,
                      chunk_chars: int = 4000,
                      ) -> list[tuple[int, int, float]]:
    """Return (start, end, score) for each Person span.

    Works with either backend:
      - GLiNER2  (preferred): result['entities']['person'] is a list of
                 {'text','start','end',('confidence')} dicts.
      - GLiNER v1: model.predict_entities(...) returns a flat list with
                 'start','end','score' keys.

    Long letters are chunked on paragraph boundaries to avoid CPU OOM on the
    transformer encoder. Offsets are remapped back to the full-text positions.
    """
    if gliner_model is None or not text.strip():
        return []

    # Build chunks: greedy paragraph-aware, never exceeding `chunk_chars`.
    chunks: list[tuple[int, str]] = []  # (abs_start_offset, chunk_text)
    if len(text) <= chunk_chars:
        chunks.append((0, text))
    else:
        paras = text.split("\n\n")
        cur_start = 0
        cur_buf: list[str] = []
        cur_len = 0
        sep = "\n\n"
        idx = 0
        for para in paras:
            para_len = len(para) + (len(sep) if cur_buf else 0)
            if cur_buf and cur_len + para_len > chunk_chars:
                chunks.append((cur_start, sep.join(cur_buf)))
                cur_start = idx
                cur_buf = [para]
                cur_len = len(para)
            else:
                cur_buf.append(para)
                cur_len += para_len
            idx += len(para) + len(sep)
        if cur_buf:
            chunks.append((cur_start, sep.join(cur_buf)))

    spans: list[tuple[int, int, float]] = []
    for offset, chunk in chunks:
        try:
            if hasattr(gliner_model, "extract_entities"):
                res = gliner_model.extract_entities(
                    chunk, ["person"],
                    include_spans=True,
                    include_confidence=True,
                    threshold=threshold,
                )
                ents = (res.get("entities") or {}).get("person", []) or []
                spans.extend(
                    (offset + int(e["start"]), offset + int(e["end"]),
                     float(e.get("confidence", 0.0)))
                    for e in ents if "start" in e and "end" in e
                )
            else:
                ents = gliner_model.predict_entities(chunk, ["Person"], threshold=threshold)
                spans.extend(
                    (offset + int(e["start"]), offset + int(e["end"]),
                     float(e.get("score", 0.0)))
                    for e in ents
                )
        except Exception as e:  # noqa: BLE001
            print(f"  person-NER failed on a chunk (len={len(chunk)}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    return spans


# ---------------------------------------------------------------------------
# Drive linking
# ---------------------------------------------------------------------------

def _scan_pages_dirs(letter_id: str) -> list[tuple[int, str]]:
    """Scan every known PAGES_ROOTS / _pages_*/ for JPEGs of `letter_id`."""
    out: list[tuple[int, str]] = []
    for root in PAGES_ROOTS:
        if not root.exists():
            continue
        for pages_dir in root.glob("_pages_*"):
            for jpeg in pages_dir.glob(f"*_{letter_id}_*"):
                parts = jpeg.stem.split("_")
                try:
                    page = int(parts[8])
                except (IndexError, ValueError):
                    page = 0
                out.append((page, jpeg.name))
    return out


def _extract_zip_for_batch(batch: str) -> Path | None:
    """If a zip named <batch>.zip exists under any ZIP_ROOTS, extract it to
    <zip.parent>/_pages_<batch>/ and return that path. Else None."""
    if not batch:
        return None
    for root in ZIP_ROOTS:
        zip_path = root / f"{batch}.zip"
        if zip_path.exists():
            import zipfile
            target = root / f"_pages_{batch}"
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            print(f"    (extracted {zip_path.relative_to(REPO)} -> {target.relative_to(REPO)})",
                  file=sys.stderr)
            return target
    return None


def _filenames_from_jpeg_list(letter_id: str, batch: str) -> list[tuple[int, str]]:
    """Read JPEG filenames for `letter_id` from data/jpeg_lists/<batch>/<batch>_jpegs.txt.

    The txt file has one filename per line (KBR convention
    `BE-KBR00_{IDN}_..._{NNNN}_{INTERNAL}.jpeg`). We only return entries whose
    IDN component matches `letter_id`.
    """
    if not batch:
        return []
    list_path = JPEG_LISTS_ROOT / batch / f"{batch}_jpegs.txt"
    if not list_path.exists():
        return []
    out: list[tuple[int, str]] = []
    needle = f"_{letter_id}_"
    for line in list_path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or needle not in name:
            continue
        parts = Path(name).stem.split("_")
        try:
            page = int(parts[8])
        except (IndexError, ValueError):
            page = 0
        out.append((page, name))
    return out


def page_filenames_for_letter(letter_id: str, batch: str = "") -> list[str]:
    """All KBR scan filenames for `letter_id`, sorted by page index.

    Resolution order:
      1. Existing extracted `_pages_<batch>/` directories.
      2. On-demand extraction of `<batch>.zip` under tomes_KBR.
      3. Lightweight `data/jpeg_lists/<batch>/<batch>_jpegs.txt` listings
         (filenames only; no JPEG bytes needed locally).
    """
    out = _scan_pages_dirs(letter_id)
    if not out and batch:
        if _extract_zip_for_batch(batch) is not None:
            out = _scan_pages_dirs(letter_id)
    if not out and batch:
        out = _filenames_from_jpeg_list(letter_id, batch)
    seen: set[str] = set()
    unique: list[str] = []
    for _, name in sorted(out):
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _btn(href: str, label: str, primary: bool = True) -> str:
    bg = "#007bff" if primary else "#6c757d"
    return (f"<a href='{href}' target='_blank' "
            f"style='background:{bg};color:white;padding:8px 12px;"
            f"border-radius:4px;text-decoration:none;font-weight:bold;"
            f"font-size:13px;margin-left:6px;white-space:nowrap;'>{label}</a>")


def build_image_links_html(filenames: list[str], drive_map: dict[str, str]) -> str:
    """One button that opens the shared Drive folder.

    `filenames` and `drive_map` are kept on the signature for backward
    compatibility but are no longer used: annotators look up the JPEG by name
    inside the folder (the filenames are displayed in the XML banner via
    `page_filenames_str`).
    """
    del filenames, drive_map  # unused
    return _btn(DRIVE_FOLDER_URL, "Open Drive folder ↗", primary=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-map",   type=Path, default=DRIVE_MAP_JSON)
    ap.add_argument("--out",         type=Path, default=TASKS_OUT)
    ap.add_argument("--gliner-model", default="fastino/gliner2-multi-v1",
                    help="HF model id. Default uses GLiNER2 (fastino/gliner2-multi-v1). "
                         "Set to e.g. 'urchade/gliner_multi-v2.1' to fall back to GLiNER v1.")
    ap.add_argument("--gliner-threshold", type=float, default=0.55)
    ap.add_argument("--skip-gliner",  action="store_true")
    ap.add_argument("--skip-grammar", action="store_true")
    ap.add_argument("--include-spelling", action="store_true",
                    help="Also flag spelling errors (noisy on 19c text; off by default).")
    ap.add_argument("--max-letters",  type=int, default=None,
                    help="Only build the first N tasks (for smoke testing).")
    ap.add_argument("--batch-prefix", default=None,
                    help="Only include letters whose HTR_OPUS_batch starts with this "
                         "prefix (e.g. 'tome_1_boite_2' to exclude boite_1 letters "
                         "already assigned to annotators).")
    ap.add_argument("--priority-ids", type=Path, default=None,
                    help="Path to a text file with one Ms_key per line (e.g. "
                         "data/sample_letters/sample_still_to_htr.txt). When set, "
                         "the builder ignores --batch-prefix and only emits tasks "
                         "for those IDs that have already been HTR'd. Default output "
                         "is routed to data/labelstudio_upload/sample_priority/.")
    args = ap.parse_args()

    # Resolve paths so relative inputs from any cwd still format cleanly.
    args.drive_map = args.drive_map.resolve()
    # If running in priority mode and the user didn't override --out, route to
    # the sample_priority folder.
    if args.priority_ids and args.out == TASKS_OUT:
        args.out = PRIORITY_OUT
    args.out       = args.out.resolve()

    # --- Drive map ---------------------------------------------------------
    if args.drive_map.exists():
        drive_map: dict[str, str] = json.loads(args.drive_map.read_text(encoding="utf-8"))
        print(f"Drive map  : {len(drive_map)} files   ({args.drive_map.relative_to(REPO)})")
    else:
        drive_map = {}
        print(f"Drive map  : NONE (links will point to the folder)   "
              f"missing: {args.drive_map.relative_to(REPO)}")

    # --- Priority IDs (optional) ------------------------------------------
    priority_norm: set[str] | None = None
    priority_raw: list[str] = []
    if args.priority_ids:
        args.priority_ids = args.priority_ids.resolve()
        if not args.priority_ids.exists():
            sys.exit(f"--priority-ids file not found: {args.priority_ids}")
        priority_raw = [l.strip() for l in args.priority_ids.read_text(
                            encoding="utf-8").splitlines() if l.strip()]
        # Drop literal 'nan' lines (artefact of pandas-to-text exports).
        priority_raw = [x for x in priority_raw if x.lower() != "nan"]
        # Normalise separators so 'Ms. II 70_1_130' matches 'Ms. II 70/1/130'.
        # Lines that are pure digit strings are treated as KBR IDNs and matched
        # against the IDN column instead of Ms_key (this lets us include letters
        # with a missing Ms_key but a known IDN+HTR output).
        priority_idns: set[str] = {x for x in priority_raw if x.isdigit()}
        priority_norm: set[str] = {x.replace("_", "/") for x in priority_raw
                                   if not x.isdigit()}
        print(f"Priority   : {len(priority_norm)} Ms_keys + {len(priority_idns)} IDNs "
              f"from {args.priority_ids.relative_to(REPO)}")
        if args.batch_prefix:
            print("  (ignoring --batch-prefix because --priority-ids is set)")
    else:
        priority_idns = set()

    # --- Select letters ----------------------------------------------------
    rows: list[dict] = []
    seen_priority: set[str] = set()
    seen_idns: set[str] = set()
    pending_priority: list[dict] = []  # priority IDs not yet HTR-eligible
    with OVERVIEW_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ms_norm = (r.get("Ms_key") or "").replace("_", "/")
            idn = (r.get("IDN") or "").strip()
            if priority_norm is not None or priority_idns:
                match_ms  = ms_norm in (priority_norm or set())
                match_idn = idn in priority_idns
                if not (match_ms or match_idn):
                    continue
                if match_ms:
                    seen_priority.add(ms_norm)
                if match_idn:
                    seen_idns.add(idn)
                if r["manually_transcribed"] == "TRUE" or not r.get("HTR_OPUS_path"):
                    pending_priority.append(r)
                    continue
                rows.append(r)
            else:
                if r["re_HTR"] == "TRUE" and r["manually_transcribed"] != "TRUE":
                    if args.batch_prefix and not r["HTR_OPUS_batch"].startswith(args.batch_prefix):
                        continue
                    rows.append(r)
    if args.max_letters:
        rows = rows[: args.max_letters]

    if priority_norm is not None or priority_idns:
        missing_ms  = (priority_norm or set()) - seen_priority
        missing_idn = priority_idns - seen_idns
        missing = missing_ms | missing_idn
        print(f"Letters    : {len(rows)} eligible (HTR'd, not manually transcribed)")
        print(f"  pending : {len(pending_priority)} priority IDs still need HTR "
              f"or are manually transcribed (skipped)")
        if missing:
            print(f"  unknown : {len(missing)} priority IDs not found in overview CSV "
                  "(typo? not yet imported?)")
            for m in sorted(missing)[:5]:
                print(f"            - {m}")
            if len(missing) > 5:
                print(f"            ... ({len(missing)-5} more)")
        print()
    else:
        scope = f" (batch prefix '{args.batch_prefix}')" if args.batch_prefix else ""
        print(f"Letters    : {len(rows)} (Opus only, not manually transcribed){scope}\n")

    # --- GLiNER / GLiNER2 --------------------------------------------------
    gliner_model = None
    if not args.skip_gliner:
        is_gliner2 = args.gliner_model.startswith("fastino/gliner2") or "gliner2" in args.gliner_model.lower()
        try:
            if is_gliner2:
                from gliner2 import GLiNER2
                print(f"Loading GLiNER2: {args.gliner_model} (first run downloads weights)...")
                gliner_model = GLiNER2.from_pretrained(args.gliner_model)
            else:
                from gliner import GLiNER
                print(f"Loading GLiNER v1: {args.gliner_model} (first run downloads weights)...")
                gliner_model = GLiNER.from_pretrained(args.gliner_model)
            print("  Name model ready.\n")
        except Exception as e:
            print(f"  Name-model load failed ({type(e).__name__}: {e}); continuing "
                  f"without name highlights.\n", file=sys.stderr)

    # --- Build tasks -------------------------------------------------------
    tasks: list[dict] = []
    counts: dict[str, int] = defaultdict(int)

    for r in rows:
        kbr  = r["IDN"]
        path = r["HTR_OPUS_path"].split(";")[0].strip()
        if not path:
            continue
        txt_path = REPO / path
        if not txt_path.exists():
            print(f"  missing file: {txt_path}", file=sys.stderr)
            continue
        text = txt_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            continue

        results: list[dict] = []

        # Format Anomaly (regex)
        for s, e in find_weird_spans(text):
            results.append({
                "id": f"fa_{s}_{e}",
                "from_name": "anomaly_labels",
                "to_name":   "reference_text",
                "type":      "labels",
                "value": {"start": s, "end": e, "text": text[s:e],
                          "labels": ["Format Anomaly"]},
            })
            counts["format_anomaly"] += 1

        # Suspicious French (pygrammalecte)
        if not args.skip_grammar:
            for s, e, _msg in find_grammar_spans(text, include_spelling=args.include_spelling):
                results.append({
                    "id": f"sf_{s}_{e}",
                    "from_name": "anomaly_labels",
                    "to_name":   "reference_text",
                    "type":      "labels",
                    "value": {"start": s, "end": e, "text": text[s:e],
                              "labels": ["Suspicious French"]},
                })
                counts["suspicious_french"] += 1

        # Person Name (GLiNER)
        for s, e, _score in find_person_spans(text, gliner_model, args.gliner_threshold):
            results.append({
                "id": f"pn_{s}_{e}",
                "from_name": "anomaly_labels",
                "to_name":   "reference_text",
                "type":      "labels",
                "value": {"start": s, "end": e, "text": text[s:e],
                          "labels": ["Person Name"]},
            })
            counts["person_name"] += 1

        # Pre-fill the editable TextArea with the same text.
        results.append({
            "from_name": "transcription",
            "to_name":   "reference_text",
            "type":      "textarea",
            "value": {"text": [text]},
        })

        page_filenames = page_filenames_for_letter(kbr, r.get("HTR_OPUS_batch", ""))
        tasks.append({
            "data": {
                "claude_text":       text,
                "image_links_html":  build_image_links_html(page_filenames, drive_map),
                "letter_id":         kbr,
                "ms_key":            r.get("Ms_key", ""),
                "n_pages":           len(page_filenames),
                "page_filenames_str": ", ".join(page_filenames) if page_filenames
                                       else "(no scans found locally)",
                "source":            r.get("Source", ""),
                "target":            r.get("Target", ""),
                "date":              r.get("Date", ""),
                "htr_opus_batch":    r.get("HTR_OPUS_batch", ""),
            },
            "predictions": [{
                "model_version": "preannotations_v1",
                "result":        results,
            }],
        })
        counts["letters"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.out.relative_to(REPO)}  ({len(tasks)} tasks)")
    print(f"  Person Name        : {counts['person_name']}")
    print(f"  Suspicious French  : {counts['suspicious_french']}")
    print(f"  Format Anomaly     : {counts['format_anomaly']}")


if __name__ == "__main__":
    main()
