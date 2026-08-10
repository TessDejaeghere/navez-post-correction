#!/usr/bin/env python3
"""
HTR a zip of KBR JPEG scans with Claude Opus 4.8 via OpenRouter.

Identical methodology to `notebooks/htr_claude_opus.ipynb`:
  - SYSTEM_PROMPT frozen from the pilot
  - 2048-px long-edge JPEG q=92, base64 data URI
  - temperature=0.0, model='anthropic/claude-opus-4.8'
  - one Claude call per *letter* (all pages of a letter passed as multi-image)
  - resumable: skips letters whose .txt already exists

Usage:
    python HTR_pipeline/run_htr_opus.py data/brieven_te_HTRen/tome_1_boite_1_50_100.zip
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFile
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

# KBR scans occasionally arrive with a few missing trailing bytes; let Pillow
# decode whatever is present instead of raising OSError.
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(__file__).resolve().parents[1]

MODEL_SLUG    = "anthropic/claude-opus-4.8"
MAX_IMAGE_DIM = 2048      # long-edge px; lower to 1536 if cost is a concern
PRICE_IN_PER_M  = 5.00
PRICE_OUT_PER_M = 25.00

# For the "already manually transcribed" guard (mirrors build_letters_overview.py)
NODEGOAT_CSV = REPO / "data" / "nodegoat_export_2106.csv"
MANUAL_DIR   = REPO / "data" / "manually_transcribed_letters"

# Post-run refresh hook: rebuild the letters_overview CSV, auto-queue any
# freshly-HTR'd sample letters for Label Studio correction, and rebuild the
# sample-pipeline status table so they stay in sync with what's on disk.
# Order matters: overview first (auto-queue reads it), then auto-queue, then
# status (so the status table reflects the new addendum file).
REFRESH_SCRIPTS = [
    REPO / "HTR_pipeline" / "build_letters_overview.py",
    REPO / "HTR_pipeline" / "auto_queue_sample_for_correction.py",
    REPO / "analysis_scripts" / "sample_pipeline_status.py",
]


def refresh_overview() -> None:
    """Run the overview + pipeline-status scripts; stream their output."""
    for script in REFRESH_SCRIPTS:
        if not script.exists():
            print(f"  [refresh] skip (missing): {script.relative_to(REPO)}")
            continue
        rel = script.relative_to(REPO)
        print(f"\n--- refresh: {rel} ---")
        try:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=REPO,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - safety net only
            print(f"  [refresh] {rel} failed: {type(exc).__name__}: {exc}")


SYSTEM_PROMPT = (
    "You are an expert archivist transcribing 19th-century French handwritten "
    "letters from the François-Joseph Navez correspondence (Belgian neo-classicist "
    "painter, 1787-1869). \n\n"
    "CRITICAL RULES:\n"
    "1. Preserve the EXACT orthography of the source. Do NOT modernise archaic "
    "forms (keep `étoit` as `étoit`, `avoit` as `avoit`, `tems` as `tems`). "
    "Do NOT archaize modern forms (do not change `était` to `étoit`). Whatever "
    "the writer wrote, copy it.\n"
    "2. Preserve line breaks as in the original.\n"
    "3. Preserve ligatures (`œ`, `æ`) and all diacritics.\n"
    "4. Do NOT translate. Do NOT summarise. Do NOT add commentary or notes.\n"
    "5. If a word is illegible, use [...] (three dots in brackets). Do not guess.\n"
    "6. Output ONLY the transcription. No preamble, no markdown, no JSON wrapper.\n\n"
    "PHYSICAL LAYOUT (bifolia):\n"
    "These letters are typically written on a single sheet of paper folded once "
    "(a bifolium = 4 written sides, like a small booklet). The scan order is "
    "NOT necessarily the reading order. In particular:\n"
    "- An ADDRESS PANEL (mostly blank, with a recipient name and a city, sometimes "
    "  postal stamps or seals) belongs at the END of the transcription as a block "
    "  prefixed `[Address:]`. Do NOT transcribe it as part of the letter body, "
    "  even if it appears on the first scan.\n"
    "- A POSTSCRIPT or text written sideways, in a margin, or crossed over earlier "
    "  text (cross-writing) belongs at the END of the body, prefixed `[PS:]` or "
    "  `[Margin:]`. Do NOT interleave it with the main text.\n"
    "- The main letter body should be transcribed in the order the writer "
    "  intended (greeting -> body -> closing -> signature), reconstructing reading "
    "  order across pages even if the scans are out of order.\n"
    "- If you cannot determine the intended order, transcribe each visible block "
    "  separately and label them `[Page 1]`, `[Page 2]`, etc.\n\n"
    "Output structure:\n"
    "  <main letter body, in reading order>\n"
    "  [PS: ...]        (only if there is a postscript)\n"
    "  [Margin: ...]    (only if there is marginalia / cross-writing)\n"
    "  [Address: ...]   (only if an address panel is visible)"
)


def user_prompt(n_pages: int) -> str:
    if n_pages == 1:
        return "Transcribe the attached letter image in full. Output only the transcription."
    return (
        f"This letter is attached as {n_pages} images, one per page, in reading order "
        f"(page 1 first). Transcribe ALL pages as ONE continuous text in reading order. "
        f"Do not repeat or label pages in the output. Output only the transcription."
    )


def image_to_data_uri(image_path: Path) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(im.size)
            im = im.resize(
                (int(im.size[0] * scale), int(im.size[1] * scale)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def parse_letter_page(fname: str) -> tuple[str, int]:
    """KBR filename: BE-KBR00_<letterID>_..._<pageIdx>_<scanHash>.jpeg"""
    parts = Path(fname).stem.split("_")
    letter_id = parts[1] if len(parts) > 1 else Path(fname).stem
    try:
        page = int(parts[8])
    except (IndexError, ValueError):
        page = 0
    return letter_id, page


def _stem_to_ms_key(stem: str) -> str:
    """`Ms. II 70_1_100` -> `Ms. II 70/1/100` (matches build_letters_overview.py)."""
    prefix = "Ms. II 70"
    if not stem.startswith(prefix):
        return stem
    rest = stem[len(prefix):].lstrip("_")
    return f"{prefix}/{rest.replace('_', '/')}"


def manually_transcribed_kbr_ids(nodegoat_csv: Path, manual_dir: Path) -> set[str]:
    """Return KBR IDs (= IDNs) whose Ms_key already has a manual transcription."""
    if not nodegoat_csv.exists() or not manual_dir.exists():
        return set()
    manual_keys = {_stem_to_ms_key(p.stem) for p in manual_dir.glob("*.txt")}
    if not manual_keys:
        return set()
    skip: set[str] = set()
    with nodegoat_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # CSV has two 'Name' columns — keep only the first (the Ms. signature).
        try:
            name_idx = header.index("Name")
            idn_idx  = header.index("[Archive] IDN")
        except ValueError:
            return set()
        for row in reader:
            if len(row) <= max(name_idx, idn_idx):
                continue
            ms_key = row[name_idx].strip()
            idn    = row[idn_idx].strip()
            if idn and ms_key in manual_keys:
                skip.add(idn)
    return skip


def transcribe(client: OpenAI, image_paths: list[Path]) -> tuple[str, dict]:
    content: list[dict] = [{"type": "text", "text": user_prompt(len(image_paths))}]
    for i, img in enumerate(image_paths, start=1):
        if len(image_paths) > 1:
            content.append({"type": "text", "text": f"--- Page {i} of {len(image_paths)} ---"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(img)}})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    for attempt in (1, 2, 3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_SLUG,
                messages=messages,
                temperature=0.0,
                max_tokens=8000,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage.model_dump() if resp.usage else {}
            return text.strip(), usage
        except (RateLimitError, APIConnectionError) as e:
            if attempt < 3:
                print(f"    retry in {8 * attempt}s ({type(e).__name__})")
                time.sleep(8 * attempt)
                continue
            raise
        except APIStatusError as e:
            if 500 <= e.status_code < 600 and attempt < 3:
                time.sleep(8 * attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path", type=Path, help="Path to a zip of KBR JPEG scans")
    ap.add_argument("--out-root",  type=Path, default=REPO / "data" / "htr_opus_outputs",
                    help="Outputs go to <out-root>/<zip-stem>/")
    ap.add_argument("--key-file",  type=Path, default=REPO / "config" / "key_openrouter.txt",
                    help="File containing the OpenRouter API key")
    ap.add_argument("--no-skip-manual", action="store_true",
                    help="Do NOT skip letters that are already in the manual set "
                         "(default: skip them, joined via nodegoat IDN <-> Ms_key).")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="Skip the post-run refresh of letters_overview.csv and "
                         "sample_pipeline_status.csv.")
    args = ap.parse_args()

    # Resolve paths so relative inputs from any cwd still print/format cleanly.
    args.zip_path = args.zip_path.resolve()
    args.out_root = args.out_root.resolve()
    args.key_file = args.key_file.resolve()

    assert args.zip_path.exists(),  f"Zip not found: {args.zip_path}"
    assert args.key_file.exists(),  f"OpenRouter key file not found: {args.key_file}"

    api_key = args.key_file.read_text(encoding="utf-8").strip()
    assert api_key.startswith("sk-or-"), "Key doesn't look like an OpenRouter key."

    batch_name = args.zip_path.stem
    pages_dir  = args.zip_path.parent / f"_pages_{batch_name}"
    out_dir    = args.out_root / batch_name
    usage_log  = out_dir / "usage.jsonl"

    pages_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Zip:        {args.zip_path.relative_to(REPO)}")
    print(f"Pages dir:  {pages_dir.relative_to(REPO)}")
    print(f"Out dir:    {out_dir.relative_to(REPO)}")
    print(f"Model:      {MODEL_SLUG}")

    # --- Extract + group pages by KBR letter ID ----------------------------
    with zipfile.ZipFile(args.zip_path) as zf:
        zf.extractall(pages_dir)

    images = sorted(p for p in pages_dir.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    print(f"Extracted {len(images)} images.")

    buckets: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in images:
        lid, page = parse_letter_page(p.name)
        buckets[lid].append((page, p))
    letters = {lid: [p for _, p in sorted(pages)] for lid, pages in buckets.items()}
    print(f"Grouped into {len(letters)} letters.")

    # Letters already manually transcribed (joined via nodegoat IDN <-> Ms_key).
    if args.no_skip_manual:
        manual_kbr: set[str] = set()
        print("Manual-skip:  DISABLED (--no-skip-manual)\n")
    else:
        manual_kbr = manually_transcribed_kbr_ids(NODEGOAT_CSV, MANUAL_DIR)
        n_in_batch = sum(1 for lid in letters if lid in manual_kbr)
        print(f"Manual-skip:  {len(manual_kbr)} KBR ids known, "
              f"{n_in_batch} of this batch will be skipped.\n")

    # --- HTR loop ----------------------------------------------------------
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/local/navez-htr",
            "X-Title": "Navez HTR - Opus 4.8 (CLI)",
        },
    )

    ok = skipped = failed = 0
    total_in = total_out = 0

    for letter_id, pages in sorted(letters.items()):
        out_path = out_dir / f"{letter_id}.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [skip] {letter_id} ({len(pages)}p) — exists")
            skipped += 1
            continue
        if letter_id in manual_kbr:
            print(f"  [skip] {letter_id} ({len(pages)}p) — already manually transcribed")
            skipped += 1
            continue
        t0 = time.time()
        try:
            text, usage = transcribe(client, pages)
        except Exception as e:
            print(f"  [FAIL] {letter_id} ({len(pages)}p): {type(e).__name__}: {e}")
            failed += 1
            continue
        out_path.write_text(text, encoding="utf-8")
        dt = time.time() - t0
        total_in  += usage.get("prompt_tokens", 0) or 0
        total_out += usage.get("completion_tokens", 0) or 0
        with usage_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "letter_id": letter_id,
                "n_pages":   len(pages),
                "model":     MODEL_SLUG,
                "seconds":   round(dt, 2),
                "chars_out": len(text),
                "usage":     usage,
            }) + "\n")
        ok += 1
        print(f"  [ok]  {letter_id} ({len(pages)}p): {len(text):>5} chars / "
              f"{usage.get('prompt_tokens', '?')}+{usage.get('completion_tokens', '?')} toks "
              f"in {dt:>5.1f}s")

    est_usd = total_in / 1_000_000 * PRICE_IN_PER_M + total_out / 1_000_000 * PRICE_OUT_PER_M
    print(f"\n=== Done ===")
    print(f"  ok: {ok}    skipped: {skipped}    failed: {failed}")
    print(f"  outputs:   {out_dir.relative_to(REPO)}")
    print(f"  usage log: {usage_log.relative_to(REPO)}")
    print(f"  tokens:    {total_in:,} in + {total_out:,} out")
    print(f"  est cost:  ${est_usd:.2f}  (exact: https://openrouter.ai/activity)")

    # Auto-refresh overview + pipeline status unless explicitly disabled.
    # Only worth doing if at least one new letter was written.
    if args.skip_refresh:
        print("\nRefresh skipped (--skip-refresh).")
    elif ok == 0:
        print("\nNo new outputs this run; skipping overview refresh.")
    else:
        refresh_overview()


if __name__ == "__main__":
    main()
