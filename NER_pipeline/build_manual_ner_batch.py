"""
Build the next NER batch from `data/manually_transcribed_letters/`
=================================================================

For every manually-transcribed letter (`data/manually_transcribed_letters/*.txt`)
that has not yet been NER-annotated in batch 207 (NER project export files
`data/labelstudio_output/NER_{annotator}_207.json`), this script:

    1. reads the plain-text transcription;
    2. attaches metadata (date / sender / receiver) from
       `data/letters_overview.csv`;
    3. builds one Label Studio NER import task per letter, following the
       schema of the previous NER batches (`data.letter_id`, `data.text`,
       `data.meta_date`, `data.meta_sender`, `data.meta_receiver`);
    4. deterministically samples 10 letters as an intermediary IAA set
       (all 4 annotators get these same 10);
    5. distributes the remaining letters across the 4 annotators
       (A1, A2, A3, A4) balancing on total text length;
    6. writes a manifest, the IAA-10 key list, and per-letter tasks.

The output layout mirrors previous batches so `gliner_predict.py` can be
run against the per-letter task folder next:

    data/labelstudio_upload/NER_manual_batch2/
      manifest.csv               <- ms_key, assignment, tome, text length
      iaa10_keys.txt             <- the 10 IAA ms_keys, one per line
      README.md                  <- run summary
      tasks/                     <- raw *_ls_import.json (pre-GLiNER)

Usage (from repo root):

    python NER_pipeline/build_manual_ner_batch.py
    python NER_pipeline/build_manual_ner_batch.py --iaa-size 10 --seed 20260702
    python NER_pipeline/build_manual_ner_batch.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANUAL_DIR   = ROOT / "data" / "manually_transcribed_letters"
NER_207_DIR  = ROOT / "data" / "labelstudio_output"
OVERVIEW_CSV = ROOT / "data" / "letters_overview.csv"

ANNOTATORS = ["A1", "A2", "A3", "A4"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def stem_to_ms(stem: str) -> str | None:
    """`Ms. II 70_1_100` -> `Ms. II 70/1/100`. Returns None on unparseable stems."""
    m = re.match(r"^(Ms\. II 70)_(.+)$", stem)
    if not m:
        return None
    return f"{m.group(1)}/" + m.group(2).replace("_", "/")


def is_valid_ms(ms: str) -> bool:
    """True iff the shelfmark has exactly `Ms. II 70/<tome>/<folio>` shape."""
    return bool(re.match(r"^Ms\. II 70/[^/]+/[^/]+$", ms))


def load_ner207_ms_keys(export_dir: Path) -> set[str]:
    """Union of `letter_id` (= ms_key) across all 4 NER-207 exports."""
    out: set[str] = set()
    for who in ANNOTATORS:
        f = export_dir / f"NER_{who}_207.json"
        if not f.exists():
            continue
        for t in json.loads(f.read_text(encoding="utf-8")):
            mk = ((t.get("data") or {}).get("letter_id") or "").strip()
            if mk:
                out.add(mk)
    return out


def load_metadata(overview_csv: Path) -> dict[str, dict]:
    """Return `ms_key -> {date, source, target, idn}` from letters_overview.csv."""
    lookup: dict[str, dict] = {}
    with overview_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ms = (row.get("Ms_key") or "").strip()
            if not ms:
                continue
            lookup[ms] = {
                "date":   (row.get("Date")   or "").strip(),
                "source": (row.get("Source") or "").strip(),
                "target": (row.get("Target") or "").strip(),
                "idn":    (row.get("IDN")    or "").strip(),
            }
    return lookup


def read_letter_text(path: Path) -> str:
    """Read a manually-transcribed letter file as UTF-8 (BOM-tolerant)."""
    return path.read_text(encoding="utf-8-sig").strip()


def illegibility_ratio(text: str) -> float:
    """Fraction of characters inside `[XXX]` gap markers."""
    total = max(len(text), 1)
    inside = sum(len(m) for m in re.findall(r"\[XXX\]", text))
    return inside / total


def tome_of(ms: str) -> str:
    """`Ms. II 70/1/100` -> `1` (best-effort; empty if it doesn't fit the pattern)."""
    parts = ms.split("/")
    return parts[1] if len(parts) >= 3 else ""


# --------------------------------------------------------------------------
# sampling + distribution
# --------------------------------------------------------------------------

def pick_iaa_sample(rows: list[dict], size: int, seed: int) -> list[str]:
    """Stratify by tome to get a balanced IAA sample, deterministic given seed.

    Simplest sensible approach: split rows by tome, allocate slots proportionally
    (rounding up), then pull them out in tome-order until we have `size` letters.
    Length is not used — the pool is already dominated by 1-2 page letters.
    """
    rng = random.Random(seed)
    by_tome: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tome[tome_of(r["ms_key"])].append(r)
    for t in by_tome.values():
        rng.shuffle(t)

    # Proportional allocation, then top-up until we hit `size`.
    total = len(rows)
    quotas: dict[str, int] = {}
    for tome, items in by_tome.items():
        quotas[tome] = max(1, round(size * len(items) / total))
    # Adjust to sum exactly `size`.
    while sum(quotas.values()) > size:
        tome_max = max(quotas, key=quotas.get)
        quotas[tome_max] -= 1
    while sum(quotas.values()) < size:
        tome_min = min(quotas, key=lambda t: quotas[t] / max(1, len(by_tome[t])))
        quotas[tome_min] += 1

    picked: list[str] = []
    for tome, n in quotas.items():
        picked.extend(r["ms_key"] for r in by_tome[tome][:n])
    return sorted(picked)


def distribute_remaining(rows: list[dict], annotators: list[str]) -> dict[str, str]:
    """Longest-processing-time-first split. Balances total text length across
    annotators. Returns `ms_key -> annotator`."""
    order = sorted(rows, key=lambda r: r["text_len"], reverse=True)
    loads = {a: 0 for a in annotators}
    assignment: dict[str, str] = {}
    for r in order:
        pick = min(annotators, key=lambda a: loads[a])
        assignment[r["ms_key"]] = pick
        loads[pick] += r["text_len"]
    return assignment


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iaa-size", type=int, default=10,
                   help="How many letters to sample for the shared IAA task (default 10).")
    p.add_argument("--seed", type=int, default=20260702,
                   help="Random seed for the IAA sample (default: today's date).")
    p.add_argument("--min-chars", type=int, default=200,
                   help="Skip letters with fewer than this many chars (default 200).")
    p.add_argument("--max-illegible", type=float, default=0.30,
                   help="Skip letters where >this fraction of the text is `[XXX]` "
                        "(default 0.30).")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "labelstudio_upload" / "NER_manual_batch2",
                   help="Where to write manifest + tasks/ folder.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print counts only; don't write anything.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not MANUAL_DIR.exists():
        print(f"ERROR: {MANUAL_DIR} missing", file=sys.stderr)
        return 1
    if not OVERVIEW_CSV.exists():
        print(f"ERROR: {OVERVIEW_CSV} missing", file=sys.stderr)
        return 1

    already_ner = load_ner207_ms_keys(NER_207_DIR)
    metadata    = load_metadata(OVERVIEW_CSV)
    print(f"Already annotated in NER 207 : {len(already_ner)} ms_keys")
    print(f"Metadata rows loaded         : {len(metadata)}")

    # Build the pool.
    rows: list[dict] = []
    skipped = defaultdict(list)
    for path in sorted(MANUAL_DIR.glob("*.txt")):
        ms = stem_to_ms(path.stem)
        if not ms:
            skipped["bad_stem"].append(path.name)
            continue
        if not is_valid_ms(ms):
            skipped["bad_shape"].append(ms)
            continue
        if ms in already_ner:
            skipped["already_ner_207"].append(ms)
            continue
        text = read_letter_text(path)
        if len(text) < args.min_chars:
            skipped["too_short"].append(ms)
            continue
        ratio = illegibility_ratio(text)
        if ratio > args.max_illegible:
            skipped["too_illegible"].append(ms)
            continue
        meta = metadata.get(ms, {})
        if not meta:
            skipped["no_metadata"].append(ms)
        rows.append({
            "ms_key":   ms,
            "path":     path,
            "text":     text,
            "text_len": len(text),
            "tome":     tome_of(ms),
            "meta":     meta,
        })

    print(f"\nPool of eligible letters     : {len(rows)}")
    print("Skipped:")
    for reason, items in skipped.items():
        print(f"  {reason:>16}: {len(items)}"
              + (f"  e.g. {items[0]}" if items else ""))

    if len(rows) < args.iaa_size + len(ANNOTATORS):
        print(f"ERROR: pool too small ({len(rows)}) for IAA-{args.iaa_size} "
              f"+ {len(ANNOTATORS)} annotators.", file=sys.stderr)
        return 1

    # --- IAA-10 sample (all annotators get the same letters) ---
    iaa_keys = pick_iaa_sample(rows, args.iaa_size, args.seed)
    iaa_set  = set(iaa_keys)
    unique_pool = [r for r in rows if r["ms_key"] not in iaa_set]

    # --- distribute the rest ---
    assignment = distribute_remaining(unique_pool, ANNOTATORS)

    # Report.
    per_annot_counts = defaultdict(int)
    per_annot_chars  = defaultdict(int)
    for r in unique_pool:
        who = assignment[r["ms_key"]]
        per_annot_counts[who] += 1
        per_annot_chars[who]  += r["text_len"]

    print(f"\nIAA-{args.iaa_size} sample (shared, seed={args.seed}):")
    for k in iaa_keys:
        row = next(r for r in rows if r["ms_key"] == k)
        print(f"  {k}   tome {row['tome']:>3}   chars {row['text_len']:>5}")

    print(f"\nAssignment of the remaining {len(unique_pool)} letters:")
    for who in ANNOTATORS:
        print(f"  {who:>8}: {per_annot_counts[who]:>3} letters, "
              f"{per_annot_chars[who]:>7} chars total")
    print(f"\nEach annotator will therefore work on "
          f"{args.iaa_size} + <their share> = "
          + " / ".join(f"{args.iaa_size + per_annot_counts[w]}" for w in ANNOTATORS)
          + " letters.")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    # --- write output ---
    out = args.output_dir
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for r in rows:
        ms = r["ms_key"]
        assign = "IAA" if ms in iaa_set else assignment[ms]
        task = {
            "data": {
                "letter_id":     ms,
                "text":          r["text"],
                "meta_date":     r["meta"].get("date", ""),
                "meta_sender":   r["meta"].get("source", ""),
                "meta_receiver": r["meta"].get("target", ""),
                "image_url":     "",
                "nodegoat_id":   r["meta"].get("idn", ""),
                "source_folder": "manually_transcribed_letters",
                "batch":         "NER_manual_batch2",
                "assignment":    assign,
            }
        }
        fname = ms.replace("/", "_") + "_ls_import.json"
        (tasks_dir / fname).write_text(
            json.dumps([task], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_rows.append({
            "ms_key":         ms,
            "assignment":     assign,
            "tome":           r["tome"],
            "text_len":       r["text_len"],
            "meta_date":      r["meta"].get("date", ""),
            "meta_sender":    r["meta"].get("source", ""),
            "meta_receiver":  r["meta"].get("target", ""),
            "source_file":    r["path"].name,
        })

    # manifest.csv
    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # IAA keys (plain list, one per line — easy to diff and to reference)
    (out / "iaa10_keys.txt").write_text(
        "\n".join(iaa_keys) + "\n", encoding="utf-8"
    )

    # README with the run parameters
    readme_lines = [
        "# NER manual batch 2 (intermediary IAA + distributed annotation)",
        "",
        f"Built from `data/manually_transcribed_letters/` on seed `{args.seed}`.",
        "",
        "## Pool",
        f"- eligible letters       : **{len(rows)}**",
        f"- IAA (shared, all 4)    : **{len(iaa_keys)}**  (see `iaa10_keys.txt`)",
        f"- distributed unique     : **{len(unique_pool)}**",
        "",
        "## Filters applied",
        f"- min chars              : {args.min_chars}",
        f"- max `[XXX]` ratio      : {args.max_illegible}",
        f"- excluded already in NER 207 : {len(skipped['already_ner_207'])}",
        f"- excluded bad stems     : {len(skipped['bad_stem']) + len(skipped['bad_shape'])}",
        "",
        "## Per-annotator load (unique letters, chars)",
    ]
    for who in ANNOTATORS:
        readme_lines.append(
            f"- {who:>8}: {per_annot_counts[who]:>3} letters,  "
            f"{per_annot_chars[who]:>7} chars"
        )
    readme_lines += [
        "",
        "## Next step",
        "```",
        "python NER_pipeline/gliner_predict.py \\",
        f"    --input-dir {tasks_dir.as_posix()} \\",
        f"    --output-dir data/preannots_GLiNER2/NER_manual_batch2 \\",
        "    --combined-name gliner_preannots.json",
        "```",
        "",
        "After that, split the pre-annotated tasks into 4 per-annotator files "
        "(`NER_manual_batch2_{who}.json`) + one shared IAA file "
        "(`NER_manual_batch2_iaa10.json`) using the same `assignment` field "
        "carried on every task.",
    ]
    (out / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {manifest_path}")
    print(f"  {out / 'iaa10_keys.txt'}")
    print(f"  {out / 'README.md'}")
    print(f"  {tasks_dir}  ({len(rows)} *_ls_import.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
