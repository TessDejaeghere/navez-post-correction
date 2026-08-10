"""Freeze the 561-letter NER gold pool + 3-fold stratified split.

Purpose
-------
One deterministic script that every downstream evaluation notebook reads
from.  Produces the two authoritative artifacts:

  1. ``NER_pipeline/eval/gold_561_merged.jsonl``
     One JSON row per unique letter, with the ART / ART_NOM labels merged
     into a single ``ART`` label (concatenation policy set for the paper).
     Fields: letter_id, annotator, sender, year, decade, text, gold_spans.

  2. ``NER_pipeline/eval/folds_writer_year.json``
     Stratified 3-fold assignment.  Stratum key = ``sender × decade`` with
     singleton fall-back to just ``decade`` and finally to a ``misc``
     bucket.  Each letter appears in exactly one fold's test partition.

Also prints a balance report to stdout so the fold quality is verifiable
before any expensive evaluation is launched.

Design decisions (see systems-paper discussion notes)
-----------------------------------------------------
* Gold pool = deduplicated across the 4 annotators using the tie-break
  order in ``gliner_baseline_eval.build_gold`` (A2 -> A1 -> A3
  -> A4).  Not re-scored here; this file just freezes what
  ``build_gold`` returns.
* Writer-stratified (NOT writer-disjoint) — deployment scenario is
  in-distribution corpus expansion within the same correspondence
  network.  Flagged in the paper's discussion.
* Year -> decade buckets (1810s, 1820s, ..., 1870s).
* No annotator stratification — accepted as a residual variance source
  and reported in the balance table so the reader can see it.
* Deterministic: seed 42, sorted iteration order everywhere.

Usage (from repo root)
----------------------
    python NER_pipeline/build_ner_folds.py
    python NER_pipeline/build_ner_folds.py --seed 42 --n-folds 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from gliner_baseline_eval import NER_EXPORTS, TIE_BREAK, _is_annotated, _spans  # noqa: E402


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OUT_DIR = REPO / "NER_pipeline" / "eval"
GOLD_OUT = OUT_DIR / "gold_561_merged.jsonl"
FOLDS_OUT = OUT_DIR / "folds_writer_year.json"
OVERVIEW_CSV = REPO / "data" / "letters_overview.csv"

# Concatenation policy: ART_NOM (indirect artwork ref) is merged into ART
# because ART alone only has 196 spans across the whole corpus.  The paper
# reports this as a coarsening; per-label metrics on the merged label are
# labelled "ART" in the output tables.
LABEL_MERGE_MAP: dict[str, str] = {
    "ART_NOM": "ART",
    "ART": "ART",
}


# --------------------------------------------------------------------------
# Gold pool assembly (extends build_gold to keep sender + year)
# --------------------------------------------------------------------------
def _load_date_backfill() -> dict[str, str]:
    """Return {normalised_letter_id: date_string} from letters_overview.csv.

    Ms_key in the CSV is like ``MS. II 70/2/330``; LS exports store
    ``Ms. II 70/2/330``.  We normalise both to lower-case, no-space form
    for the lookup.
    """
    if not OVERVIEW_CSV.exists():
        return {}
    import csv
    out: dict[str, str] = {}
    with OVERVIEW_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = _norm_ms(row.get("Ms_key", ""))
            date = (row.get("Date") or "").strip()
            if key and date:
                out[key] = date
    return out


def _norm_ms(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def build_gold_with_meta() -> list[dict]:
    """Return one dict per unique letter with metadata + merged labels."""
    date_backfill = _load_date_backfill()

    picked: dict[str, dict] = {}
    for name in TIE_BREAK:
        for t in json.loads(NER_EXPORTS[name].read_text(encoding="utf-8")):
            if not _is_annotated(t):
                continue
            data = t.get("data") or {}
            lid = data.get("letter_id")
            if not lid or lid in picked:
                continue

            raw_spans = _spans(t)
            merged_spans = [
                (s, e, LABEL_MERGE_MAP.get(lab, lab))
                for (s, e, lab) in raw_spans
            ]

            meta_date = (data.get("meta_date") or "").strip()
            if not meta_date:
                meta_date = date_backfill.get(_norm_ms(lid), "")

            picked[lid] = {
                "letter_id":    lid,
                "annotator":    name,
                "sender":       (data.get("meta_sender") or "").strip(),
                "date":         meta_date,
                "year":         _year_from_date(meta_date),
                "text":         data.get("text", "") or "",
                "gold_spans":   merged_spans,
            }

    out = sorted(picked.values(), key=lambda r: r["letter_id"])
    for r in out:
        r["decade"] = _decade(r["year"])
        r["text_length"] = len(r["text"])
        r["n_gold_spans"] = len(r["gold_spans"])
    return out


_YEAR_RE = re.compile(r"(1[78]\d{2})")


def _year_from_date(s: str | None) -> int | None:
    """Extract a 4-digit year from a free-form date string.  Returns None
    if no plausible 18xx/17xx year is found."""
    if not s:
        return None
    m = _YEAR_RE.search(str(s))
    return int(m.group(1)) if m else None


def _decade(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------
def _sender_key(sender: str) -> str:
    """Normalise a sender for stratum keying: drop life-dates, collapse
    whitespace, lower-case.  Two sender strings that differ only in
    trailing '(1817-1890)' will collapse to the same key."""
    s = re.sub(r"\(.*?\)", "", sender or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or "unknown"


def build_stratum_labels(letters: list[dict], n_folds: int) -> list[str]:
    """Compound stratum = sender × decade, with progressive fallback:
        1. sender × decade  (if the cell has >= n_folds letters)
        2. decade           (if the decade alone has >= n_folds letters)
        3. 'misc'           (everything else)
    """
    raw = [f"{_sender_key(r['sender'])}__{r['decade']}" for r in letters]
    decade_only = [r["decade"] for r in letters]

    counts = Counter(raw)
    dec_counts = Counter(decade_only)

    labels: list[str] = []
    for i, r in enumerate(raw):
        if counts[r] >= n_folds:
            labels.append(r)
        elif dec_counts[decade_only[i]] >= n_folds:
            labels.append(f"__decade__{decade_only[i]}")
        else:
            labels.append("__misc__")
    return labels


def assign_folds(letters: list[dict], n_folds: int, seed: int) -> list[int]:
    """Return a list of fold indices (0..n_folds-1) aligned with `letters`."""
    strata = build_stratum_labels(letters, n_folds)
    y = np.array(strata)
    X = np.zeros(len(letters))  # StratifiedKFold ignores X values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_ix = np.full(len(letters), -1, dtype=int)
    for fold_i, (_, test_ix) in enumerate(skf.split(X, y)):
        fold_ix[test_ix] = fold_i
    assert (fold_ix >= 0).all(), "some letters unassigned"
    return fold_ix.tolist()


# --------------------------------------------------------------------------
# Balance report
# --------------------------------------------------------------------------
def print_balance(letters: list[dict], folds: list[int], n_folds: int) -> None:
    print("\n=== Fold balance ===")
    print(f"Total letters: {len(letters)}")
    for f in range(n_folds):
        in_fold = [r for r, k in zip(letters, folds) if k == f]
        print(f"\n--- Fold {f} ---   n = {len(in_fold)}")
        print(f"  spans (total)     : {sum(r['n_gold_spans'] for r in in_fold)}")
        print(f"  text length (mean): {np.mean([r['text_length'] for r in in_fold]):.0f}")
        print(f"  text length (max) : {max(r['text_length'] for r in in_fold)}")

        # Top writers
        wc = Counter(_sender_key(r["sender"]) for r in in_fold)
        top = wc.most_common(6)
        top_str = ", ".join(f"{s or 'unknown'}={n}" for s, n in top)
        print(f"  top senders       : {top_str}")

        # Decade balance
        dc = Counter(r["decade"] for r in in_fold)
        dc_str = ", ".join(f"{d}={n}" for d, n in sorted(dc.items()))
        print(f"  decades           : {dc_str}")

        # Annotator balance (report, don't stratify)
        ac = Counter(r["annotator"] for r in in_fold)
        ac_str = ", ".join(f"{a}={n}" for a, n in sorted(ac.items()))
        print(f"  annotators        : {ac_str}")

    # Label distribution
    print(f"\n--- Label counts across all folds ---")
    label_by_fold: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r, k in zip(letters, folds):
        for _, _, lbl in r["gold_spans"]:
            label_by_fold[lbl][k] += 1
    header = " " * 12 + "".join(f"  fold{i:>2}" for i in range(n_folds)) + "    TOTAL"
    print(header)
    for lbl in sorted(label_by_fold):
        counts = [label_by_fold[lbl][f] for f in range(n_folds)]
        print(f"  {lbl:<10}" + "".join(f"  {c:>6}" for c in counts) + f"  {sum(counts):>7}")


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------
def write_gold(letters: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with GOLD_OUT.open("w", encoding="utf-8") as f:
        for r in letters:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {GOLD_OUT.relative_to(REPO)}  ({len(letters)} letters)")


def write_folds(letters: list[dict], folds: list[int], n_folds: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed":              seed,
        "n_folds":           n_folds,
        "stratification":    "sender x decade (with singleton fallback)",
        "label_merge_map":   LABEL_MERGE_MAP,
        "letters": [
            {
                "letter_id":     r["letter_id"],
                "annotator":     r["annotator"],
                "sender":        r["sender"],
                "year":          r["year"],
                "decade":        r["decade"],
                "text_length":   r["text_length"],
                "n_gold_spans":  r["n_gold_spans"],
                "fold":          k,
            }
            for r, k in zip(letters, folds)
        ],
    }
    FOLDS_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {FOLDS_OUT.relative_to(REPO)}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    print("Building gold pool (dedup with tie-break: "
          + " -> ".join(TIE_BREAK) + ") ...")
    letters = build_gold_with_meta()
    print(f"  {len(letters)} unique letters")

    # Sanity: how many letters have a parseable year?
    unknown = sum(1 for r in letters if r["year"] is None)
    if unknown:
        print(f"  (!) {unknown} letters have no parseable year -> "
              f"stratum bucket 'unknown'")

    print(f"\nAssigning {args.n_folds}-fold stratified split (seed={args.seed}) ...")
    folds = assign_folds(letters, args.n_folds, args.seed)

    print_balance(letters, folds, args.n_folds)

    write_gold(letters)
    write_folds(letters, folds, args.n_folds, args.seed)


if __name__ == "__main__":
    main()
