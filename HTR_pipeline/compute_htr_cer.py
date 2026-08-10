#!/usr/bin/env python3
"""
Compute per-letter CER between Claude HTR (`data.claude_text`) and the
human-corrected transcription (Label Studio `textarea` result), for every
HTR-correction task exported from Label Studio.

Sources scanned (all globbed automatically):
  - HTR_pipeline/HTR_intermediary_analysis_*.json
  - data/labelstudio_upload/HTR_corrections_finished/*.json

Each LS task carries the Claude HTR in `data.claude_text` and the annotator's
corrected transcription in the `textarea` result with `from_name=transcription`.
That gives a clean (claude, corrected) pair per (ms_key, annotator) without
having to find separate ground-truth files on disk.

Output:
  HTR_pipeline/trust_assessment/cer_pairs.csv

Columns:
  ms_key, letter_id, source_sender, annotator, source_file,
  len_claude, len_corrected, cer_body, cer_raw,
  pct_placeholders, pct_marginalia, pct_allcaps_runs,
  has_address, has_margin, has_ps

`cer_body` = CER after asymmetric normalisation (annotation-tag aware).
`cer_raw`  = CER with only whitespace + case + NFC normalisation. Reported for
comparison so you can see how much structural tags inflate things.

When the same letter appears in multiple LS files, all annotator passes are
kept as separate rows. Use `groupby('ms_key').agg(...)` downstream if you want
a single number per letter.

Run:
    python HTR_pipeline/compute_htr_cer.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

LS_GLOBS = [
    REPO / "data" / "labelstudio_upload" / "HTR_corrections_finished" / "HTR_corrections_*.json",
    REPO / "data" / "labelstudio_output" / "HTR_*.json",
]

OUT_DIR = REPO / "HTR_pipeline" / "trust_assessment"
OUT_CSV = OUT_DIR / "cer_pairs.csv"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# Corrected transcriptions use:
#   [+ Antoine-Jean Gros]  -> historian's entity annotation, not letter text
#   [? que l'avoit]        -> uncertain reading; text inside IS committed
#   [...]                  -> same "gave up" semantics as Claude
#
# Claude output uses:
#   [...]                  -> illegible
#   [Address: ...]         -> address panel block (real text, just relocated)
#   [Margin:  ...]         -> marginalia / cross-writing (real text)
#   [PS:      ...]         -> postscript (real text)
#   [Page N]               -> fallback labels when reading order is unclear
RE_MANUAL_ENTITY = re.compile(r"\[\+\s*[^\]]*\]")          # [+ X]    -> drop
RE_MANUAL_UNSURE = re.compile(r"\[\?\s*([^\]]*)\]")        # [? X]    -> X
RE_PLACEHOLDER   = re.compile(r"\[\s*\.\.\.\s*\]")         # [...]    -> drop
RE_PAGE_LABEL    = re.compile(r"\[Page\s+\d+\]", re.I)     # [Page N] -> drop
RE_LABELED_BLOCK = re.compile(                              # [Tag: X] -> X
    r"\[(Address|Margin|PS)\s*:\s*(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)
RE_WS = re.compile(r"\s+")


def nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def normalize_body(text: str, *, is_corrected: bool) -> str:
    """Asymmetric normalisation so both texts represent the same letter body.

    `is_corrected=True`  -> strip the human-annotation conventions
    `is_corrected=False` -> strip Claude's structural wrapper tags
    """
    t = text
    if is_corrected:
        t = RE_MANUAL_ENTITY.sub(" ", t)        # drop [+ Name] annotations
        t = RE_MANUAL_UNSURE.sub(r"\1", t)      # keep text inside [? ... ]
    # Common: collapse "I gave up" placeholders and page labels.
    t = RE_PLACEHOLDER.sub(" ", t)
    t = RE_PAGE_LABEL.sub(" ", t)
    # Unwrap labelled blocks: drop label, keep contents.
    t = RE_LABELED_BLOCK.sub(lambda m: " " + m.group(2) + " ", t)
    # Standard CER prep: NFC + case-fold + whitespace-collapse.
    return RE_WS.sub(" ", nfc_lower(t)).strip()


def normalize_raw(text: str) -> str:
    """Minimal normalisation: NFC + lowercase + whitespace collapse."""
    return RE_WS.sub(" ", nfc_lower(text)).strip()


# ---------------------------------------------------------------------------
# Cheap HTR-quality signals (from Claude output only)
# ---------------------------------------------------------------------------
RE_ALLCAPS_RUN = re.compile(r"\b[A-ZÀ-Ý]{4,}\b")  # 4+ uppercase letters

# Distinctive non-French markers. Each list contains tokens that are highly
# unlikely to appear in 19th-century French letters; presence of several is a
# strong hint that Claude transcribed (or hallucinated) foreign-language text.
# Lowercase, accent-preserved; matched against word-boundary-split tokens of
# the lowercased+NFC text.
ITALIAN_MARKERS = frozenset({
    # articulated prepositions (extremely distinctive)
    "della", "dello", "delle", "degli", "dalla", "dallo", "dalle", "dagli",
    "alla",  "allo",  "alle",  "agli",  "sulla", "sullo", "sulle", "sugli",
    "nella", "nello", "nelle", "negli",
    # function words / common verbs
    "gli", "siamo", "siete", "abbiamo", "abbastanza", "niente", "qualcosa",
    "anche", "ancora", "molto", "però", "perché",
    # possessives / forms distinctive to IT
    "miei", "tuoi", "suoi", "vostri", "nostri",
    "vorrei", "vorremmo", "carissimo", "carissima", "bellissimo", "bellissima",
    "signora", "signore", "signor",
})
LATIN_MARKERS = frozenset({
    "sunt", "esse", "erat", "fuerunt", "fuit",
    "omnia", "omnis", "omnem", "omni",
    "deus", "dei", "deum", "domino", "dominus",
    "sancti", "sanctus", "sancta", "amen",
    "gloria", "ecclesia", "requiem", "saecula", "saeculum",
    "propter", "magna", "magnum", "bonum",
    "quae", "quod", "quia", "ergo", "igitur", "autem",
})
RE_TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+")

# Writers whose hand is consistently easy to HTR (low expected CER).
# Substring-matched against the `source` field, case-insensitive.
EASY_WRITER_NAMES = (
    "Caillard de Beauvoir",
    "Keyser",            # matches "Nicaise de Keyser" and "Nicaise De Keyser"
)


def signals_from_claude(text: str) -> dict[str, float]:
    """Cheap signals computable without spaCy / dictionaries."""
    n = max(len(text), 1)
    chars_ph   = sum(m.end() - m.start() for m in RE_PLACEHOLDER.finditer(text))
    chars_marg = sum(m.end() - m.start() for m in RE_LABELED_BLOCK.finditer(text))
    chars_caps = sum(m.end() - m.start() for m in RE_ALLCAPS_RUN.finditer(text))

    tokens = [t.lower() for t in RE_TOKEN.findall(unicodedata.normalize("NFC", text))]
    n_tokens = max(len(tokens), 1)
    n_it = sum(1 for tok in tokens if tok in ITALIAN_MARKERS)
    n_la = sum(1 for tok in tokens if tok in LATIN_MARKERS)

    return {
        "pct_placeholders": round(chars_ph   / n, 4),
        "pct_marginalia":   round(chars_marg / n, 4),
        "pct_allcaps_runs": round(chars_caps / n, 4),
        "pct_non_french":   round((n_it + n_la) / n_tokens, 4),
        "n_italian_tokens": n_it,
        "n_latin_tokens":   n_la,
        "has_address":      int(bool(re.search(r"\[Address\s*:", text, re.I))),
        "has_margin":       int(bool(re.search(r"\[Margin\s*:",  text, re.I))),
        "has_ps":           int(bool(re.search(r"\[PS\s*:",      text, re.I))),
    }


def is_easy_writer(source: str) -> int:
    """1 if `source` matches a known easy-hand writer, else 0."""
    if not source:
        return 0
    s = source.lower()
    return int(any(name.lower() in s for name in EASY_WRITER_NAMES))


# ---------------------------------------------------------------------------
# LS extraction
# ---------------------------------------------------------------------------
def _extract_textarea_transcription(ann: dict) -> str | None:
    """Return the corrected transcription from a single LS annotation,
    or None if no textarea result with from_name='transcription' is present."""
    for r in ann.get("result", []):
        if r.get("type") == "textarea" and r.get("from_name") == "transcription":
            val = r.get("value", {})
            text = val.get("text")
            if isinstance(text, list):
                return text[0] if text else None
            if isinstance(text, str):
                return text
    return None


# Canonical annotator names — matched (case-insensitive) as substrings of the
# LS export filename. Order matters only for tie-break stability.
CANONICAL_ANNOTATORS = ("A1", "A2", "A3", "A4")

# Fallback: LabelStudio user id → canonical name, when the filename doesn't
# carry the annotator's name (e.g. batch exports like `HTR_corrections_HTR107_IV`).
USER_ID_TO_NAME = {22: "A1", 23: "A3", 24: "A2", 26: "A4"}


def _annotator_name(source_file: str, completed_by: int | None = None) -> str:
    """Derive a canonical annotator name.

    First tries to find one of `A1 / A2 / A3 / A4` as a substring
    of the filename. If none match and a LabelStudio `completed_by` user id is
    provided, uses the USER_ID_TO_NAME fallback. Otherwise returns the file stem.
    """
    stem = Path(source_file).stem
    low = stem.lower()
    for name in CANONICAL_ANNOTATORS:
        if name.lower() in low:
            return name
    if completed_by is not None and completed_by in USER_ID_TO_NAME:
        return USER_ID_TO_NAME[completed_by]
    return stem


def iter_ls_pairs():
    """Yield dicts: one per (LS task, annotator) with claude/corrected text.

    When the same (ms_key, annotator) appears in multiple export files, we
    keep only the pair from the **most recent** file (by mtime). The `1407_final`
    exports therefore automatically win over earlier `_207` / `_607` iterations.
    """
    # First pass: collect every candidate pair together with its file mtime.
    candidates: dict[tuple[str, str], dict] = {}
    seen_files: set[Path] = set()
    for glob in LS_GLOBS:
        for fp in sorted(glob.parent.glob(glob.name)):
            if fp in seen_files:
                continue
            seen_files.add(fp)
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [warn] {fp.name}: {type(e).__name__}: {e}")
                continue
            if not isinstance(data, list):
                continue
            fp_mtime = fp.stat().st_mtime
            for task in data:
                if not isinstance(task, dict):
                    continue
                td = task.get("data") or {}
                ms_key    = td.get("ms_key", "")
                claude    = td.get("claude_text", "") or ""
                sender    = td.get("source", "")
                letter_id = td.get("letter_id", "")
                if not ms_key or not claude:
                    continue
                for ann in task.get("annotations", []) or []:
                    corrected = _extract_textarea_transcription(ann)
                    if corrected is None:
                        continue
                    annotator = _annotator_name(fp.name, ann.get("completed_by"))
                    key = (ms_key, annotator)
                    prev = candidates.get(key)
                    if prev is not None and prev["_mtime"] >= fp_mtime:
                        continue  # older or equal — skip
                    candidates[key] = {
                        "ms_key":        ms_key,
                        "letter_id":     letter_id,
                        "source_sender": sender,
                        "annotator":     annotator,
                        "source_file":   str(fp.relative_to(REPO)),
                        "claude_text":   claude,
                        "corrected":     corrected,
                        "_mtime":        fp_mtime,
                    }

    for entry in candidates.values():
        entry.pop("_mtime", None)
        yield entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_ls_pairs())
    print(f"Extracted {len(pairs)} (ms_key, annotator) pairs from LS exports.")
    if not pairs:
        sys.exit("No pairs found - check that LS exports exist.")

    rows: list[dict] = []
    for p in pairs:
        ref_body = normalize_body(p["corrected"],   is_corrected=True)
        hyp_body = normalize_body(p["claude_text"], is_corrected=False)
        ref_raw  = normalize_raw(p["corrected"])
        hyp_raw  = normalize_raw(p["claude_text"])

        cer_body = jiwer.cer(ref_body, hyp_body) if ref_body and hyp_body else float("nan")
        cer_raw  = jiwer.cer(ref_raw,  hyp_raw)  if ref_raw  and hyp_raw  else float("nan")
        wer_body = jiwer.wer(ref_body, hyp_body) if ref_body and hyp_body else float("nan")
        wer_raw  = jiwer.wer(ref_raw,  hyp_raw)  if ref_raw  and hyp_raw  else float("nan")

        sigs = signals_from_claude(p["claude_text"])
        rows.append({
            "ms_key":        p["ms_key"],
            "letter_id":     p["letter_id"],
            "source_sender": p["source_sender"],
            "annotator":     p["annotator"],
            "source_file":   p["source_file"],
            "len_claude":    len(p["claude_text"]),
            "len_corrected": len(p["corrected"]),
            "cer_body":      round(cer_body, 4) if cer_body == cer_body else "",
            "cer_raw":       round(cer_raw,  4) if cer_raw  == cer_raw  else "",
            "wer_body":      round(wer_body, 4) if wer_body == wer_body else "",
            "wer_raw":       round(wer_raw,  4) if wer_raw  == wer_raw  else "",
            "is_easy_writer": is_easy_writer(p["source_sender"]),
            **sigs,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {OUT_CSV.relative_to(REPO)}  ({len(out_df)} rows)")

    # ---- summary --------------------------------------------------------
    cer_b = pd.to_numeric(out_df["cer_body"], errors="coerce")
    cer_r = pd.to_numeric(out_df["cer_raw"],  errors="coerce")

    def quantiles(s: pd.Series) -> str:
        s = s.dropna()
        if s.empty:
            return "(empty)"
        return (f"n={len(s):3d}  min={s.min():.3f}  p25={s.quantile(.25):.3f}  "
                f"median={s.median():.3f}  p75={s.quantile(.75):.3f}  "
                f"p90={s.quantile(.90):.3f}  max={s.max():.3f}")

    wer_b = pd.to_numeric(out_df["wer_body"], errors="coerce")
    wer_r = pd.to_numeric(out_df["wer_raw"],  errors="coerce")

    print("\nDistance distribution across ALL (ms_key, annotator) rows:")
    print(f"  cer_body : {quantiles(cer_b)}")
    print(f"  cer_raw  : {quantiles(cer_r)}")
    print(f"  wer_body : {quantiles(wer_b)}")
    print(f"  wer_raw  : {quantiles(wer_r)}")

    # Per-letter view: median across annotators
    out_df["cer_body_f"] = cer_b
    per_letter = (out_df.groupby("ms_key")["cer_body_f"]
                         .median()
                         .dropna())
    print(f"\nCER per unique letter (median across annotators), n={len(per_letter)}:")
    print(f"  median={per_letter.median():.3f}  "
          f"p75={per_letter.quantile(.75):.3f}  "
          f"p90={per_letter.quantile(.90):.3f}  "
          f"max={per_letter.max():.3f}")

    def bucket_counts(s: pd.Series, lo: float, hi: float) -> list[tuple[str, int]]:
        s = s.dropna()
        return [
            (f"OK (<{lo:.0%})",           int((s <  lo).sum())),
            (f"Good [{lo:.0%},{hi:.0%})", int(((s >= lo) & (s < hi)).sum())),
            (f"Needs check (>={hi:.0%})", int((s >= hi).sum())),
        ]

    print("\nBucket counts at proposed thresholds (2% / 8%), per unique letter:")
    for k, v in bucket_counts(per_letter, 0.02, 0.08):
        print(f"  {k:24s} {v:4d}")
    print("Bucket counts at alternative thresholds (5% / 15%), per unique letter:")
    for k, v in bucket_counts(per_letter, 0.05, 0.15):
        print(f"  {k:24s} {v:4d}")

    # Per-sender medians (senders with >=5 corrected letters)
    print("\nPer-sender median CER (>=5 corrected letters, body):")
    letter_tab = (out_df.groupby("ms_key")
                          .agg(source=("source_sender", "first"),
                               cer=("cer_body_f", "median"))
                          .dropna())
    agg = (letter_tab.groupby("source")
                     .agg(n=("cer", "size"),
                          median_cer=("cer", "median"),
                          p90_cer=("cer", lambda s: s.quantile(.90)))
                     .query("n >= 5")
                     .sort_values("median_cer"))
    if agg.empty:
        print("  (no sender has >=5 letters)")
    else:
        for src, row in agg.iterrows():
            print(f"  n={int(row['n']):3d}  median={row['median_cer']:.3f}  "
                  f"p90={row['p90_cer']:.3f}  {src}")

    # Correlation of cheap signals with CER
    print("\nSpearman correlation of cheap signals with cer_body (per-row):")
    for col in ["pct_placeholders", "pct_marginalia", "pct_allcaps_runs",
                "pct_non_french", "is_easy_writer", "len_claude"]:
        rho = out_df[col].corr(cer_b, method="spearman")
        print(f"  {col:18s} rho = {rho:+.3f}")

    # Easy-writer breakdown: what fraction of rows have the flag set, and
    # what does CER look like inside / outside that group?
    if "is_easy_writer" in out_df.columns and out_df["is_easy_writer"].any():
        easy = out_df[out_df["is_easy_writer"] == 1]
        hard = out_df[out_df["is_easy_writer"] == 0]
        cer_easy = pd.to_numeric(easy["cer_body"], errors="coerce").dropna()
        cer_hard = pd.to_numeric(hard["cer_body"], errors="coerce").dropna()
        print(f"\nEasy-writer flag breakdown (Caillard, Keyser):")
        print(f"  is_easy_writer=1: n={len(cer_easy):3d}  median={cer_easy.median():.3f}  "
              f"p90={cer_easy.quantile(.90):.3f}")
        print(f"  is_easy_writer=0: n={len(cer_hard):3d}  median={cer_hard.median():.3f}  "
              f"p90={cer_hard.quantile(.90):.3f}")

    # Non-French detection breakdown
    if "pct_non_french" in out_df.columns:
        with_foreign = (out_df["pct_non_french"] > 0).sum()
        print(f"\nNon-French marker detection:")
        print(f"  rows with >=1 IT or LA marker token: {with_foreign} / {len(out_df)}")
        if with_foreign:
            foreign = out_df[out_df["pct_non_french"] > 0].sort_values(
                "pct_non_french", ascending=False).head(10)
            print("  top 10 by pct_non_french:")
            for _, r in foreign.iterrows():
                print(f"    {r['ms_key']:18s}  pct={r['pct_non_french']:.3f}  "
                      f"it={r['n_italian_tokens']:2d} la={r['n_latin_tokens']:2d}  "
                      f"cer={r['cer_body']}")

    # IAA-style CER spread when a letter has multiple annotators
    dup = (out_df.groupby("ms_key")
                  .agg(n=("annotator", "nunique"),
                       cer_min=("cer_body_f", "min"),
                       cer_max=("cer_body_f", "max"))
                  .query("n >= 2"))
    if not dup.empty:
        dup["spread"] = dup["cer_max"] - dup["cer_min"]
        print(f"\nIAA-style CER spread (n={len(dup)} letters with >=2 annotators):")
        print(f"  median spread = {dup['spread'].median():.3f}  "
              f"p90 spread = {dup['spread'].quantile(.90):.3f}  "
              f"max spread = {dup['spread'].max():.3f}")


if __name__ == "__main__":
    main()
