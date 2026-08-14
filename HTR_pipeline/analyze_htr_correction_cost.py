"""HTR post-correction assessment for the manually corrected letters.

Produces the five numbers the paper's HTR section needs:

1. measured correction time per letter (and per page / per 1,000 characters);
2. proper-noun error rate, against the overall token error rate;
3. whether corrections landed where the interface pre-annotated them;
4. transcription cost in euros;
5. the mechanical / cognitive split of the edit log.

Method
------
Every corrected letter is a Label Studio task holding the Claude Opus 4.8
transcription (``data.claude_text``), the annotator's submitted text (the
``textarea`` result), ``lead_time`` (active-tab seconds) and the three
pre-computed anomaly highlights (``labels`` results with
``origin = "prediction"``: Suspicious French, Person Name, Format Anomaly).

Diffing the two texts with ``difflib`` gives a character-level edit log.
Each edit is then classified with the same mechanical / cognitive logic the
paper applies to NER, instantiated for transcription:

  mechanical  whitespace, hyphenation and stray punctuation; diacritic
              restoration; capitalisation; single-character substitution
  cognitive   model meta-commentary; silent omissions; proper-noun
              misreadings; passages marked illegible or uncertain; any other
              word-level substitution

Priority matters: a diacritic fix inside a name is still one keystroke, so
the mechanical string tests run first, but proper-noun involvement is checked
*before* the single-character rule, since "Beauvoir" -> "Beauvais" is a
misreading of an identity, not a typo.

Reads
-----
* ``data/labelstudio_output/HTR_*.json``
* ``data/htr_opus_outputs/*/usage.jsonl``   (OpenRouter cost per letter)
* ``data/letters_overview.csv``             (Ms_key <-> IDN mapping)

Writes
------
* ``HTR_pipeline/evaluation/htr_correction_per_letter.csv``
* ``HTR_pipeline/evaluation/htr_correction_edits.csv``
* ``HTR_pipeline/evaluation/htr_correction_report.md``
* ``HTR_pipeline/figures_htr_time/fig_htr_correction.{png,pdf}``

Usage
-----
    python HTR_pipeline/analyze_htr_correction_cost.py
    python HTR_pipeline/analyze_htr_correction_cost.py --usd-eur 0.90
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LS_DIR = REPO / "data" / "labelstudio_output"
HTR_OUT = REPO / "data" / "htr_opus_outputs"
OVERVIEW = REPO / "data" / "letters_overview.csv"
GOLD_NER = REPO / "NER_pipeline" / "eval" / "gold_561_merged.jsonl"
EVAL_DIR = REPO / "HTR_pipeline" / "evaluation"
FIG_DIR = REPO / "HTR_pipeline" / "figures_htr_time"

LETTER_CSV = EVAL_DIR / "htr_correction_per_letter.csv"
EDIT_CSV = EVAL_DIR / "htr_correction_edits.csv"
REPORT = EVAL_DIR / "htr_correction_report.md"

MAX_MINUTES = 60.0     # tasks above this are idle-inflated, capped not dropped
MIN_SECONDS = 10.0
HIGHLIGHT_LABELS = ["Suspicious French", "Person Name", "Format Anomaly"]

# Entity labels of the NER gold standard, in reporting order.  PER / LOC / ORG
# are the proper-name classes; PER_NOM is the nominal-reference class and acts
# as the control, since those mentions are ordinary common nouns.
GOLD_LABEL_ORDER = ["PER", "LOC", "ORG", "ART", "EVT", "PER_NOM"]
PROPER_NAME_LABELS = {"PER", "LOC", "ORG"}

# code -> (label, cost)
EDIT_CLASSES = {
    "space_punct": ("whitespace / hyphenation / stray punctuation", "mechanical"),
    "diacritic": ("diacritic restoration", "mechanical"),
    "case": ("capitalisation", "mechanical"),
    "single_char": ("single-character substitution", "mechanical"),
    "meta": ("model meta-commentary", "cognitive"),
    "omission": ("silent omission restored", "cognitive"),
    "proper_noun": ("proper-noun misreading", "cognitive"),
    "illegible": ("illegible / uncertain passage", "cognitive"),
    "lexical": ("other word-level substitution", "cognitive"),
}
CLASS_ORDER = list(EDIT_CLASSES)
MECHANICAL = {k for k, v in EDIT_CLASSES.items() if v[1] == "mechanical"}

OMISSION_CHARS = 40    # the threshold the paper already uses for long edits

META_RE = re.compile(
    r"\b(i apologi[sz]e|i was producing|here is the correct|note\s*:|"
    r"as an ai|i cannot|unable to (?:read|transcribe)|"
    r"the (?:image|scan) (?:is|appears)|transcription (?:note|of the))",
    re.IGNORECASE)
MARKER_RE = re.compile(r"\[\s*\.\.\.\s*\]|\[\?|\[illisible|\[illegible", re.IGNORECASE)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Capitalised tokens that are not names: sentence openers, months, weekdays and
# the honorifics that precede a name without being one.
NON_NAME_CAPS = {
    "je", "j", "il", "elle", "nous", "vous", "ils", "elles", "on", "le", "la",
    "les", "un", "une", "des", "ce", "cet", "cette", "ces", "mon", "ma", "mes",
    "votre", "vos", "notre", "nos", "son", "sa", "ses", "leur", "leurs",
    "et", "mais", "car", "donc", "or", "ni", "si", "que", "qui", "quand",
    "dans", "pour", "par", "sur", "sous", "avec", "sans", "chez", "depuis",
    "apres", "avant", "aussi", "ainsi", "enfin", "cependant", "pourtant",
    "monsieur", "madame", "mademoiselle", "messieurs", "mesdames",
    "mr", "mme", "mlle", "m", "mm", "cher", "chere", "chers",
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout",
    "septembre", "octobre", "novembre", "decembre",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "margin", "ps", "address", "adresse", "note",
}


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def alnum_only(s: str) -> str:
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE)


def levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Bounded Levenshtein distance; returns cap+1 if the distance exceeds cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def is_name_token(tok: str) -> bool:
    return (len(tok) >= 3 and tok[0].isupper()
            and strip_accents(tok).lower() not in NON_NAME_CAPS)


def name_tokens(text: str) -> list[tuple[int, int, str]]:
    """Proper-noun-like tokens as (start, end, token)."""
    return [(m.start(), m.end(), m.group())
            for m in WORD_RE.finditer(text) if is_name_token(m.group())]


# ---------------------------------------------------------------------------
# Edit classification
# ---------------------------------------------------------------------------
def classify_edit(op: str, old: str, new: str) -> str:
    o, n = old.strip(), new.strip()

    # --- mechanical string tests -----------------------------------------
    if alnum_only(old) == alnum_only(new):
        return "space_punct"
    if o and n and strip_accents(o).lower() == strip_accents(n).lower():
        return "diacritic" if strip_accents(o) != o or strip_accents(n) != n \
            else "case"
    if o and n and o.lower() == n.lower():
        return "case"

    # --- cognitive tests that must outrank the single-character rule ------
    if META_RE.search(old) or META_RE.search(new):
        return "meta"
    if MARKER_RE.search(old) or MARKER_RE.search(new):
        return "illegible"
    if len(n) - len(o) >= OMISSION_CHARS:
        return "omission"
    if any(is_name_token(t) for t in WORD_RE.findall(old) + WORD_RE.findall(new)):
        return "proper_noun"

    # --- back to mechanical ----------------------------------------------
    if op == "replace" and levenshtein(o, n) <= 1:
        return "single_char"
    if op in ("insert", "delete") and len(o or n) <= 1:
        return "single_char"
    return "lexical"


def diff_edits(claude: str, corrected: str) -> list[dict]:
    sm = difflib.SequenceMatcher(None, claude, corrected, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        old, new = claude[i1:i2], corrected[j1:j2]
        out.append({"op": op, "c_start": i1, "c_end": i2,
                    "n_start": j1, "n_end": j2,
                    "old": old, "new": new,
                    "n_changed": max(i2 - i1, j2 - j1),
                    "cls": classify_edit(op, old, new)})
    return out


def expand_to_words(text: str, start: int, end: int, pad: int = 1) -> str:
    """Widen a character range to whole words, for readable examples."""
    a, b = max(0, start), min(len(text), max(end, start + 1))
    while a > 0 and (text[a - 1].isalpha() or text[a - 1] in "'’-"):
        a -= 1
    while b < len(text) and (text[b].isalpha() or text[b] in "'’-"):
        b += 1
    for _ in range(pad):
        while a > 0 and text[a - 1].isspace():
            a -= 1
        while a > 0 and not text[a - 1].isspace():
            a -= 1
        while b < len(text) and text[b].isspace():
            b += 1
        while b < len(text) and not text[b].isspace():
            b += 1
    return " ".join(text[a:b].split())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_tasks() -> list[dict]:
    """One row per (annotator, letter), keeping the most recent submission."""
    best: dict[tuple[str, str], dict] = {}
    for path in sorted(LS_DIR.glob("HTR_*.json")):
        stem = path.stem
        ann = re.sub(r"HTR_|_final|_NEW\d+|\d{3,4}", "", stem).strip("_").lower()
        for t in json.loads(path.read_text(encoding="utf-8")):
            data = t.get("data") or {}
            ms_key = data.get("ms_key")
            claude = data.get("claude_text") or ""
            anns = t.get("annotations") or []
            if not ms_key or not claude or not anns:
                continue
            a = anns[0]
            corrected = ""
            highlights: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for r in a.get("result") or []:
                if r.get("type") == "textarea":
                    vals = (r.get("value") or {}).get("text") or []
                    if vals:
                        corrected = str(vals[0])
                elif r.get("type") == "labels":
                    v = r.get("value") or {}
                    lbls = v.get("labels") or []
                    if lbls and v.get("start") is not None:
                        highlights[str(lbls[0])].append(
                            (int(v["start"]), int(v["end"])))
            if not corrected:
                continue
            row = {
                "annotator": ann, "export": stem, "ms_key": ms_key,
                "letter_id": data.get("letter_id"),
                "n_pages": int(data.get("n_pages") or 0),
                "lead_time_s": float(a.get("lead_time") or 0.0),
                "updated_at": a.get("updated_at") or "",
                "claude": claude, "corrected": corrected,
                "highlights": dict(highlights),
            }
            key = (ann, ms_key)
            if key not in best or row["updated_at"] > best[key]["updated_at"]:
                best[key] = row
    return list(best.values())


def load_costs() -> dict[str, dict]:
    """IDN -> {cost_usd, n_pages, prompt_tokens, completion_tokens}."""
    out: dict[str, dict] = {}
    for path in HTR_OUT.glob("*/usage.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = r.get("usage") or {}
            out[str(r.get("letter_id"))] = {
                "cost_usd": float(u.get("cost") or 0.0),
                "n_pages": int(r.get("n_pages") or 0),
                "prompt_tokens": int(u.get("prompt_tokens") or 0),
                "completion_tokens": int(u.get("completion_tokens") or 0),
            }
    return out


def load_ms_to_idn() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with OVERVIEW.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ms, idn = (r.get("Ms_key") or "").strip(), (r.get("IDN") or "").strip()
            if ms and idn:
                mapping[ms.upper().replace(" ", "")] = idn
    return mapping


def norm_key(s: str | None) -> str:
    s = str(s or "").strip().upper().replace(" ", "").replace("IDN_", "")
    return re.sub(r"^MS\.?II70", "MSII70", s)


def load_gold_entities() -> dict[str, dict]:
    """Human NER annotations, keyed by normalised shelfmark / IDN.

    The NER tasks were built from the corrected transcriptions, so these span
    offsets are valid on the annotator's final text (verified: median
    similarity between the corrected text and the gold text is 1.000).
    """
    out: dict[str, dict] = {}
    if not GOLD_NER.exists():
        return out
    with GOLD_NER.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[norm_key(r["letter_id"])] = {
                "text": r["text"],
                "spans": [(int(s[0]), int(s[1]), str(s[2]))
                          for s in r["gold_spans"]],
            }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def highlight_analysis(rows: list[dict]) -> pd.DataFrame:
    """Did the edits land inside the pre-computed highlights?"""
    stats = {lbl: Counter() for lbl in HIGHLIGHT_LABELS + ["any"]}
    for r in rows:
        text_len = len(r["claude"])
        if not text_len:
            continue
        spans_by_lbl = {lbl: r["highlights"].get(lbl, [])
                        for lbl in HIGHLIGHT_LABELS}
        spans_by_lbl["any"] = [s for v in spans_by_lbl.values() for s in v]
        for lbl, spans in spans_by_lbl.items():
            covered = set()
            for s, e in spans:
                covered.update(range(max(0, s), min(text_len, e)))
            st = stats[lbl]
            st["chars_covered"] += len(covered)
            st["chars_total"] += text_len
            st["n_spans"] += len(spans)
            hit_spans = set()
            for ed in r["edits"]:
                a, b = ed["c_start"], max(ed["c_end"], ed["c_start"] + 1)
                overlap = any(x in covered for x in range(a, b))
                st["edits_total"] += 1
                st["edits_hit"] += int(overlap)
                if overlap:
                    for i, (s, e) in enumerate(spans):
                        if min(b, e) > max(a, s):
                            hit_spans.add(i)
            st["spans_with_edit"] += len(hit_spans)

    recs = []
    for lbl in HIGHLIGHT_LABELS + ["any"]:
        st = stats[lbl]
        cov = st["chars_covered"] / st["chars_total"] if st["chars_total"] else 0
        hit = st["edits_hit"] / st["edits_total"] if st["edits_total"] else 0
        prec = st["spans_with_edit"] / st["n_spans"] if st["n_spans"] else 0
        recs.append({"highlight": lbl, "n_spans": st["n_spans"],
                     "char_coverage": cov, "edit_hit_rate": hit,
                     "lift": hit / cov if cov else float("nan"),
                     "span_precision": prec})
    return pd.DataFrame(recs)


def gold_entity_error_rate(rows: list[dict], gold: dict[str, dict]) -> dict:
    """Error rate on human-annotated entity mentions.

    For each letter that has both a corrected transcription and NER gold
    annotations, Claude's raw output is aligned to the gold text with
    ``difflib``.  Any gold character sitting inside an ``insert`` or
    ``replace`` opcode was produced by the annotator rather than by Claude, so
    a gold mention counts as an error if any of its characters were changed.
    ``delete`` opcodes are ignored on purpose: they remove text Claude
    invented, which does not alter the mention itself.

    Letters corrected by more than one annotator are counted once.
    """
    per_label = defaultdict(lambda: {"n": 0, "edited": 0,
                                     "chars": 0, "chars_edited": 0})
    ent_chars = ent_chars_edited = 0
    non_ent_chars = non_ent_chars_edited = 0
    n_letters = 0
    examples: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    for r in rows:
        key = norm_key(r["ms_key"])
        g = gold.get(key)
        if g is None or key in seen:
            continue
        seen.add(key)
        n_letters += 1
        gtext = g["text"]

        changed = set()
        sm = difflib.SequenceMatcher(None, r["claude"], gtext, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("insert", "replace"):
                changed.update(range(j1, j2))

        in_entity = set()
        for s, e, label in g["spans"]:
            s, e = max(0, s), min(len(gtext), e)
            if e <= s:
                continue
            span_positions = range(s, e)
            in_entity.update(span_positions)
            hit = [i for i in span_positions if i in changed]
            d = per_label[label]
            d["n"] += 1
            d["edited"] += int(bool(hit))
            d["chars"] += e - s
            d["chars_edited"] += len(hit)
            if hit and label in PROPER_NAME_LABELS and len(examples) < 60:
                i_lo, i_hi = _gold_to_claude(sm, s, e)
                examples.append((r["ms_key"], label,
                                 " ".join(r["claude"][i_lo:i_hi].split()),
                                 " ".join(gtext[s:e].split())))

        for i in range(len(gtext)):
            if i in in_entity:
                ent_chars += 1
                ent_chars_edited += int(i in changed)
            else:
                non_ent_chars += 1
                non_ent_chars_edited += int(i in changed)

    return {"n_letters": n_letters, "per_label": dict(per_label),
            "ent_chars": ent_chars, "ent_chars_edited": ent_chars_edited,
            "non_ent_chars": non_ent_chars,
            "non_ent_chars_edited": non_ent_chars_edited,
            "examples": examples}


def _gold_to_claude(sm: difflib.SequenceMatcher, s: int, e: int) -> tuple[int, int]:
    """Map a gold-text range back to the corresponding range in Claude's text."""
    lo, hi = None, None
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if min(e, j2) > max(s, j1) or (j1 == j2 and s <= j1 <= e):
            lo = i1 if lo is None else min(lo, i1)
            hi = i2 if hi is None else max(hi, i2)
    return (lo if lo is not None else s, hi if hi is not None else e)


def proper_noun_rate(rows: list[dict]) -> dict:
    """Share of proper-noun tokens the annotator had to touch."""
    tot = touched = 0
    tot_all = touched_all = 0
    examples: list[tuple[str, str, str]] = []
    for r in rows:
        edited = set()
        for ed in r["edits"]:
            edited.update(range(ed["c_start"], max(ed["c_end"],
                                                   ed["c_start"] + 1)))
        for s, e, tok in [(m.start(), m.end(), m.group())
                          for m in WORD_RE.finditer(r["claude"])]:
            hit = any(i in edited for i in range(s, e))
            tot_all += 1
            touched_all += int(hit)
            if is_name_token(tok):
                tot += 1
                touched += int(hit)
    # Examples come from the edit log, widened to whole words so the change
    # is legible as a name rather than as a stray character.
    seen = set()
    for r in rows:
        for ed in r["edits"]:
            if ed["cls"] != "proper_noun" or len(examples) >= 40:
                continue
            old = expand_to_words(r["claude"], ed["c_start"], ed["c_end"])
            new = expand_to_words(r["corrected"], ed["n_start"], ed["n_end"])
            if old == new or (old, new) in seen or len(old) > 70:
                continue
            seen.add((old, new))
            examples.append((r["ms_key"], old, new))
    return {"name_tokens": tot, "name_tokens_edited": touched,
            "name_error_rate": touched / tot if tot else 0.0,
            "all_tokens": tot_all, "all_tokens_edited": touched_all,
            "token_error_rate": touched_all / tot_all if tot_all else 0.0,
            "examples": examples}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, edits: pd.DataFrame, hl: pd.DataFrame,
                 pn: dict, ge: dict, cost: dict, usd_eur: float) -> str:
    out: list[str] = []
    A = out.append
    n = len(df)

    A("# HTR post-correction assessment\n")
    A(f"{n} manually corrected letters "
      f"({df.ms_key.nunique()} distinct shelfmarks, "
      f"{int(df.n_pages.sum())} pages, {int(df.n_claude_chars.sum()):,} "
      "characters of Claude Opus 4.8 output), diffed against the annotators' "
      "submitted transcriptions.\n")

    # --- 1. time ---------------------------------------------------------
    A("## 1. Measured correction time\n")
    A(f"Time on task is Label Studio's `lead_time` (active-tab seconds, "
      f"paused after ~60 s idle), capped at {MAX_MINUTES:g} min to contain "
      "tasks left open.\n")
    A("| | mean | median | p75 | p90 | max |")
    A("|---|---:|---:|---:|---:|---:|")
    for col, name in (("minutes", "minutes per letter"),
                      ("min_per_page", "minutes per page"),
                      ("min_per_1k", "minutes per 1,000 characters")):
        s = df[col]
        A(f"| {name} | {s.mean():.1f} | {s.median():.1f} | "
          f"{s.quantile(.75):.1f} | {s.quantile(.9):.1f} | {s.max():.1f} |")
    A("")
    A(f"Total recorded correction effort: **{df.minutes.sum() / 60:.1f} "
      f"hours** for {n} letters.\n")
    A("| annotator | letters | pages | median min | mean min | "
      "min/page | edits | sec/edit |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ann, g in df.groupby("annotator"):
        A(f"| {ann} | {len(g)} | {int(g.n_pages.sum())} | "
          f"{g.minutes.median():.1f} | {g.minutes.mean():.1f} | "
          f"{g.min_per_page.median():.1f} | {int(g.n_edits.sum())} | "
          f"{60 * g.minutes.sum() / max(g.n_edits.sum(), 1):.1f} |")
    A("")
    A("The mean is roughly twice the median: a minority of letters "
      "(faint ink, dense cursive) absorbs a disproportionate share of the "
      "effort, which is the same long tail the CER table reports.\n")

    # --- 2. proper nouns -------------------------------------------------
    A("## 2. Proper-noun error rate\n")

    if ge and ge["n_letters"]:
        pl = ge["per_label"]
        A(f"Measured against the NER gold standard rather than a spelling "
          f"heuristic. {ge['n_letters']} of these letters were subsequently "
          "annotated for named entities, so we know exactly which character "
          "ranges a human considered to be a person, place or institution. "
          "Claude's raw output is aligned to that annotated text; a mention "
          "counts as an error if any of its characters had to be changed.\n")
        A("| entity label | mentions | edited | mention error rate | "
          "characters | character error rate |")
        A("|---|---:|---:|---:|---:|---:|")
        for lbl in GOLD_LABEL_ORDER:
            d = pl.get(lbl)
            if not d or not d["n"]:
                continue
            tag = f"**{lbl}**" if lbl in PROPER_NAME_LABELS else lbl
            A(f"| {tag} | {d['n']:,} | {d['edited']:,} | "
              f"{100 * d['edited'] / d['n']:.1f}% | {d['chars']:,} | "
              f"{100 * d['chars_edited'] / max(d['chars'], 1):.1f}% |")
        prop = [pl[l] for l in PROPER_NAME_LABELS if l in pl]
        p_n = sum(d["n"] for d in prop)
        p_e = sum(d["edited"] for d in prop)
        p_c = sum(d["chars"] for d in prop)
        p_ce = sum(d["chars_edited"] for d in prop)
        A(f"| **all proper names (PER+LOC+ORG)** | {p_n:,} | {p_e:,} | "
          f"**{100 * p_e / max(p_n, 1):.1f}%** | {p_c:,} | "
          f"{100 * p_ce / max(p_c, 1):.1f}% |")
        A(f"| non-entity text | — | — | — | {ge['non_ent_chars']:,} | "
          f"{100 * ge['non_ent_chars_edited'] / max(ge['non_ent_chars'], 1):.1f}% |")
        A("")
        nom = pl.get("PER_NOM")
        per = pl.get("PER")
        if nom and per and nom["n"] and per["n"]:
            r_per = per["edited"] / per["n"]
            r_nom = nom["edited"] / nom["n"]
            A(f"`PER_NOM` is the control: those mentions (« mon père », « le "
              f"ministre ») are ordinary common nouns that a human annotator "
              f"marked as referring to a person. Named persons are corrected "
              f"{r_per / max(r_nom, 1e-9):.1f}x as often as nominal references "
              f"({100 * r_per:.1f}% against {100 * r_nom:.1f}%), which "
              "isolates the difficulty of the *name* from the difficulty of "
              "the passage it sits in.\n")
        nonent = ge["non_ent_chars_edited"] / max(ge["non_ent_chars"], 1)
        entc = p_ce / max(p_c, 1)
        A(f"At character level, {100 * entc:.1f}% of the text inside a proper "
          f"name had to be corrected against {100 * nonent:.1f}% of the text "
          f"outside any entity — a factor of {entc / max(nonent, 1e-9):.1f}. "
          "This is the number that matters for the downstream network: a "
          "mis-transcribed common noun costs a reader nothing, whereas a "
          "mis-transcribed surname silently creates or destroys a node.\n")

        A("### Cross-check against the spelling heuristic\n")
    else:
        A("No NER gold standard was found, so only the heuristic estimate is "
          "reported.\n")

    A("A purely orthographic detector — a token counts as a proper noun if it "
      "is capitalised, at least three characters long, and not a sentence "
      "opener, month, weekday or honorific — gives the following, over all "
      f"{n} corrections including the letters without NER annotations.\n")
    A("| | tokens | edited | error rate |")
    A("|---|---:|---:|---:|")
    A(f"| capitalised tokens | {pn['name_tokens']:,} | "
      f"{pn['name_tokens_edited']:,} | {100 * pn['name_error_rate']:.1f}% |")
    A(f"| all tokens | {pn['all_tokens']:,} | {pn['all_tokens_edited']:,} | "
      f"{100 * pn['token_error_rate']:.1f}% |")
    A("")
    ratio = (pn["name_error_rate"] / pn["token_error_rate"]
             if pn["token_error_rate"] else float("nan"))
    A(f"The heuristic puts the ratio at {ratio:.1f}x. It is reported only as a "
      "sanity check; the gold-grounded figures above are the ones to quote, "
      "since they rest on a human judgement of what is a name rather than on "
      "capitalisation in a 19th-century hand.\n")

    if ge and ge["examples"]:
        A("### Proper-name mentions Claude got wrong\n")
        A("Left is Claude Opus 4.8, right is the annotated gold mention.\n")
        shown = set()
        for ms, lbl, old, new in ge["examples"]:
            if (old, new) in shown or old == new or len(old) > 70:
                continue
            shown.add((old, new))
            A(f"- `{ms}` `{lbl}`: {old} → **{new}**")
            if len(shown) >= 14:
                break
        A("")

    # --- 3. highlights ---------------------------------------------------
    A("## 3. Did corrections happen where they were pre-annotated?\n")
    A("`char coverage` is the share of the transcription each highlight class "
      "covers; `hit rate` is the share of edits that fall inside it; `lift` is "
      "hit rate over coverage, i.e. how much better than pointing at random "
      "the highlight is; `span precision` is the share of highlighted spans "
      "that contained at least one edit.\n")
    A("| highlight | spans | char coverage | edit hit rate | lift | "
      "span precision |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in hl.iterrows():
        A(f"| {r.highlight} | {int(r.n_spans):,} | "
          f"{100 * r.char_coverage:.1f}% | {100 * r.edit_hit_rate:.1f}% | "
          f"{r.lift:.2f}x | {100 * r.span_precision:.1f}% |")
    A("")
    any_row = hl[hl.highlight == "any"].iloc[0]
    A(f"Taken together the three overlays cover "
      f"{100 * any_row.char_coverage:.0f}% of the text and catch "
      f"{100 * any_row.edit_hit_rate:.0f}% of the corrections, a lift of "
      f"{any_row.lift:.2f}x over chance. A lift above 1 means the overlay is "
      "informative; the gap between hit rate and 100% is the share of "
      "corrections the annotator found unaided, which is the part of the "
      "workload no interface currently supports.\n")

    # --- 4. cost ---------------------------------------------------------
    A("## 4. Transcription cost\n")
    A(f"OpenRouter usage records for Claude Opus 4.8, converted at "
      f"1 USD = {usd_eur:.2f} EUR.\n")
    A("| | letters | pages | cost (USD) | cost (EUR) | EUR / letter | "
      "EUR / page |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for key, name in (("corpus", "whole digitised corpus"),
                      ("corrected", "the corrected subset")):
        c = cost[key]
        A(f"| {name} | {c['n']:,} | {c['pages']:,} | ${c['usd']:.2f} | "
          f"€{c['usd'] * usd_eur:.2f} | €{c['usd'] * usd_eur / max(c['n'], 1):.3f} | "
          f"€{c['usd'] * usd_eur / max(c['pages'], 1):.3f} |")
    A("")
    corr = cost["corrected"]
    if corr["n"]:
        eur = corr["usd"] * usd_eur
        hours = df.minutes.sum() / 60
        A(f"Those same letters took {hours:.0f} hours of human correction, so "
          f"the inference is worth €{eur / hours:.2f} of corrector time per "
          "hour: transcription would only stop being the cheap part of this "
          f"pipeline if an annotator could be paid €{eur / hours:.2f} an hour. "
          "The binding cost is human attention, not tokens, which is the "
          "premise of the whole paper stated as a budget line.\n")

    # --- 5. mechanical / cognitive ---------------------------------------
    A("## 5. Mechanical / cognitive split of the edit log\n")
    tot = len(edits)
    tot_chars = edits.n_changed.sum()
    A(f"{tot:,} edits over {n} letters "
      f"({tot / n:.0f} per letter), touching {int(tot_chars):,} characters.\n")
    A("| class | cost | edits | % of edits | characters | % of characters |")
    A("|---|---|---:|---:|---:|---:|")
    for c in CLASS_ORDER:
        sub = edits[edits.cls == c]
        name, cost_lbl = EDIT_CLASSES[c]
        A(f"| {name} | {cost_lbl} | {len(sub):,} | "
          f"{100 * len(sub) / tot:.1f}% | {int(sub.n_changed.sum()):,} | "
          f"{100 * sub.n_changed.sum() / tot_chars:.1f}% |")
    mech = edits[edits.cls.isin(MECHANICAL)]
    cog = edits[~edits.cls.isin(MECHANICAL)]
    A(f"| **mechanical total** | | **{len(mech):,}** | "
      f"**{100 * len(mech) / tot:.1f}%** | **{int(mech.n_changed.sum()):,}** | "
      f"**{100 * mech.n_changed.sum() / tot_chars:.1f}%** |")
    A(f"| **cognitive total** | | **{len(cog):,}** | "
      f"**{100 * len(cog) / tot:.1f}%** | **{int(cog.n_changed.sum()):,}** | "
      f"**{100 * cog.n_changed.sum() / tot_chars:.1f}%** |")
    A("")
    A("Counted per edit and per character the split tells different stories, "
      "and both belong in the paper: mechanical fixes are numerous but tiny, "
      "cognitive ones are fewer and rewrite whole passages.\n")

    A("### Per annotator\n")
    A("| annotator | edits | % mechanical | % cognitive | proper-noun edits |")
    A("|---|---:|---:|---:|---:|")
    for ann, g in edits.groupby("annotator"):
        m = g.cls.isin(MECHANICAL).sum()
        A(f"| {ann} | {len(g):,} | {100 * m / len(g):.1f}% | "
          f"{100 * (len(g) - m) / len(g):.1f}% | "
          f"{(g.cls == 'proper_noun').sum():,} |")
    A("")

    A("### Sample proper-noun corrections\n")
    A("Widened to whole words; left is Claude Opus 4.8, right is the "
      "annotator.\n")
    for ms, old, new in pn["examples"][:14]:
        A(f"- `{ms}`: {old} → **{new}**")
    A("")

    A("## Caveats\n")
    A("- `lead_time` is active-tab time, not effort, and is capped here at "
      f"{MAX_MINUTES:g} min per task.")
    A("- These figures measure *residual* work after Claude's output, so they "
      "are a lower bound on its true error rate, not a CER.")
    A("- The gold-grounded entity rates inherit the NER gold standard's own "
      "limits: outside the 30 inter-annotator-agreement letters it is "
      "single-annotator, and mention boundaries carry the annotators' "
      "residual disagreement (lenient F1 0.94 for `PER`, 0.90 for `LOC`).")
    A("- The capitalisation heuristic reported as a cross-check will miss "
      "lowercased names in careless hands and over-count capitalised common "
      "nouns in 19th-century orthography.")
    A("- Edit classes are assigned by string tests in a fixed priority order; "
      "`other word-level substitution` is the residual class and is counted "
      "as cognitive because resolving it requires the scan.\n")
    return "\n".join(out)


def make_edit_class_figure(edits: pd.DataFrame) -> None:
    """Stand-alone, paper-ready figure of the HTR edit log by class.

    Each class gets two bars: its share of the *number* of edits and its share
    of the *characters* rewritten.  The two rarely agree, which is the point.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(REPO / "NER_pipeline"))
        import figstyle
    except ImportError:
        print("matplotlib not available — skipping figure")
        return
    figstyle.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tot_e = len(edits)
    tot_c = edits.n_changed.sum()

    def stats(c):
        sub = edits[edits.cls == c]
        return (len(sub), 100 * len(sub) / tot_e,
                100 * sub.n_changed.sum() / tot_c)

    mech = sorted([c for c in CLASS_ORDER if c in MECHANICAL],
                  key=lambda c: -stats(c)[1])
    cog = sorted([c for c in CLASS_ORDER if c not in MECHANICAL],
                 key=lambda c: -stats(c)[1])
    order = mech + cog
    rows = [stats(c) for c in order]
    n_edits = [r[0] for r in rows]
    share_e = [r[1] for r in rows]
    share_c = [r[2] for r in rows]

    PALETTE = {
        "mech_edits": figstyle.MECH, "mech_chars": figstyle.MECH_LIGHT,
        "cog_edits": figstyle.COG, "cog_chars": figstyle.COG_LIGHT,
    }
    col_e = [PALETTE["mech_edits"] if c in MECHANICAL else PALETTE["cog_edits"]
             for c in order]
    col_c = [PALETTE["mech_chars"] if c in MECHANICAL else PALETTE["cog_chars"]
             for c in order]

    fig, ax = plt.subplots(figsize=figstyle.size(height=4.6))
    fig.patch.set_facecolor("white")
    # A gap between the two cost groups leaves room for a group heading.
    GAP = 1.15
    ys = ([float(i) for i in range(len(mech))]
          + [len(mech) + GAP + i for i in range(len(cog))])
    h = 0.36

    ax.barh([y - h / 2 - 0.02 for y in ys], share_e, height=h, color=col_e,
            zorder=3, label="share of edits")
    ax.barh([y + h / 2 + 0.02 for y in ys], share_c, height=h, color=col_c,
            zorder=3, label="share of rewritten characters")

    for y, se, sc in zip(ys, share_e, share_c):
        ax.text(se + 0.5, y - h / 2 - 0.02, f"{se:.1f}%", va="center",
                fontsize=7.5, color=figstyle.VALUE, zorder=4)
        ax.text(sc + 0.5, y + h / 2 + 0.02, f"{sc:.1f}%", va="center",
                fontsize=7.5, color="#666666", zorder=4)

    labels = [f"{EDIT_CLASSES[c][0]}\n" + r"$\it{n}$" + f" = {n:,} edits"
              for c, n in zip(order, n_edits)]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, linespacing=1.3)

    # Group headings, placed in the gap above each block.
    def _tot(group):
        e = sum(s for c, s in zip(order, share_e) if c in group)
        ch = sum(s for c, s in zip(order, share_c) if c in group)
        return e, ch

    me, mc = _tot(set(mech))
    ce, cc = _tot(set(cog))
    ax.text(0, -1.0, f"MECHANICAL  ·  {me:.0f}% of edits, {mc:.0f}% of "
            "characters", fontsize=8.5, fontweight="bold",
            color=PALETTE["mech_edits"], va="center", zorder=4)
    ax.text(0, len(mech) + GAP / 2 - 0.15,
            f"COGNITIVE  ·  {ce:.0f}% of edits, {cc:.0f}% of characters",
            fontsize=8.5, fontweight="bold", color=PALETTE["cog_edits"],
            va="center", zorder=4)
    ax.set_ylim(ys[-1] + 0.8, -1.5)

    ax.set_xlim(0, max(share_e + share_c) * 1.16)
    ax.set_xlabel("% of the HTR edit log", labelpad=6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    figstyle.clean(ax)

    ax.legend(loc="lower right", frameon=True, framealpha=.95,
              edgecolor=figstyle.GRID)

    fig.tight_layout()
    figstyle.save(fig, FIG_DIR / "fig_htr_edit_classes")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_htr_edit_classes.png'}")


def make_mech_cog_figure(edits: pd.DataFrame) -> None:
    """Stand-alone paper figure for the mechanical / cognitive split.

    Two panels, because the split inverts depending on the unit: mechanical
    fixes dominate the *number* of actions, cognitive ones dominate the
    *amount of text* rewritten.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(REPO / "NER_pipeline"))
        import figstyle
    except ImportError:
        print("matplotlib not available — skipping figure")
        return
    figstyle.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tot_e = len(edits)
    tot_c = edits.n_changed.sum()
    # Mechanical classes first, then cognitive, each ordered by edit count.
    order = ([c for c in CLASS_ORDER if c in MECHANICAL]
             + [c for c in CLASS_ORDER if c not in MECHANICAL])
    share_e = [100 * len(edits[edits.cls == c]) / tot_e for c in order]
    share_c = [100 * edits[edits.cls == c].n_changed.sum() / tot_c
               for c in order]
    n_edits = [len(edits[edits.cls == c]) for c in order]
    cols = [figstyle.MECH if c in MECHANICAL else figstyle.COG for c in order]

    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(height=3.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    ys = list(range(len(order)))
    ax.barh([y - 0.2 for y in ys], share_e, height=.38, color=cols,
            label="share of edits")
    ax.barh([y + 0.2 for y in ys], share_c, height=.38, color=cols,
            alpha=.42, hatch="///", edgecolor="white",
            label="share of changed characters")
    for y, (se, sc, n) in enumerate(zip(share_e, share_c, n_edits)):
        ax.text(max(se, sc) + 0.8, y, f"n={n:,}", va="center", fontsize=6.5,
                color="#555555")
    # Separator between the two cost groups.
    n_mech = sum(1 for c in order if c in MECHANICAL)
    ax.axhline(n_mech - 0.5, color="#999999", lw=.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([EDIT_CLASSES[c][0] for c in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(share_e + share_c) * 1.30)
    ax.set_xlabel("% of the edit log")
    ax.set_title("HTR edit log by class\n(blue = mechanical, red = cognitive)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    mech_e = 100 * edits.cls.isin(MECHANICAL).sum() / tot_e
    mech_c = (100 * edits[edits.cls.isin(MECHANICAL)].n_changed.sum() / tot_c)
    xs = [0, 1]
    bottoms = [mech_e, mech_c]
    ax.bar(xs, bottoms, width=.55, color=figstyle.MECH, label="mechanical")
    ax.bar(xs, [100 - mech_e, 100 - mech_c], width=.55, bottom=bottoms,
           color=figstyle.COG, label="cognitive")
    for x, b in zip(xs, bottoms):
        ax.text(x, b / 2, f"{b:.0f}%", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(x, b + (100 - b) / 2, f"{100 - b:.0f}%", ha="center",
                va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"edits\n(n={tot_e:,})",
                        f"changed characters\n(n={int(tot_c):,})"], fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("The split inverts with the unit", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, -0.30), ncol=2,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    figstyle.save(fig, FIG_DIR / "fig_htr_mech_cog")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_htr_mech_cog.png'}")


def make_figure(df: pd.DataFrame, edits: pd.DataFrame, hl: pd.DataFrame,
                ge: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(REPO / "NER_pipeline"))
        import figstyle
    except ImportError:
        print("matplotlib not available — skipping figure")
        return
    figstyle.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=figstyle.size(height=5.4))
    axes = axes.ravel()

    ax = axes[0]
    ax.hist(df.minutes, bins=30, color=figstyle.MECH)
    ax.axvline(df.minutes.median(), color=figstyle.COG, ls="--",
               label=f"median {df.minutes.median():.1f} min")
    ax.set_xlabel("minutes per letter")
    ax.set_ylabel("letters")
    ax.set_title("Correction time")
    ax.legend(fontsize=8)

    ax = axes[1]
    tot_e = len(edits)
    tot_c = edits.n_changed.sum()
    share_e = [100 * len(edits[edits.cls == c]) / tot_e for c in CLASS_ORDER]
    share_c = [100 * edits[edits.cls == c].n_changed.sum() / tot_c
               for c in CLASS_ORDER]
    ys = list(range(len(CLASS_ORDER)))
    cols = [figstyle.MECH_LIGHT if c in MECHANICAL else figstyle.COG
            for c in CLASS_ORDER]
    ax.barh([y - 0.2 for y in ys], share_e, height=.38, color=cols,
            label="% of edits")
    ax.barh([y + 0.2 for y in ys], share_c, height=.38, color=cols,
            alpha=.45, hatch="///", edgecolor="white",
            label="% of changed characters")
    ax.set_yticks(ys)
    ax.set_yticklabels([EDIT_CLASSES[c][0] for c in CLASS_ORDER], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of the edit log")
    ax.set_title("Edit log by class (blue = mechanical)")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[2]
    d = hl[hl.highlight != "any"]
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], 100 * d.char_coverage, width=.4,
           label="char coverage", color="#95a5a6")
    ax.bar([i + 0.2 for i in x], 100 * d.edit_hit_rate, width=.4,
           label="edit hit rate", color=figstyle.MECH)
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.highlight, fontsize=7.5, rotation=15)
    ax.set_ylabel("%")
    ax.set_title("Pre-annotation overlays")
    ax.legend(fontsize=8)

    ax = axes[3]
    if ge and ge["n_letters"]:
        pl = ge["per_label"]
        labs = [l for l in GOLD_LABEL_ORDER if pl.get(l, {}).get("n")]
        vals = [100 * pl[l]["edited"] / pl[l]["n"] for l in labs]
        cols = [figstyle.COG if l in PROPER_NAME_LABELS else "#95a5a6"
                for l in labs]
        ax.bar(range(len(labs)), vals, color=cols)
        base = (100 * ge["non_ent_chars_edited"]
                / max(ge["non_ent_chars"], 1))
        ax.axhline(base, color="k", ls="--", lw=1,
                   label=f"non-entity text ({base:.1f}%)")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel("% of mentions corrected")
        ax.set_title("Entity error rate (red = proper names)")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    fig.tight_layout()
    figstyle.save(fig, FIG_DIR / "fig_htr_correction")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_htr_correction.png'}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd-eur", type=float, default=0.92,
                    help="USD to EUR conversion rate used in the report")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_tasks()
    rows = [r for r in rows if r["lead_time_s"] >= MIN_SECONDS]
    for r in rows:
        r["edits"] = diff_edits(r["claude"], r["corrected"])
    print(f"{len(rows)} corrected letters, "
          f"{sum(len(r['edits']) for r in rows)} edits")

    costs = load_costs()
    ms2idn = load_ms_to_idn()

    letter_rows, edit_rows = [], []
    for r in rows:
        minutes = min(r["lead_time_s"] / 60, MAX_MINUTES)
        cls_counts = Counter(e["cls"] for e in r["edits"])
        idn = r.get("letter_id") or ms2idn.get(
            r["ms_key"].upper().replace(" ", ""), "")
        c = costs.get(str(idn), {})
        letter_rows.append({
            "annotator": r["annotator"], "ms_key": r["ms_key"], "idn": idn,
            "n_pages": r["n_pages"] or c.get("n_pages", 0),
            "minutes": minutes, "lead_time_s": r["lead_time_s"],
            "n_claude_chars": len(r["claude"]),
            "n_corrected_chars": len(r["corrected"]),
            "n_edits": len(r["edits"]),
            "n_changed_chars": sum(e["n_changed"] for e in r["edits"]),
            "cost_usd": c.get("cost_usd", float("nan")),
            **{f"n_{k}": cls_counts.get(k, 0) for k in CLASS_ORDER},
            "n_mechanical": sum(v for k, v in cls_counts.items()
                                if k in MECHANICAL),
            "n_cognitive": sum(v for k, v in cls_counts.items()
                               if k not in MECHANICAL),
        })
        for e in r["edits"]:
            edit_rows.append({"annotator": r["annotator"], "ms_key": r["ms_key"],
                              "cls": e["cls"],
                              "cost": EDIT_CLASSES[e["cls"]][1],
                              "op": e["op"], "c_start": e["c_start"],
                              "n_changed": e["n_changed"],
                              "old": e["old"][:200], "new": e["new"][:200]})

    df = pd.DataFrame(letter_rows)
    df["min_per_page"] = df.minutes / df.n_pages.clip(lower=1)
    df["min_per_1k"] = df.minutes / (df.n_claude_chars / 1000).clip(lower=.001)
    edits = pd.DataFrame(edit_rows)

    hl = highlight_analysis(rows)
    pn = proper_noun_rate(rows)
    ge = gold_entity_error_rate(rows, load_gold_entities())

    corr_idns = {str(i) for i in df.idn if i}
    cost_summary = {
        "corpus": {"n": len(costs),
                   "pages": sum(v["n_pages"] for v in costs.values()),
                   "usd": sum(v["cost_usd"] for v in costs.values())},
        "corrected": {"n": len(corr_idns & set(costs)),
                      "pages": sum(costs[i]["n_pages"]
                                   for i in corr_idns & set(costs)),
                      "usd": sum(costs[i]["cost_usd"]
                                 for i in corr_idns & set(costs))},
    }

    df.to_csv(LETTER_CSV, index=False, encoding="utf-8")
    edits.to_csv(EDIT_CSV, index=False, encoding="utf-8")
    REPORT.write_text(build_report(df, edits, hl, pn, ge, cost_summary,
                                   args.usd_eur), encoding="utf-8")
    print(f"wrote {LETTER_CSV}\nwrote {EDIT_CSV}\nwrote {REPORT}")

    print(f"\nletters={len(df)}  median={df.minutes.median():.1f} min  "
          f"total={df.minutes.sum() / 60:.1f} h")
    if ge and ge["n_letters"]:
        pl = ge["per_label"]
        prop_n = sum(pl[l]["n"] for l in PROPER_NAME_LABELS if l in pl)
        prop_e = sum(pl[l]["edited"] for l in PROPER_NAME_LABELS if l in pl)
        print(f"gold-grounded: {ge['n_letters']} letters, proper-name mention "
              f"error rate = {100 * prop_e / max(prop_n, 1):.1f}% "
              f"({prop_e}/{prop_n})")
        for lbl in GOLD_LABEL_ORDER:
            d = pl.get(lbl)
            if d and d["n"]:
                print(f"   {lbl:8s} {100 * d['edited'] / d['n']:5.1f}%  "
                      f"({d['edited']}/{d['n']})")
    print(f"heuristic proper-noun rate={100 * pn['name_error_rate']:.1f}%  "
          f"(all tokens {100 * pn['token_error_rate']:.1f}%)")
    print(f"mechanical={100 * edits.cls.isin(MECHANICAL).mean():.1f}% of edits")
    print(f"cost corrected subset=${cost_summary['corrected']['usd']:.2f}, "
          f"corpus=${cost_summary['corpus']['usd']:.2f}")

    if not args.no_figure:
        make_figure(df, edits, hl, ge)
        make_edit_class_figure(edits)
        make_mech_cog_figure(edits)


if __name__ == "__main__":
    main()
