"""Post-correction error taxonomy for NER: mechanical vs cognitive cost.

Rationale
---------
Strict / lenient F1 tell us how far a system's output sits from the gold
standard, but not what it *costs* a historian to repair that output.  This
script re-scores the same predictions along the two axes the paper defines:

* **detectability** — is the error self-evident in the correction interface
  (``evident``), does it need the surrounding sentence (``semi``), or does
  the interface show nothing at all that signals a problem (``silent``)?
* **locality** — can it be repaired from the span string alone (``in_span``),
  from the letter text (``in_text``), or only by consulting the source scan,
  the annotation guidelines or prosopographical reference works
  (``external``)?

From those two axes each error class inherits a **cost** label:

* ``mechanical`` = evident/semi detectability *and* span- or text-local
  resolution -> a deterministic script or one keystroke in Label Studio.
* ``cognitive``  = silent detectability *and/or* external resolution ->
  the domain expert has to read, verify and decide.

Accounting
----------
The unit of measurement is the **edit event**: one action a corrector has to
perform.  Every gold span that is not strictly recovered and every predicted
span that is not aligned to a gold span produces exactly one event, so the
event set covers the whole strict error budget.  It is smaller than that
budget by construction, because an aligned-but-wrong pair (wrong boundary or
wrong label on the right characters) costs the corrector one repair while
strict scoring charges it twice:

    events + aligned_repairs == strict FN + strict FP

The gap between the two numbers is itself informative: it is the share of
the strict penalty that is double-counted from a post-correction point of
view.  It is asserted at run time.

One event per aligned pair also means that a pair wrong in *both* its label and
its boundary is charged to the label class alone.  Such events carry a
``hidden_boundary`` field naming the class the uncharged boundary repair would
have fallen into (``M2``, ``M3`` or ``C4``); §3 of the report aggregates it, so
the resulting understatement of mechanical labour can be quantified rather than
assumed.

Reads
-----
* ``NER_pipeline/eval/gold_561_merged.jsonl``
* ``NER_pipeline/eval/folds_writer_year.json``
* ``<pred-dir>/<model>_fold<k>_run0.jsonl``

Writes
------
* ``NER_pipeline/eval/error_taxonomy.csv``       (long form: model x fold x class x label)
* ``NER_pipeline/eval/error_taxonomy_events.csv``(one row per individual edit event)
* ``NER_pipeline/eval/error_taxonomy.md``        (paper-ready tables + examples)
* ``NER_pipeline/eval/figures/fig_taxonomy_*.{png,pdf}``

Scoring a non-default ``--pred-dir`` suffixes every output with the part of the
directory name after ``predictions_`` (as in ``compute_ner_metrics.py``), so
``--pred-dir NER_pipeline/eval/predictions_filtered`` writes
``error_taxonomy_filtered.csv``, ``error_taxonomy_filtered.md`` and
``fig_taxonomy_*_filtered.{png,pdf}``.

Usage
-----
    python NER_pipeline/error_taxonomy.py
    python NER_pipeline/error_taxonomy.py --no-figure
    python NER_pipeline/error_taxonomy.py \
        --pred-dir NER_pipeline/eval/predictions_filtered
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "NER_pipeline" / "eval"
GOLD_PATH = EVAL_DIR / "gold_561_merged.jsonl"
FOLDS_PATH = EVAL_DIR / "folds_writer_year.json"
PRED_DIR = EVAL_DIR / "predictions"
FIG_DIR = EVAL_DIR / "figures"

CSV_OUT = EVAL_DIR / "error_taxonomy.csv"
EVENTS_OUT = EVAL_DIR / "error_taxonomy_events.csv"
REPORT_OUT = EVAL_DIR / "error_taxonomy.md"

SEED = 42
RUN = 0
CONTEXT_CHARS = 55
EXAMPLES_PER_CLASS = 3
LABEL_MERGE = {"ART_NOM": "ART"}

MODELS = [
    "gliner2",
    "gliner2-ft",
    "dalembert",
    "camembertav2",
    "claude-opus-4.8",
    "qwen3-235b-a22b-2507",
    "qwen3-vl-30b-a3b-instruct",
]
FAMILY = {
    "gliner2": "zero-shot",
    "gliner2-ft": "fine-tuned",
    "dalembert": "fine-tuned",
    "camembertav2": "fine-tuned",
    "claude-opus-4.8": "prompted LLM",
    "qwen3-235b-a22b-2507": "prompted LLM",
    "qwen3-vl-30b-a3b-instruct": "prompted LLM",
}

# Labels whose mentions feed the downstream entity-clustering step; errors on
# these propagate into the network, hence they are reported separately.
IDENTITY_LABELS = {"PER", "LOC"}


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------
# code -> (short name, detectability, locality, scope, cost, repair action)
#
# ``F0`` is not a taxonomy class.  It is the material the validity filter
# deletes before the taxonomy is applied, and it is scored only so that the
# pre-filter run can report how much each system emits.  The taxonomy proper is
# the eight numbered classes below it.  (Same convention as ``A0`` in
# ``analyze_correction_time.py``: a leading zero marks a non-error class.)
TAXONOMY: dict[str, tuple[str, str, str, str, str, str]] = {
    "F0": ("malformed span", "evident", "in_span", "bounded", "mechanical",
           "deleted by a validity rule (offsets / alphabetic content / word edges)"),
    "M1": ("redundant span", "evident", "in_span", "bounded", "mechanical",
           "deduplicated by an overlap rule"),
    "M2": ("affix drift", "evident", "in_span", "bounded", "mechanical",
           "normalised by a strip rule (whitespace, punctuation, article, honorific)"),
    "M3": ("token drift", "semi", "in_text", "bounded", "mechanical",
           "one drag of the span handle in Label Studio"),
    "M4": ("decidable label flip", "semi", "in_text", "bounded", "mechanical",
           "one keystroke; PER/PER_NOM recoverable from capitalisation"),
    "C1": ("missed mention", "silent", "external", "unbounded", "cognitive",
           "re-read the letter (and the scan) to find what is not there"),
    "C2": ("spurious mention", "silent", "in_text", "bounded", "cognitive",
           "read the context and adjudicate against the guidelines"),
    "C3": ("semantic label confusion", "silent", "external", "bounded", "cognitive",
           "resolve metonymy / consult prosopography"),
    "C4": ("mention segmentation", "silent", "external", "bounded", "cognitive",
           "decide who or what the mention denotes, then re-cut the span"),
}
FILTER_CLASS = "F0"        # removed before the taxonomy applies
CLASS_ORDER = ["F0", "M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4"]
MECHANICAL = {c for c in CLASS_ORDER if TAXONOMY[c][4] == "mechanical"}
COGNITIVE = {c for c in CLASS_ORDER if TAXONOMY[c][4] == "cognitive"}
UNBOUNDED = {c for c in CLASS_ORDER if TAXONOMY[c][3] == "unbounded"}

# Classes where one edit event repairs an aligned gold/prediction pair, i.e.
# clears one strict FN *and* one strict FP at once.
PAIRED_CLASSES = {"M2", "M3", "M4", "C3", "C4"}


def active_classes(counts_per_model: dict[str, dict[str, int]]) -> list[str]:
    """CLASS_ORDER restricted to classes with at least one event.

    A class can be empty by construction rather than by accident: ``F0`` is
    emptied by the validity filter that runs before events are counted, so
    scoring ``predictions_filtered`` leaves it at zero for every system.
    Composition tables and figures report labour, so a class that can never
    cost an action is dropped from them.
    """
    return [c for c in CLASS_ORDER
            if any(counts.get(c, 0) for counts in counts_per_model.values())]

# Sub-typing for the two classes where the paper needs a finer grain.
SUBTYPES = {
    "C1": ("identity-critical (PER/LOC)", "other labels"),
    "C4": ("merge", "split", "identity-altering boundary"),
}


# ---------------------------------------------------------------------------
# French surface heuristics
# ---------------------------------------------------------------------------
# Tokens that may be stripped from a span edge without changing which entity
# the mention denotes: articles, contracted articles, prepositions,
# possessives, demonstratives and the honorifics used in 19th-century letters.
AFFIX_TOKENS = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
    "a", "en", "dans", "chez", "pour", "par", "sur", "et",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs",
    "ce", "cet", "cette", "ces",
    "monsieur", "messieurs", "madame", "mesdames", "mademoiselle",
    "mr", "mrs", "m", "mm", "mme", "mlle", "sieur", "sr",
    "monseigneur", "docteur", "dr", "general", "colonel", "baron", "baronne",
    "comte", "comtesse", "duc", "duchesse", "prince", "princesse",
    "roi", "reine", "saint", "st", "ste",
}
# Capitalised tokens that never carry referential identity on their own.
NON_IDENTITY_CAPS = AFFIX_TOKENS | {"the", "of"}

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _tokens(s: str) -> list[str]:
    return _WORD_RE.findall(s)


def _core(s: str) -> str:
    """Normalised referential core: lowercase, accent-free, edge affixes gone."""
    toks = [_strip_accents(t).lower() for t in _tokens(s)]
    while toks and toks[0] in AFFIX_TOKENS:
        toks.pop(0)
    while toks and toks[-1] in AFFIX_TOKENS:
        toks.pop()
    return " ".join(toks)


def _proper_tokens(s: str) -> frozenset[str]:
    """Capitalised, identity-bearing tokens of a span (the clustering key)."""
    out = set()
    for t in _tokens(s):
        if len(t) < 2 or not t[0].isupper():
            continue
        if _strip_accents(t).lower() in NON_IDENTITY_CAPS:
            continue
        out.add(_strip_accents(t).lower())
    return frozenset(out)


def _case_decidable_label(s: str) -> str:
    """Capitalisation rule separating a named person from a nominal reference."""
    return "PER" if _proper_tokens(s) else "PER_NOM"


def is_malformed(text: str, start: int, end: int) -> str | None:
    """Return the malformed-span reason, or None if the span is well formed.

    Mirrors ``apply_validity_filter.py`` so that the ``F0`` count in this
    report equals the number of spans that filter removes.
    """
    if start < 0 or end <= start or end > len(text):
        return "invalid_offsets"
    frag = text[start:end]
    if not any(c.isalpha() for c in frag):
        return "empty_content"
    if len(frag) <= 3:
        left_inside = start > 0 and text[start - 1].isalpha() and frag[0].isalpha()
        right_inside = end < len(text) and text[end].isalpha() and frag[-1].isalpha()
        if left_inside and right_inside:
            return "tiny_midword"
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_gold() -> dict[str, dict]:
    gold = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            spans = sorted({(int(s[0]), int(s[1]), LABEL_MERGE.get(str(s[2]), str(s[2])))
                            for s in r["gold_spans"]})
            gold[r["letter_id"]] = {
                "text": r["text"],
                "spans": spans,
                "sender": r.get("sender", ""),
                "year": r.get("year"),
            }
    return gold


def load_folds() -> tuple[dict[str, int], int]:
    payload = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    letter_fold = {r["letter_id"]: int(r["fold"]) for r in payload["letters"]}
    return letter_fold, int(payload.get("n_folds", max(letter_fold.values()) + 1))


def load_preds(model: str, fold: int,
               pred_dir: Path = PRED_DIR) -> dict[str, list[tuple[int, int, str]]]:
    path = pred_dir / f"{model}_fold{fold}_run{RUN}.jsonl"
    out: dict[str, list[tuple[int, int, str]]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            spans = set()
            for s in r.get("pred_spans", []):
                if len(s) != 3:
                    continue
                spans.add((int(s[0]), int(s[1]),
                           LABEL_MERGE.get(str(s[2]), str(s[2]))))
            out[r["letter_id"]] = sorted(spans)
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def align(gold: list[tuple[int, int, str]],
         pred: list[tuple[int, int, str]]) -> dict[int, int]:
    """Label-agnostic greedy best-IoU 1:1 alignment gold_idx -> pred_idx.

    Label-agnostic on purpose: a wrong label on the right characters is an
    error we want to *see*, not a miss plus a spurious span.
    """
    cands = []
    for gi, (gs, ge, gl) in enumerate(gold):
        for pi, (ps, pe, pl) in enumerate(pred):
            iou = _iou((gs, ge), (ps, pe))
            if iou <= 0:
                continue
            exact = (gs, ge) == (ps, pe)
            cands.append((exact and gl == pl, exact, gl == pl, iou, gi, pi))
    cands.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    mapping: dict[int, int] = {}
    for *_, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        mapping[gi] = pi
    return mapping


def _max_disjoint(spans: list[tuple[int, int, str]]) -> int:
    """Largest number of pairwise non-overlapping spans in the list.

    Needed because the gold standard permits *nested* annotations: an `ART`
    span such as « portrait de Charlotte Delavault » contains a `PER` span
    « Charlotte Delavault ».  A prediction that correctly recovers the inner
    mention overlaps two gold spans, which would otherwise be scored as a
    merge.  Counting only mutually disjoint spans keeps `C4` restricted to
    genuine segmentation failures, where a single span really does swallow
    two distinct mentions.
    """
    chosen_end = -1
    n = 0
    for s, e, _ in sorted(spans, key=lambda x: x[1]):
        if s >= chosen_end:
            n += 1
            chosen_end = e
    return n


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_letter(text: str,
                    gold: list[tuple[int, int, str]],
                    pred: list[tuple[int, int, str]]) -> list[dict]:
    """Return one event dict per required correction action for this letter.

    Event keys: cls, subtype, label, pred_label, start, end, surface.
    """
    events: list[dict] = []

    # --- Step 0: the deterministic validity filter runs before the human ----
    kept: list[tuple[int, int, str]] = []
    gold_strict = set(gold)
    for sp in pred:
        reason = is_malformed(text, sp[0], sp[1])
        if reason and sp not in gold_strict:
            events.append({"cls": FILTER_CLASS, "subtype": reason, "label": sp[2],
                           "pred_label": sp[2], "start": max(sp[0], 0),
                           "end": min(sp[1], len(text)),
                           "g_span": None, "p_span": sp})
        else:
            kept.append(sp)
    pred = kept

    mapping = align(gold, pred)
    matched_p = set(mapping.values())

    # Overlap graph, used to detect merges and splits.
    g_over_p: dict[int, list[int]] = defaultdict(list)
    p_over_g: dict[int, list[int]] = defaultdict(list)
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if min(g[1], p[1]) > max(g[0], p[0]):
                g_over_p[gi].append(pi)
                p_over_g[pi].append(gi)

    # --- Gold-side events ---------------------------------------------------
    for gi, (gs, ge, gl) in enumerate(gold):
        pi = mapping.get(gi)
        if pi is None:
            sub = ("identity-critical (PER/LOC)" if gl in IDENTITY_LABELS
                   else "other labels")
            events.append({"cls": "C1", "subtype": sub, "label": gl,
                           "pred_label": "", "start": gs, "end": ge,
                           "g_span": (gs, ge, gl), "p_span": None})
            continue

        ps, pe, pl = pred[pi]
        if (gs, ge, gl) == (ps, pe, pl):
            continue  # strict hit, no correction needed

        g_txt, p_txt = text[gs:ge], text[ps:pe]
        pair = {"g_span": (gs, ge, gl), "p_span": (ps, pe, pl)}

        # C4a merge: this prediction swallows several *distinct* gold
        # mentions.  Nested gold annotations do not count (see _max_disjoint).
        if _max_disjoint([gold[j] for j in p_over_g[pi]]) > 1:
            events.append({"cls": "C4", "subtype": "merge", "label": gl,
                           "pred_label": pl, "start": ps, "end": pe, **pair})
            continue
        # C4b split: this gold mention is cut into several disjoint pieces.
        if _max_disjoint([pred[j] for j in g_over_p[gi]]) > 1:
            events.append({"cls": "C4", "subtype": "split", "label": gl,
                           "pred_label": pl, "start": gs, "end": ge, **pair})
            continue

        if gl == pl:
            if _core(g_txt) == _core(p_txt):
                events.append({"cls": "M2", "subtype": "", "label": gl,
                               "pred_label": pl, "start": ps, "end": pe,
                               **pair})
            elif _proper_tokens(g_txt) == _proper_tokens(p_txt):
                events.append({"cls": "M3", "subtype": "", "label": gl,
                               "pred_label": pl, "start": ps, "end": pe,
                               **pair})
            else:
                # The identity key of the mention changed: a name token was
                # dropped or absorbed, so the corrector must decide who it is.
                events.append({"cls": "C4", "subtype": "identity-altering boundary",
                               "label": gl, "pred_label": pl,
                               "start": ps, "end": pe, **pair})
            continue

        # Labels disagree. One event per aligned pair is what keeps the identity
        # with the strict counts, but it means a pair that is wrong in both its
        # label and its boundary is charged to the label class only. Record the
        # class the uncharged boundary repair would have fallen into, so that
        # build_report can bound how much the instrument understates.
        hidden = ""
        if (gs, ge) != (ps, pe):
            if _core(g_txt) == _core(p_txt):
                hidden = "M2"
            elif _proper_tokens(g_txt) == _proper_tokens(p_txt):
                hidden = "M3"
            else:
                hidden = "C4"
        cls = ("M4" if {gl, pl} == {"PER", "PER_NOM"}
               and _case_decidable_label(g_txt) == gl else "C3")
        events.append({"cls": cls, "subtype": f"{pl}->{gl}", "label": gl,
                       "pred_label": pl, "start": ps, "end": pe,
                       "hidden_boundary": hidden, **pair})

    # --- Prediction-side events (nothing in the gold to align to) ----------
    for pi, (ps, pe, pl) in enumerate(pred):
        if pi in matched_p:
            continue
        dup = any(pj != pi and pred[pj][2] == pl
                  and min(pe, pred[pj][1]) > max(ps, pred[pj][0])
                  for pj in range(len(pred)))
        cls = "M1" if dup else "C2"
        sub = "overlapping same-label span" if dup else (
            "identity-critical (PER/LOC)" if pl in IDENTITY_LABELS else "other labels")
        events.append({"cls": cls, "subtype": sub, "label": pl,
                       "pred_label": pl, "start": ps, "end": pe,
                       "g_span": None, "p_span": (ps, pe, pl)})

    for ev in events:
        s, e = max(0, ev["start"]), min(len(text), ev["end"])
        ev["surface"] = text[s:e].replace("\n", " ") if e > s else ""
        ev.setdefault("hidden_boundary", "")
    return events


def snippet(text: str, s: int, e: int) -> str:
    s, e = max(0, s), min(len(text), e)
    a, b = max(0, s - CONTEXT_CHARS), min(len(text), e + CONTEXT_CHARS)
    left = text[a:s].replace("\n", " ")
    right = text[e:b].replace("\n", " ")
    mid = text[s:e].replace("\n", " ")
    return (("…" if a > 0 else "") + left + f"**[{mid}]**" + right
            + ("…" if b < len(text) else ""))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "—"


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation CI over the folds.

    Same convention as ``compute_ner_metrics.normal_ci``: with k = 5 folds the
    interval is wide and is reported as an indication of fold sensitivity,
    not as a population estimate.
    """
    k = len(values)
    if k < 2:
        return (values[0] if values else 0.0), 0.0
    mu = sum(values) / k
    var = sum((v - mu) ** 2 for v in values) / (k - 1)
    return mu, z * sqrt(var / k)


def _ci_cell(values: list[float]) -> str:
    mu, hw = mean_ci(values)
    return f"{mu:.2f} ± {hw:.2f}"


def build_report(rows: list[dict], per_model: dict, gold: dict,
                 n_letters: int, n_gold_spans: int,
                 fold_rates: dict) -> str:
    out: list[str] = []
    A = out.append

    A("# NER post-correction error taxonomy\n")
    A(f"Scored on all {n_letters} gold letters ({n_gold_spans} gold spans), "
      f"pooled over the {max(r['fold'] for r in rows) + 1} cross-validation "
      f"test folds (run {RUN}).\n")
    A("Every event below is one action a corrector has to take. Events cover "
      "the whole strict error budget but are fewer than it, because a wrong "
      "boundary or a wrong label on the right characters costs one repair "
      "while strict scoring charges it twice "
      "(`events + aligned repairs == strict FN + strict FP`).\n")

    A("## 1. The taxonomy\n")
    A("| code | error class | detectability | locality | search scope | cost | repair |")
    A("|---|---|---|---|---|---|---|")
    for c in CLASS_ORDER:
        name, det, loc, scope, cost, fix = TAXONOMY[c]
        A(f"| `{c}` | {name} | {det} | {loc} | {scope} | **{cost}** | {fix} |")
    A("")
    A("*Detectability*: `evident` = demonstrably wrong without reading the "
      "letter; `semi` = visible, but the surrounding sentence is needed; "
      "`silent` = the interface shows nothing that signals a problem.  "
      "*Locality*: `in_span` = repairable from the span string; `in_text` = "
      "from the letter; `external` = requires the scan, the guidelines or "
      "prosopographical reference works.  *Search scope*: `bounded` = an "
      "on-screen span triggers the action, so the work is proportional to the "
      "number of predictions; `unbounded` = nothing triggers the action, so "
      "the whole letter has to be re-read.\n")

    A("## 2. Post-correction burden per system\n")
    A("| system | family | events | mechanical | cognitive | % cognitive | "
      "mech./letter | cogn./letter | unbounded/letter | cognitive load ratio |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        d = per_model[m]
        tot = d["total"]
        A(f"| `{m}` | {FAMILY[m]} | {tot} | {d['mech']} | {d['cog']} | "
          f"{_pct(d['cog'], tot)} | {d['mech'] / n_letters:.2f} | "
          f"{d['cog'] / n_letters:.2f} | {d['unbounded'] / n_letters:.2f} | "
          f"{d['cog'] / n_gold_spans:.2f} |")
    A("")
    A("*unbounded/letter* is the subset of the cognitive events that no "
      "on-screen artefact points at, i.e. the mentions the corrector can only "
      "recover by re-reading the letter.\n")
    A(f"The *cognitive load ratio* normalises the cognitive events by the "
      f"{n_gold_spans} gold spans ({n_gold_spans / n_letters:.1f} per letter) "
      f"that a from-scratch pass would have to create by hand. It is a rough "
      f"proxy for the share of the manual burden that survives "
      f"post-correction; it is not bounded by 1, because a system can also "
      f"add spurious mentions the annotator never had to consider.\n")
    A("### Fold variance of the per-letter rates\n")
    A("Mean over the 5 test folds ± half-width of a 95% normal-approximation "
      "CI (same convention as the F1 tables; with k = 5 the interval is wide "
      "and indicates fold sensitivity rather than a population estimate).\n")
    A("| system | mechanical / letter | cognitive / letter | unbounded / letter |")
    A("|---|---|---|---|")
    for m in MODELS:
        fr = fold_rates[m]
        A(f"| `{m}` | {_ci_cell(fr['mech'])} | {_ci_cell(fr['cog'])} | "
          f"{_ci_cell(fr['unbounded'])} |")
    A("")
    A("## 3. Event composition per system\n")
    active = active_classes({m: per_model[m]["by_class"] for m in MODELS})
    dropped = [c for c in CLASS_ORDER if c not in active]
    if dropped:
        A("Classes with no events in this run are omitted from the two tables "
          "below: " + ", ".join(f"`{c}` ({TAXONOMY[c][0]})" for c in dropped) +
          ". `F0` is not a taxonomy class: it is what the validity filter "
          "deletes, so it is empty by construction whenever that filter has "
          "run over the predictions being scored.\n")
    header = "| system | " + " | ".join(f"`{c}`" for c in active) + " |"
    A(header)
    A("|---" * (len(active) + 1) + "|")
    for m in MODELS:
        cells = " | ".join(str(per_model[m]["by_class"].get(c, 0))
                           for c in active)
        A(f"| `{m}` | {cells} |")
    A("")

    A("### Share of each system's events (row-normalised)\n")
    A(header)
    A("|---" * (len(active) + 1) + "|")
    for m in MODELS:
        tot = per_model[m]["total"]
        cells = " | ".join(_pct(per_model[m]["by_class"].get(c, 0), tot)
                           for c in active)
        A(f"| `{m}` | {cells} |")
    A("")

    A("### Boundary repairs hidden inside label events\n")
    A("Charging one event per aligned pair is what makes "
      "`events + aligned repairs == strict FN + strict FP` hold, and it has a "
      "cost: a prediction wrong in both its label and its boundary is charged "
      "to the label class (`M4` or `C3`) and the boundary repair rides along "
      "uncounted. The columns below give the class that repair would have "
      "fallen into, so the table bounds how much this understates the burden.\n")
    A("| system | label events | hidden repair | share | `M2` | `M3` | `C4` | "
      "% of all events |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    lab_all = hid_all = 0
    kind_all: Counter = Counter()
    for m in MODELS:
        lab = [r for r in rows if r["model"] == m and r["cls"] in ("M4", "C3")]
        hid = [r for r in lab if r.get("hidden_boundary")]
        kind = Counter(r["hidden_boundary"] for r in hid)
        lab_all += len(lab)
        hid_all += len(hid)
        kind_all += kind
        A(f"| `{m}` | {len(lab)} | {len(hid)} | {_pct(len(hid), len(lab))} | "
          f"{kind['M2']} | {kind['M3']} | {kind['C4']} | "
          f"{_pct(len(hid), per_model[m]['total'])} |")
    A(f"| **all systems** | **{lab_all}** | **{hid_all}** | "
      f"**{_pct(hid_all, lab_all)}** | **{kind_all['M2']}** | "
      f"**{kind_all['M3']}** | **{kind_all['C4']}** | — |")
    A("")
    mech_hidden = kind_all["M2"] + kind_all["M3"]
    A(f"{_pct(mech_hidden, hid_all)} of the hidden repairs would themselves be "
      "mechanical, so the bias understates mechanical labour and is "
      "conservative with respect to the mechanical/cognitive split reported "
      "above.\n")

    A("## 4. Cognitive events on the identity-critical labels (PER, LOC)\n")
    A("These are the events that propagate into the entity-clustering step.\n")
    A("| system | missed PER/LOC | spurious PER/LOC | segmentation | "
      "semantic label | total | per letter |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        d = per_model[m]["identity"]
        tot = sum(d.values())
        A(f"| `{m}` | {d['C1']} | {d['C2']} | {d['C4']} | {d['C3']} | "
          f"{tot} | {tot / n_letters:.2f} |")
    A("")

    A("## 5. Sub-types\n")
    for cls in ("C1", "C4", "C3", "M4"):
        A(f"### `{cls}` — {TAXONOMY[cls][0]}\n")
        subs = per_model["_subtypes"][cls]
        keys = [k for k, _ in Counter(
            {k: sum(v.values()) for k, v in subs.items()}).most_common(8)]
        A("| system | " + " | ".join(f"`{k}`" for k in keys) + " |")
        A("|---" * (len(keys) + 1) + "|")
        for m in MODELS:
            cells = " | ".join(str(subs.get(k, {}).get(m, 0)) for k in keys)
            A(f"| `{m}` | {cells} |")
        A("")

    A("## 6. Worked examples\n")
    rng = random.Random(SEED)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["cls"]].append(r)

    def _readable(ev: dict) -> tuple:
        """Prefer short, identity-critical spans: they make clearer examples."""
        g, p = ev.get("g_span"), ev.get("p_span")
        width = max((g[1] - g[0]) if g else 0, (p[1] - p[0]) if p else 0)
        return (0 if ev["label"] in IDENTITY_LABELS else 1,
                0 if width <= 45 else 1, rng.random())

    for c in CLASS_ORDER:
        pool = by_class.get(c, [])
        if not pool:
            continue
        name, det, loc, scope, cost, fix = TAXONOMY[c]
        A(f"### `{c}` {name} — {cost} ({det} / {loc} / {scope})\n")
        A(f"{len(pool)} events across all systems. Repair: {fix}.\n")
        for ev in sorted(pool, key=_readable)[:EXAMPLES_PER_CLASS]:
            text = gold[ev["letter_id"]]["text"]
            g, p = ev.get("g_span"), ev.get("p_span")

            def _show(sp):
                # Keep whitespace visible: for affix drift the difference
                # between the two sides is often exactly a space or newline.
                s = text[sp[0]:sp[1]].replace("\n", "⏎").replace("\t", "→")
                return f"«{s}» `{sp[2]}`"

            gold_s = _show(g) if g else "— *(nothing in the gold)*"
            pred_s = _show(p) if p else "— *(no prediction)*"
            extra = f" · *{ev['subtype']}*" if ev["subtype"] else ""
            A(f"- `{ev['model']}` · letter `{ev['letter_id']}`{extra}")
            A(f"  - gold: {gold_s}")
            A(f"  - predicted: {pred_s}")
            A(f"  - context: {snippet(text, ev['start'], ev['end'])}")
        A("")
    return "\n".join(out)


def make_figure(per_model: dict, n_letters: int, counts_csv: Path = CSV_OUT,
                suffix: str = "") -> None:
    """Render the figures by delegating to ``plot_error_taxonomy``.

    All plotting lives in that module so there is a single source of truth for
    the figures; it reads the long-form counts CSV that has just been written.
    Each visualisation is emitted as its own PNG/PDF rather than as panels of
    a combined figure.
    """
    try:
        import plot_error_taxonomy  # local import: it imports from this module
    except ImportError as exc:
        print(f"could not import plot_error_taxonomy ({exc}) — skipping figures")
        return
    plot_error_taxonomy.main(counts_csv=counts_csv, suffix=suffix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--pred-dir", type=str, default=str(PRED_DIR),
                    help="directory of prediction JSONLs to score "
                         "(default: NER_pipeline/eval/predictions)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    if not pred_dir.exists():
        raise SystemExit(f"No predictions dir: {pred_dir}")
    # Derive output filenames from the pred-dir name, as compute_ner_metrics.py
    # does: predictions -> error_taxonomy.*, predictions_filtered -> *_filtered.*
    suffix = ""
    if pred_dir.name != PRED_DIR.name:
        suffix = "_" + pred_dir.name.replace("predictions_", "")
    csv_out = EVAL_DIR / f"error_taxonomy{suffix}.csv"
    events_out = EVAL_DIR / f"error_taxonomy{suffix}_events.csv"
    report_out = EVAL_DIR / f"error_taxonomy{suffix}.md"
    print(f"scoring {pred_dir.name} -> error_taxonomy{suffix}.*")

    gold = load_gold()
    letter_fold, n_folds = load_folds()

    events: list[dict] = []
    strict_check: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for model in MODELS:
        for fold in range(n_folds):
            preds = load_preds(model, fold, pred_dir)
            fold_letters = [lid for lid, f in letter_fold.items()
                            if f == fold and lid in gold]
            for lid in sorted(fold_letters):
                text = gold[lid]["text"]
                g = gold[lid]["spans"]
                p = preds.get(lid, [])
                for ev in classify_letter(text, g, p):
                    ev.update({"model": model, "fold": fold, "letter_id": lid})
                    events.append(ev)
                gset, pset = set(g), set(p)
                strict_check[model][0] += len(gset - pset)   # FN
                strict_check[model][1] += len(pset - gset)   # FP

    n_letters = len({lid for lid in letter_fold if lid in gold})
    n_gold_spans = sum(len(gold[lid]["spans"]) for lid in gold
                       if lid in letter_fold)
    fold_sizes = Counter(f for lid, f in letter_fold.items() if lid in gold)

    # Aggregate ------------------------------------------------------------
    per_model: dict = {m: {"by_class": Counter(), "mech": 0, "cog": 0,
                           "total": 0, "paired": 0, "unbounded": 0,
                           "identity": {"C1": 0, "C2": 0, "C3": 0, "C4": 0}}
                       for m in MODELS}
    subtypes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    long_rows: Counter = Counter()
    fold_counts: dict[tuple[str, int], Counter] = defaultdict(Counter)

    for ev in events:
        m, c = ev["model"], ev["cls"]
        d = per_model[m]
        d["by_class"][c] += 1
        d["total"] += 1
        d["mech" if c in MECHANICAL else "cog"] += 1
        fold_counts[(m, ev["fold"])]["mech" if c in MECHANICAL else "cog"] += 1
        if c in UNBOUNDED:
            d["unbounded"] += 1
            fold_counts[(m, ev["fold"])]["unbounded"] += 1
        if c in PAIRED_CLASSES:
            d["paired"] += 1
        if c in COGNITIVE and ev["label"] in IDENTITY_LABELS:
            d["identity"][c] += 1
        if ev["subtype"]:
            subtypes[c][ev["subtype"]][m] += 1
        long_rows[(m, ev["fold"], c, ev["subtype"], ev["label"])] += 1
    per_model["_subtypes"] = subtypes

    # Per-fold rates (events per letter within each test fold).
    fold_rates = {m: {k: [fold_counts[(m, f)][k] / fold_sizes[f]
                          for f in sorted(fold_sizes)]
                      for k in ("mech", "cog", "unbounded")}
                  for m in MODELS}

    # Sanity check: events + aligned repairs must exhaust the strict error set.
    for m in MODELS:
        fn, fp = strict_check[m]
        got = per_model[m]["total"]
        paired = per_model[m]["paired"]
        flag = "OK" if got + paired == fn + fp else "MISMATCH"
        print(f"{m:28s} events={got:5d}  paired={paired:5d}  "
              f"strictFN+FP={fn + fp:5d}  [{flag}]")

    # Write CSVs -----------------------------------------------------------
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "fold", "class", "class_name", "subtype",
                    "label", "detectability", "locality", "scope", "cost", "n"])
        for (m, fold, c, sub, lbl), n in sorted(long_rows.items()):
            name, det, loc, scope, cost, _ = TAXONOMY[c]
            w.writerow([m, FAMILY[m], fold, c, name, sub, lbl, det, loc,
                        scope, cost, n])
    print(f"wrote {csv_out}")

    with events_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "letter_id", "class", "class_name",
                    "subtype", "cost", "scope", "gold_label", "pred_label",
                    "start", "end", "hidden_boundary", "surface"])
        for ev in events:
            w.writerow([ev["model"], ev["fold"], ev["letter_id"], ev["cls"],
                        TAXONOMY[ev["cls"]][0], ev["subtype"],
                        TAXONOMY[ev["cls"]][4], TAXONOMY[ev["cls"]][3],
                        ev["label"], ev["pred_label"],
                        ev["start"], ev["end"],
                        ev.get("hidden_boundary", ""), ev["surface"]])
    print(f"wrote {events_out}  ({len(events)} events)")

    report_out.write_text(
        build_report(events, per_model, gold, n_letters, n_gold_spans,
                     fold_rates),
        encoding="utf-8")
    print(f"wrote {report_out}")

    if not args.no_figure:
        make_figure(per_model, n_letters, csv_out, suffix)


if __name__ == "__main__":
    main()
