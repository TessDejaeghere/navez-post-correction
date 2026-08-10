"""Qualitative NER error analysis across all seven models on fold 0.

Purpose
-------
Complements the aggregate metrics with concrete French-language examples
of the error modes that drive the numbers, for the paper's appendix and
for internal error-diagnosis discussions.

Reads
-----
* ``NER_pipeline/eval/gold_561_merged.jsonl``       (gold spans + text)
* ``NER_pipeline/eval/folds_writer_year.json``      (fold membership)
* ``NER_pipeline/eval/predictions/<model>_fold0_run0.jsonl``  (predictions)

Writes
------
* ``NER_pipeline/eval/error_analysis.md``  (paper-ready qualitative report)

Error categories reported
-------------------------
A. Fine-tuning saved us:      gold span caught only by fine-tuned systems
B. Boundary drift (fine-tuned): lenient hit but not strict hit
C. Label collisions:           correct boundaries, wrong label
D. Universally missed:         no system caught the gold span
E. Universally spurious:       every fine-tuned system predicted a span
                                that is NOT in the gold (annotation gap?)

Usage
-----
    python NER_pipeline/error_analysis.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "NER_pipeline" / "eval"
GOLD_PATH = EVAL_DIR / "gold_561_merged.jsonl"
FOLDS_PATH = EVAL_DIR / "folds_writer_year.json"
PRED_DIR = EVAL_DIR / "predictions"
REPORT = EVAL_DIR / "error_analysis.md"

TARGET_FOLD = 0
CONTEXT_CHARS = 60          # chars of context around a span in examples
MAX_EXAMPLES_PER_LABEL = 2  # per (category, label) subsection
SEED = 42
LABEL_MERGE = {"ART_NOM": "ART"}
IOU_THRESHOLD = 0.5         # aligned with compute_ner_metrics LENIENT_IOU_THRESHOLD

# Labels appear in this order in the per-label sub-tables and example blocks.
# Frequency-descending, following the project schema.
LABEL_ORDER = ["PER", "LOC", "ORG", "EVT", "ART", "PER_NOM"]

MODELS = [
    "gliner2",                     # zero-shot
    "gliner2-ft",                  # fine-tuned span extractor
    "dalembert",                   # fine-tuned encoder
    "camembertav2",                # fine-tuned encoder
    "claude-opus-4.8",             # prompted LLM
    "qwen3-235b-a22b-2507",        # prompted LLM
    "qwen3-vl-30b-a3b-instruct",   # prompted LLM
]
FT_MODELS = {"gliner2-ft", "dalembert", "camembertav2"}
PROMPTED_LLMS = {"claude-opus-4.8", "qwen3-235b-a22b-2507",
                 "qwen3-vl-30b-a3b-instruct"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_gold_fold(fold: int) -> dict[str, dict]:
    letter_fold = {r["letter_id"]: r["fold"]
                   for r in json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["letters"]}
    gold = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if letter_fold.get(r["letter_id"]) != fold:
                continue
            gold[r["letter_id"]] = {
                "text": r["text"],
                "spans": frozenset((int(s[0]), int(s[1]),
                                    LABEL_MERGE.get(s[2], s[2]))
                                   for s in r["gold_spans"]),
            }
    return gold


def load_preds(model: str, fold: int) -> dict[str, frozenset]:
    path = PRED_DIR / f"{model}_fold{fold}_run0.jsonl"
    out: dict[str, frozenset] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            spans = set()
            for s in r.get("pred_spans", []):
                if len(s) != 3:
                    continue
                lbl = LABEL_MERGE.get(str(s[2]), str(s[2]))
                spans.add((int(s[0]), int(s[1]), lbl))
            out[r["letter_id"]] = frozenset(spans)
    return out


# ---------------------------------------------------------------------------
# Categorisation helpers
# ---------------------------------------------------------------------------
def _snippet(text: str, s: int, e: int) -> str:
    """Return `... left CTX [SPAN] right CTX ...` with the span in brackets."""
    a = max(0, s - CONTEXT_CHARS)
    b = min(len(text), e + CONTEXT_CHARS)
    left  = text[a:s].replace("\n", " ")
    right = text[e:b].replace("\n", " ")
    span  = text[s:e].replace("\n", " ")
    prefix = "…" if a > 0 else ""
    suffix = "…" if b < len(text) else ""
    return f"{prefix}{left}**[{span}]**{right}{suffix}"


def _span_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union else 0.0


def match_gold_to_preds(gold: tuple, preds: frozenset) -> str:
    """Classify a single gold span w.r.t. a model's predictions.

    Returns one of: 'strict_hit', 'boundary_only', 'label_only',
                    'boundary_and_label', 'missed'.

    "boundary_only" and "boundary_and_label" require IoU >= IOU_THRESHOLD
    (0.5), matching the lenient F1 definition used by compute_ner_metrics.py.
    """
    gs, ge, gl = gold
    if (gs, ge, gl) in preds:
        return "strict_hit"
    same_offsets_diff_label = any(ps == gs and pe == ge and pl != gl
                                  for ps, pe, pl in preds)
    if same_offsets_diff_label:
        return "label_only"
    same_label_iou = any(pl == gl and _span_iou(gs, ge, ps, pe) >= IOU_THRESHOLD
                         for ps, pe, pl in preds)
    if same_label_iou:
        return "boundary_only"
    diff_label_iou = any(pl != gl and _span_iou(gs, ge, ps, pe) >= IOU_THRESHOLD
                         for ps, pe, pl in preds)
    if diff_label_iou:
        return "boundary_and_label"
    return "missed"


def predicted_but_not_gold(preds: frozenset, gold: frozenset) -> set:
    """Return predicted spans that have no strict-hit and no lenient
    (IoU>=0.5, same label) match in gold."""
    result = set()
    for ps, pe, pl in preds:
        strict = (ps, pe, pl) in gold
        lenient = any(pl == gl and _span_iou(ps, pe, gs, ge) >= IOU_THRESHOLD
                      for gs, ge, gl in gold)
        if not strict and not lenient:
            result.add((ps, pe, pl))
    return result


# ---------------------------------------------------------------------------
# Category collectors
# ---------------------------------------------------------------------------
def collect_examples(gold: dict, preds_by_model: dict[str, dict]) -> dict:
    """Walk all letters and gold spans; bucket into named categories."""
    buckets: dict[str, list[dict]] = defaultdict(list)

    for lid, entry in gold.items():
        text = entry["text"]
        gspans = entry["spans"]

        # per-model classification of every gold span
        per_gold_status = {g: {} for g in gspans}
        for m in MODELS:
            preds = preds_by_model[m].get(lid, frozenset())
            for g in gspans:
                per_gold_status[g][m] = match_gold_to_preds(g, preds)

        for g, status_by_model in per_gold_status.items():
            gs, ge, gl = g
            snippet = _snippet(text, gs, ge)

            hits = {m for m, s in status_by_model.items() if s == "strict_hit"}
            lenient_hits = {m for m, s in status_by_model.items()
                            if s in ("strict_hit", "boundary_only")}

            # -- A. Fine-tuning saved us -------------------------------------
            # Gold caught by ≥2 fine-tuned systems, missed (or wrong label) by
            # ALL prompted LLMs.
            ft_hits = hits & FT_MODELS
            llm_status = {m: status_by_model[m] for m in PROMPTED_LLMS}
            llm_all_bad = all(s in ("missed", "label_only",
                                    "boundary_and_label")
                              for s in llm_status.values())
            if len(ft_hits) >= 2 and llm_all_bad:
                buckets["A_ft_saved_us"].append({
                    "letter": lid, "gold": g, "label": gl,
                    "snippet": snippet,
                    "status": status_by_model,
                })

            # -- B. Boundary drift on fine-tuned -----------------------------
            # ≥2 fine-tuned systems have a lenient hit but ≤1 has a strict hit.
            ft_len = lenient_hits & FT_MODELS
            if len(ft_len) >= 2 and len(ft_hits) <= 1:
                buckets["B_boundary_drift"].append({
                    "letter": lid, "gold": g, "label": gl,
                    "snippet": snippet,
                    "status": status_by_model,
                    "ft_predicted_spans": {
                        m: _find_matching_pred_span(preds_by_model[m].get(lid, frozenset()), g)
                        for m in ft_len
                    },
                })

            # -- C. Label collisions -----------------------------------------
            # ≥2 systems have EXACT boundaries with the WRONG label.
            label_only = {m for m, s in status_by_model.items()
                          if s == "label_only"}
            if len(label_only) >= 2:
                buckets["C_label_collision"].append({
                    "letter": lid, "gold": g, "label": gl,
                    "snippet": snippet,
                    "status": status_by_model,
                    "wrong_labels": {
                        m: _find_pred_label_at_offsets(
                            preds_by_model[m].get(lid, frozenset()), gs, ge)
                        for m in label_only
                    },
                })

            # -- D. Universally missed ---------------------------------------
            if all(s == "missed" for s in status_by_model.values()):
                buckets["D_universally_missed"].append({
                    "letter": lid, "gold": g, "label": gl,
                    "snippet": snippet,
                    "status": status_by_model,
                })

        # -- E. Universally spurious (walk predicted, not gold) --------------
        # A predicted span common to ≥2 fine-tuned systems that is not in gold.
        pred_by_ft = {m: preds_by_model[m].get(lid, frozenset())
                      for m in FT_MODELS}
        span_votes: dict[tuple, set] = defaultdict(set)
        for m, ps in pred_by_ft.items():
            for span in predicted_but_not_gold(ps, gspans):
                span_votes[span].add(m)
        for span, voters in span_votes.items():
            if len(voters) >= 2:
                ps, pe, pl = span
                buckets["E_universally_spurious"].append({
                    "letter": lid, "predicted": span, "label": pl,
                    "snippet": _snippet(text, ps, pe),
                    "voters": sorted(voters),
                })
        # -- F. Malformed spurious spans -------------------------------------
        # For each model: predicted spans that fail a validity check AND have
        # no strict/lenient match in gold (i.e. spurious AND ill-formed).
        for m in MODELS:
            preds = preds_by_model[m].get(lid, frozenset())
            for span in predicted_but_not_gold(preds, gspans):
                ps, pe, pl = span
                tag = check_malformed(text, ps, pe)
                if tag is None:
                    continue
                # Snippet uses safe offset clipping (invalid_offsets may be OOB).
                safe_e = min(pe, len(text))
                safe_s = max(0, ps)
                buckets["F_malformed"].append({
                    "letter": lid, "predicted": span, "label": pl,
                    "malformed_type": tag,
                    "model": m,
                    "snippet": _snippet(text, safe_s, safe_e),
                })
    return buckets


def _find_matching_pred_span(preds: frozenset, gold: tuple) -> tuple | None:
    gs, ge, gl = gold
    for ps, pe, pl in preds:
        if pl == gl and _span_iou(gs, ge, ps, pe) >= IOU_THRESHOLD:
            return (ps, pe, pl)
    return None


def _find_pred_label_at_offsets(preds: frozenset, gs: int, ge: int) -> str | None:
    for ps, pe, pl in preds:
        if ps == gs and pe == ge:
            return pl
    return None


# ---------------------------------------------------------------------------
# Validity checks — is the span readable at all?
# ---------------------------------------------------------------------------
# These detect spans that do not denote a readable substring of the letter.
# They say nothing about whether a well-formed span names a real entity: a
# prediction that quotes the letter correctly but invents an entity — what is
# usually meant by "hallucination" — passes every check below untouched.
#
# Operationalised as three checks over a predicted span (start, end, label)
# against the underlying letter text:
#
#   invalid_offsets : end > len(text) OR start < 0 OR end <= start
#                     Only a decoder LLM emitting offsets by text-completion
#                     can produce this; encoders cannot by construction.
#   empty_content   : text[start:end] contains no alphabetic characters
#                     (whitespace-only, punctuation-only, or empty).
#   tiny_midword    : span length <= 3 chars AND both edges are strictly
#                     inside a word (BPE/tokeniser artifact).

def check_malformed(text: str, s: int, e: int) -> str | None:
    """Return the failed-validity tag if the span is unreadable, else None."""
    if s < 0 or e <= s or e > len(text):
        return "invalid_offsets"
    span = text[s:e]
    if not any(c.isalpha() for c in span):
        return "empty_content"
    if (e - s) <= 3:
        prev_c = text[s - 1] if s > 0 else " "
        first_c = span[0]
        last_c = span[-1]
        next_c = text[e] if e < len(text) else " "
        starts_midword = prev_c.isalnum() and first_c.isalnum()
        ends_midword   = last_c.isalnum() and next_c.isalnum()
        if starts_midword and ends_midword:
            return "tiny_midword"
    return None


MALFORMED_TYPES = ["invalid_offsets", "empty_content", "tiny_midword"]


# ---------------------------------------------------------------------------
# Aggregate confusion table
# ---------------------------------------------------------------------------
def aggregate_counts(gold: dict, preds_by_model: dict[str, dict]) -> dict:
    """Return per-model counts of gold-span classification categories."""
    counts = {m: defaultdict(int) for m in MODELS}
    for lid, entry in gold.items():
        for g in entry["spans"]:
            for m in MODELS:
                preds = preds_by_model[m].get(lid, frozenset())
                counts[m][match_gold_to_preds(g, preds)] += 1
    return counts


def aggregate_malformed(gold: dict, preds_by_model: dict[str, dict]) -> dict:
    """Return per-model counts of each malformed-span type, over ALL predictions
    (whether or not they match gold).  A gold-matched span can still fail a
    check (e.g. a valid tiny word like 'or'); we count them all here and
    let the qualitative examples show which are genuinely spurious."""
    counts = {m: defaultdict(int) for m in MODELS}
    for lid, entry in gold.items():
        text = entry["text"]
        for m in MODELS:
            for s, e, _lbl in preds_by_model[m].get(lid, frozenset()):
                tag = check_malformed(text, s, e)
                if tag is not None:
                    counts[m][tag] += 1
                    counts[m]["_total_preds"] += 0  # placeholder; filled below
    # add total predictions per model for percentage reporting
    for m in MODELS:
        n_preds = sum(len(v) for v in preds_by_model[m].values())
        counts[m]["_total_preds"] = n_preds
    return counts


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------
def emit_report(gold: dict, buckets: dict, counts: dict,
                malformed: dict | None = None) -> None:
    lines: list[str] = []
    lines.append("# NER error analysis — fold 0")
    lines.append("")
    lines.append(f"Fold 0 contains {len(gold)} letters with "
                 f"{sum(len(v['spans']) for v in gold.values())} gold spans.")
    lines.append(f"Snippets: `[SPAN]` is the gold entity, surrounded by "
                 f"±{CONTEXT_CHARS} chars of context (line-breaks stripped).")
    lines.append("")

    # --- summary table ---
    lines.append("## Aggregate per-model status of gold spans")
    lines.append("")
    lines.append("| Model | strict hit | boundary only | label only | boundary+label | missed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in MODELS:
        c = counts[m]
        total = sum(c.values())
        def pct(k: str) -> str:
            n = c.get(k, 0)
            return f"{n} ({n / total:.0%})" if total else str(n)
        lines.append(f"| `{m}` | {pct('strict_hit')} | "
                     f"{pct('boundary_only')} | {pct('label_only')} | "
                     f"{pct('boundary_and_label')} | {pct('missed')} |")
    lines.append("")

    # --- malformed-span aggregate table ---
    if malformed is not None:
        lines.append("## Malformed spans — per-model validity-check counts")
        lines.append("")
        lines.append("Counted over ALL predicted spans (gold-matched or not). "
                     "These checks ask whether a span denotes a readable "
                     "substring of the letter, not whether it names a real "
                     "entity: a well-formed span that invents an entity is "
                     "counted as spurious, not here. Checks: "
                     "`invalid_offsets` (start<0, end<=start, or "
                     "end>len(text) — LLM-only failure mode); `empty_content` "
                     "(no alphabetic characters in the span); `tiny_midword` "
                     "(<=3 chars, both edges strictly inside a word).")
        lines.append("")
        lines.append("| Model | total preds | invalid_offsets | empty_content | "
                     "tiny_midword | total malformed |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for m in MODELS:
            hc = malformed[m]
            n_preds = hc.get("_total_preds", 0)
            io = hc.get("invalid_offsets", 0)
            ec = hc.get("empty_content", 0)
            tm = hc.get("tiny_midword", 0)
            tot = io + ec + tm
            pct_str = f" ({tot / n_preds:.1%})" if n_preds else ""
            lines.append(f"| `{m}` | {n_preds} | {io} | {ec} | {tm} | "
                         f"{tot}{pct_str} |")
        lines.append("")

    # --- qualitative examples ---
    section_titles = {
        "A_ft_saved_us":
            "A. Fine-tuning saved us — caught by ≥ 2 fine-tuned systems, "
            "missed / mislabelled by ALL prompted LLMs",
        "B_boundary_drift":
            "B. Boundary drift on fine-tuned systems — lenient hit but not "
            "strict hit for ≥ 2 fine-tuned systems (illustrates the "
            "strict/lenient gap the paper reports)",
        "C_label_collision":
            "C. Label collisions — ≥ 2 systems have the exact boundaries but "
            "the wrong label (typically PER ↔ PER_NOM, LOC ↔ ORG)",
        "D_universally_missed":
            "D. Universally missed — no system caught the gold span "
            "(diagnostic for annotation quirks or genuinely hard cases)",
        "E_universally_spurious":
            "E. Universally spurious — ≥ 2 fine-tuned systems agree on a "
            "span that is NOT in the gold (possible gold-annotation gap; "
            "label is the *predicted* label, not a gold label)",
    }

    rng = random.Random(SEED)
    for key, title in section_titles.items():
        examples = buckets.get(key, [])
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"*{len(examples)} candidate spans in total; showing up "
                     f"to {MAX_EXAMPLES_PER_LABEL} per label (seed={SEED}).*")
        lines.append("")

        # per-label count table
        by_label: dict[str, list[dict]] = defaultdict(list)
        for ex in examples:
            by_label[ex.get("label", "?")].append(ex)
        lines.append("| Label | " + " | ".join(LABEL_ORDER) + " | other |")
        lines.append("|---|" + "|".join("---:" for _ in LABEL_ORDER) + "|---:|")
        counts_row = [str(len(by_label.get(l, []))) for l in LABEL_ORDER]
        other = sum(len(v) for k, v in by_label.items() if k not in LABEL_ORDER)
        lines.append("| n | " + " | ".join(counts_row) + f" | {other} |")
        lines.append("")

        if not examples:
            lines.append("_(none)_"); lines.append(""); continue

        # examples grouped per label
        for lbl in LABEL_ORDER:
            lbl_examples = by_label.get(lbl, [])
            if not lbl_examples:
                continue
            lines.append(f"### {lbl}  ({len(lbl_examples)} spans, showing "
                         f"{min(len(lbl_examples), MAX_EXAMPLES_PER_LABEL)})")
            lines.append("")
            sample = rng.sample(lbl_examples,
                                min(MAX_EXAMPLES_PER_LABEL,
                                    len(lbl_examples)))
            for ex in sample:
                _emit_example(lines, gold, ex)

    # --- Category F (malformed spurious spans) — grouped by type × model ---
    malformed_examples = buckets.get("F_malformed", [])
    lines.append("## F. Malformed spans — spurious spans that also fail a "
                 "validity check (see the aggregate table above for "
                 "definitions)")
    lines.append("")
    lines.append(f"*{len(malformed_examples)} spurious ill-formed spans in "
                 f"total; showing up to {MAX_EXAMPLES_PER_LABEL} per "
                 f"(type × model) cell (seed={SEED}).*")
    lines.append("")

    # type × model count table
    lines.append("| Model | " + " | ".join(MALFORMED_TYPES) + " |")
    lines.append("|---|" + "|".join("---:" for _ in MALFORMED_TYPES) + "|")
    by_type_model: dict[tuple, list[dict]] = defaultdict(list)
    for ex in malformed_examples:
        by_type_model[(ex["malformed_type"], ex["model"])].append(ex)
    for m in MODELS:
        row = [str(len(by_type_model.get((t, m), []))) for t in MALFORMED_TYPES]
        lines.append(f"| `{m}` | " + " | ".join(row) + " |")
    lines.append("")

    for t in MALFORMED_TYPES:
        t_examples = [ex for ex in malformed_examples
                      if ex["malformed_type"] == t]
        if not t_examples:
            continue
        lines.append(f"### {t}  ({len(t_examples)} spans)")
        lines.append("")
        # group by model within this type
        by_m: dict[str, list[dict]] = defaultdict(list)
        for ex in t_examples:
            by_m[ex["model"]].append(ex)
        for m in MODELS:
            m_examples = by_m.get(m, [])
            if not m_examples:
                continue
            n_show = min(len(m_examples), MAX_EXAMPLES_PER_LABEL)
            lines.append(f"#### `{m}`  ({len(m_examples)} spans, showing {n_show})")
            lines.append("")
            sample = rng.sample(m_examples, n_show)
            for ex in sample:
                _emit_example(lines, gold, ex)

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(REPO)}")


def _emit_example(lines: list[str], gold: dict, ex: dict) -> None:
    """Emit one example block; shared by all category sections."""
    lid = ex["letter"]
    lines.append(f"**Letter `{lid}`**")
    if "gold" in ex:
        gs, ge, gl = ex["gold"]
        lines.append(f"- gold: `{gl}` at chars [{gs}:{ge}]")
    if "predicted" in ex:
        ps, pe, pl = ex["predicted"]
        lines.append(f"- predicted (not in gold): `{pl}` at chars [{ps}:{pe}]")
        if "voters" in ex:
            lines.append(f"- agreeing models: "
                         + ", ".join(f"`{m}`" for m in ex["voters"]))
        elif "model" in ex:
            lines.append(f"- model: `{ex['model']}`")
        if "malformed_type" in ex:
            lines.append(f"- validity check failed: `{ex['malformed_type']}`")
    lines.append(f"- context: {ex['snippet']}")
    if "status" in ex:
        lines.append("- per-model status of this gold span:")
        for m in MODELS:
            s = ex["status"].get(m, "?")
            lines.append(f"    - `{m}`: {s}")
    if "ft_predicted_spans" in ex:
        lines.append("- fine-tuned systems' actual predicted spans:")
        for m, span in ex["ft_predicted_spans"].items():
            if span is None:
                continue
            ps, pe, pl = span
            txt = gold[lid]["text"][ps:pe].replace("\n", " ")
            lines.append(f"    - `{m}`: `{pl}` [{ps}:{pe}] = "
                         f"'{txt}'")
    if "wrong_labels" in ex:
        lines.append("- systems' wrong labels at the exact boundaries:")
        for m, lbl in ex["wrong_labels"].items():
            lines.append(f"    - `{m}`: predicted `{lbl}`")
    lines.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    gold = load_gold_fold(TARGET_FOLD)
    print(f"Loaded {len(gold)} gold letters on fold {TARGET_FOLD}")

    preds_by_model: dict[str, dict] = {}
    for m in MODELS:
        preds_by_model[m] = load_preds(m, TARGET_FOLD)
        print(f"  {m:<28} {len(preds_by_model[m])} letters, "
              f"{sum(len(v) for v in preds_by_model[m].values())} pred spans")

    counts = aggregate_counts(gold, preds_by_model)
    malformed = aggregate_malformed(gold, preds_by_model)
    buckets = collect_examples(gold, preds_by_model)

    print("\nCategory counts (before sub-sampling):")
    for key in ("A_ft_saved_us", "B_boundary_drift", "C_label_collision",
                "D_universally_missed", "E_universally_spurious",
                "F_malformed"):
        print(f"  {key:<25} {len(buckets.get(key, []))}")

    print("\nMalformed-span counts per model:")
    for m in MODELS:
        io = malformed[m].get("invalid_offsets", 0)
        ec = malformed[m].get("empty_content", 0)
        tm = malformed[m].get("tiny_midword", 0)
        print(f"  {m:<28} invalid={io}  empty={ec}  midword={tm}")

    emit_report(gold, buckets, counts, malformed)


if __name__ == "__main__":
    main()
