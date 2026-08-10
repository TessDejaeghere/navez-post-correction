#!/usr/bin/env python3
"""
Compute inter-annotator agreement (IAA) for Label Studio NER exports.

Span-level precision / recall / F1 between every pair of annotators, plus
per-label breakdown. Pass 2+ LS export JSONs (one per annotator).

What "agreement" means here
---------------------------
For each shared letter, treat annotator A's spans as gold and annotator B's
as predicted (F1 is symmetric, so it doesn't matter which is which):

  strict :  span (start, end, label) must match exactly
  lenient:  any character overlap + same label counts as a hit
            (boundary disagreements are forgiven)

Textarea pre-fills are ignored; duplicate spans (LS sometimes keeps a
'prediction' + 'prediction-changed' copy of the same untouched pre-annotation)
are collapsed via set semantics.

Usage
-----
    python NER_pipeline/calculate_ner_iaa.py \\
        NER_pipeline/IAA_annots/project-72-...json \\
        NER_pipeline/IAA_annots/project-76-...json

For N >= 3 files, every pair is reported plus a macro-average row.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

Span = tuple[int, int, str]  # (start, end, label)


def load_annotator(path: Path) -> tuple[str, dict[str, set[Span]]]:
    """Return (annotator_label, {letter_id: {spans}})."""
    tasks = json.loads(path.read_text(encoding="utf-8"))
    by_letter: dict[str, set[Span]] = {}
    user_ids: Counter = Counter()

    for t in tasks:
        letter_id = (t.get("data") or {}).get("letter_id") or ""
        if not letter_id:
            continue
        annots = t.get("annotations") or []
        if not annots:
            continue
        ann = annots[0]                       # one annotation per task in IAA exports
        user_ids[ann.get("completed_by")] += 1

        spans: set[Span] = set()
        for r in ann.get("result") or []:
            if r.get("type") != "labels":     # skip textareas, relations, ...
                continue
            v = r.get("value") or {}
            labels = v.get("labels") or []
            if not labels:
                continue
            try:
                s, e = int(v["start"]), int(v["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e <= s:
                continue
            spans.add((s, e, labels[0]))      # set => dedup
        by_letter[letter_id] = spans

    user_id = user_ids.most_common(1)[0][0] if user_ids else "?"
    parts = path.stem.split("-")
    proj  = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else path.stem
    return f"{proj}_u{user_id}", by_letter


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def strict_match(a: set[Span], b: set[Span]):
    tp_set = a & b
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for s in tp_set:    per[s[2]][0] += 1
    for s in b - a:     per[s[2]][1] += 1
    for s in a - b:     per[s[2]][2] += 1
    return (len(tp_set), len(b - a), len(a - b),
            {k: tuple(v) for k, v in per.items()})


def lenient_match(a: set[Span], b: set[Span]):
    """Greedy 1:1 matching: same label + any character overlap = hit."""
    a_list = sorted(a)
    b_list = sorted(b)
    used: set[int] = set()
    tp_pairs: list[tuple[Span, Span]] = []
    for sa in a_list:
        for j, sb in enumerate(b_list):
            if j in used or sa[2] != sb[2]:
                continue
            if sa[1] <= sb[0] or sb[1] <= sa[0]:    # no overlap
                continue
            used.add(j)
            tp_pairs.append((sa, sb))
            break
    matched_a = {pa for pa, _ in tp_pairs}
    matched_b = {pb for _, pb in tp_pairs}
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, sb in tp_pairs:      per[sb[2]][0] += 1
    for sb in b - matched_b:    per[sb[2]][1] += 1
    for sa in a - matched_a:    per[sa[2]][2] += 1
    tp = len(tp_pairs)
    return tp, len(b) - tp, len(a) - tp, {k: tuple(v) for k, v in per.items()}


def span_iou(sa: Span, sb: Span) -> float:
    """Character-level IoU between two spans (labels ignored)."""
    s1, e1, _ = sa
    s2, e2, _ = sb
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union else 0.0


def iou_match(a: set[Span], b: set[Span], threshold: float = 0.5):
    """Greedy 1:1 matching: same label + IoU >= threshold = hit.

    This is the boundary-relaxed match used for the "lenient" column in
    NER_pipeline/compute_ner_metrics.py.  Historically the codebase used
    any-character-overlap; from 2026-07-23 we standardise on IoU >= 0.5
    to align with the IAA convention and modern NER practice.

    For each gold span we take the best-IoU predicted span with the same
    label that clears the threshold; predictions are consumed greedily.
    """
    a_list = sorted(a)
    b_list = sorted(b)
    used: set[int] = set()
    tp_pairs: list[tuple[Span, Span]] = []
    for sa in a_list:
        best_j, best_iou = -1, threshold
        for j, sb in enumerate(b_list):
            if j in used or sa[2] != sb[2]:
                continue
            iou = span_iou(sa, sb)
            if iou >= best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            tp_pairs.append((sa, b_list[best_j]))
    matched_a = {pa for pa, _ in tp_pairs}
    matched_b = {pb for _, pb in tp_pairs}
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, sb in tp_pairs:      per[sb[2]][0] += 1
    for sb in b - matched_b:    per[sb[2]][1] += 1
    for sa in a - matched_a:    per[sa[2]][2] += 1
    tp = len(tp_pairs)
    return tp, len(b) - tp, len(a) - tp, {k: tuple(v) for k, v in per.items()}


def pairwise(name_a, name_b, da, db) -> dict:
    common = sorted(set(da) & set(db))
    tot_s = [0, 0, 0]
    tot_l = [0, 0, 0]
    per_s: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_l: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_task = []
    for lid in common:
        sa, sb = da[lid], db[lid]
        s_tp, s_fp, s_fn, s_per = strict_match(sa, sb)
        l_tp, l_fp, l_fn, l_per = lenient_match(sa, sb)
        tot_s[0] += s_tp; tot_s[1] += s_fp; tot_s[2] += s_fn
        tot_l[0] += l_tp; tot_l[1] += l_fp; tot_l[2] += l_fn
        for k, (tp, fp, fn) in s_per.items():
            per_s[k][0] += tp; per_s[k][1] += fp; per_s[k][2] += fn
        for k, (tp, fp, fn) in l_per.items():
            per_l[k][0] += tp; per_l[k][1] += fp; per_l[k][2] += fn
        per_task.append({
            "letter_id":  lid,
            "spans_A":    len(sa),
            "spans_B":    len(sb),
            "strict_F1":  round(_f1(s_tp, s_fp, s_fn)[2], 3),
            "lenient_F1": round(_f1(l_tp, l_fp, l_fn)[2], 3),
        })
    sP, sR, sF = _f1(*tot_s)
    lP, lR, lF = _f1(*tot_l)
    return {
        "annotators": [name_a, name_b],
        "n_common":   len(common),
        "n_only_A":   len(set(da) - set(db)),
        "n_only_B":   len(set(db) - set(da)),
        "strict":  {"P": sP, "R": sR, "F1": sF,
                    "TP": tot_s[0], "FP": tot_s[1], "FN": tot_s[2]},
        "lenient": {"P": lP, "R": lR, "F1": lF,
                    "TP": tot_l[0], "FP": tot_l[1], "FN": tot_l[2]},
        "per_label_strict":  {k: tuple(v) for k, v in per_s.items()},
        "per_label_lenient": {k: tuple(v) for k, v in per_l.items()},
        "per_task": per_task,
    }


def print_pair(rep: dict) -> None:
    a, b = rep["annotators"]
    print(f"\n{'=' * 72}\n  {a}  vs.  {b}\n{'=' * 72}")
    print(f"  Letters annotated by both : {rep['n_common']}")
    if rep["n_only_A"] or rep["n_only_B"]:
        print(f"  Only in {a}: {rep['n_only_A']}    only in {b}: {rep['n_only_B']}")
    s, l = rep["strict"], rep["lenient"]
    print(f"\n  STRICT  (exact span+label) : P={s['P']:.3f}  R={s['R']:.3f}  "
          f"F1={s['F1']:.3f}   (TP={s['TP']}, FP={s['FP']}, FN={s['FN']})")
    print(f"  LENIENT (overlap + label)  : P={l['P']:.3f}  R={l['R']:.3f}  "
          f"F1={l['F1']:.3f}   (TP={l['TP']}, FP={l['FP']}, FN={l['FN']})")
    for title, key in [("Per-label STRICT",  "per_label_strict"),
                       ("Per-label LENIENT", "per_label_lenient")]:
        print(f"\n  {title}:")
        for lbl in sorted(rep[key]):
            tp, fp, fn = rep[key][lbl]
            P, R, F = _f1(tp, fp, fn)
            print(f"    {lbl:<10} TP={tp:4d}  FP={fp:4d}  FN={fn:4d}"
                  f"  P={P:.3f}  R={R:.3f}  F1={F:.3f}")
    print("\n  Worst 5 letters by strict F1:")
    for t in sorted(rep["per_task"], key=lambda x: x["strict_F1"])[:5]:
        print(f"    {t['letter_id']:<22} spans {t['spans_A']:2}/{t['spans_B']:2}  "
              f"strict={t['strict_F1']:.2f}  lenient={t['lenient_F1']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path,
                    help="Two or more Label Studio export JSONs.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional: dump the full report (incl. per-task) as JSON.")
    args = ap.parse_args()
    if len(args.files) < 2:
        sys.exit("Need at least 2 annotator files.")

    annotators = []
    for p in args.files:
        if not p.exists():
            sys.exit(f"not found: {p}")
        name, data = load_annotator(p)
        n_spans = sum(len(s) for s in data.values())
        print(f"loaded {name:<22} {len(data):3} letters, {n_spans:5} spans   "
              f"({p.name})")
        annotators.append((name, data))

    reports = []
    for (na, da), (nb, db) in combinations(annotators, 2):
        rep = pairwise(na, nb, da, db)
        print_pair(rep)
        reports.append(rep)

    if len(annotators) > 2:
        avg_s = sum(r["strict"]["F1"]  for r in reports) / len(reports)
        avg_l = sum(r["lenient"]["F1"] for r in reports) / len(reports)
        print(f"\n{'=' * 72}\n  Macro-average over {len(reports)} pairs:"
              f"\n     strict  F1 = {avg_s:.3f}"
              f"\n     lenient F1 = {avg_l:.3f}\n{'=' * 72}")

    if args.out:
        args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\nWrote full report -> {args.out}")


if __name__ == "__main__":
    main()
