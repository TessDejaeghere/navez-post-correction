"""Aggregate NER predictions from every model / fold / run into one metrics
report.

Reads
-----
* ``NER_pipeline/eval/gold_561_merged.jsonl``       (from build_ner_folds.py)
* ``NER_pipeline/eval/folds_writer_year.json``      (from build_ner_folds.py)
* ``NER_pipeline/eval/predictions/*.jsonl``          (from the Colab notebooks)

Prediction file naming convention (enforced)
--------------------------------------------
::

    NER_pipeline/eval/predictions/<model>_fold<k>_run<r>.jsonl

Each JSONL row must contain::

    {"letter_id": "...", "pred_spans": [[start, end, "LABEL"], ...]}

Labels must already be in the merged schema (ART_NOM collapsed into ART).
If a model predicts un-merged labels the mapping is applied here defensively.

Match criteria
--------------
* **strict**  : exact character-offset match, same label.
* **lenient** : IoU >= 0.5, same label (see ``LENIENT_IOU_THRESHOLD``).
  This aligns with the IAA convention and modern NER-literature practice;
  the historical any-character-overlap rule remained available as
  ``calculate_ner_iaa.lenient_match`` for the annotator-agreement scripts
  that used it.

Writes
------
* ``NER_pipeline/eval/ner_final_metrics.csv``  (long-form: one row per
  model x fold x run x label x mode)
* ``NER_pipeline/eval/ner_final_summary.md``   (paper-ready tables)

The summary reports two variance decompositions:
  1. Fold variance   — 1 run per fold across all folds (all models)
  2. Run variance    — 3 runs on a single fold (LLMs only, fold specified
                        by ``--variance-fold``)

Usage
-----
    python NER_pipeline/compute_ner_metrics.py
    python NER_pipeline/compute_ner_metrics.py --variance-fold 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from calculate_ner_iaa import _f1, iou_match, strict_match  # noqa: E402

# Boundary-relaxed match threshold used for the "lenient" column.
# 0.5 aligns with the IAA convention and modern NER-literature practice.
LENIENT_IOU_THRESHOLD = 0.5

EVAL_DIR = REPO / "NER_pipeline" / "eval"
GOLD_JSONL = EVAL_DIR / "gold_561_merged.jsonl"
FOLDS_JSON = EVAL_DIR / "folds_writer_year.json"
PRED_DIR = EVAL_DIR / "predictions"

LABEL_MERGE_MAP = {"ART_NOM": "ART"}

# File-name pattern: <model>_fold<k>_run<r>.jsonl  (model may contain
# hyphens / dots but not underscores followed by "fold").
FNAME_RE = re.compile(r"^(?P<model>.+)_fold(?P<fold>\d+)_run(?P<run>\d+)\.jsonl$")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_gold() -> dict[str, set[tuple[int, int, str]]]:
    out: dict[str, set[tuple[int, int, str]]] = {}
    with GOLD_JSONL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["letter_id"]] = {tuple(s) for s in r["gold_spans"]}
    return out


def load_folds() -> tuple[dict[str, int], int]:
    payload = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    letter_fold = {r["letter_id"]: r["fold"] for r in payload["letters"]}
    return letter_fold, int(payload.get("n_folds", max(letter_fold.values()) + 1))


def load_predictions(path: Path) -> dict[str, set[tuple[int, int, str]]]:
    out: dict[str, set[tuple[int, int, str]]] = {}
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
                start, end, label = int(s[0]), int(s[1]), str(s[2])
                label = LABEL_MERGE_MAP.get(label, label)
                if end > start:
                    spans.add((start, end, label))
            out[r["letter_id"]] = spans
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_fold(
    gold: dict[str, set],
    preds: dict[str, set],
    letter_ids_in_fold: set[str],
) -> dict:
    """Compute strict + lenient F1 on the letters of one fold.

    "strict": exact character-offset match, same label.
    "lenient": IoU >= LENIENT_IOU_THRESHOLD (0.5), same label.
    """
    tot_s = [0, 0, 0]
    tot_l = [0, 0, 0]
    per_s: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_l: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for lid in letter_ids_in_fold:
        gs = gold.get(lid, set())
        ps = preds.get(lid, set())
        s_tp, s_fp, s_fn, s_per = strict_match(gs, ps)
        l_tp, l_fp, l_fn, l_per = iou_match(gs, ps, threshold=LENIENT_IOU_THRESHOLD)
        tot_s[0] += s_tp; tot_s[1] += s_fp; tot_s[2] += s_fn
        tot_l[0] += l_tp; tot_l[1] += l_fp; tot_l[2] += l_fn
        for k, (tp, fp, fn) in s_per.items():
            per_s[k][0] += tp; per_s[k][1] += fp; per_s[k][2] += fn
        for k, (tp, fp, fn) in l_per.items():
            per_l[k][0] += tp; per_l[k][1] += fp; per_l[k][2] += fn

    return {
        "strict_micro":  _f1(*tot_s),
        "lenient_micro": _f1(*tot_l),
        "strict_per_label":  {k: _f1(*v) for k, v in per_s.items()},
        "lenient_per_label": {k: _f1(*v) for k, v in per_l.items()},
        "strict_tp_fp_fn":  tuple(tot_s),
        "lenient_tp_fp_fn": tuple(tot_l),
    }


def normal_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Return (mean, half-width) using normal approximation.
    For k=3 the CI is wide; report as caveat in the paper."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, z * sqrt(var / n)


# --------------------------------------------------------------------------
# Long-form CSV
# --------------------------------------------------------------------------
def emit_csv(rows: list[dict], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "fold", "run", "mode", "label",
              "P", "R", "F1", "TP", "FP", "FN"]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {csv_out.relative_to(REPO)}  ({len(rows)} rows)")


def score_to_rows(model: str, fold: int, run: int, s: dict) -> list[dict]:
    rows: list[dict] = []
    for mode, key_micro, key_per, key_tfn in [
        ("strict",  "strict_micro",  "strict_per_label",  "strict_tp_fp_fn"),
        ("lenient", "lenient_micro", "lenient_per_label", "lenient_tp_fp_fn"),
    ]:
        P, R, F = s[key_micro]
        TP, FP, FN = s[key_tfn]
        rows.append({"model": model, "fold": fold, "run": run, "mode": mode,
                     "label": "MICRO", "P": f"{P:.4f}", "R": f"{R:.4f}",
                     "F1": f"{F:.4f}", "TP": TP, "FP": FP, "FN": FN})
        for lbl, (P, R, F) in sorted(s[key_per].items()):
            rows.append({"model": model, "fold": fold, "run": run, "mode": mode,
                         "label": lbl, "P": f"{P:.4f}", "R": f"{R:.4f}",
                         "F1": f"{F:.4f}"})
    return rows


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
def write_summary(
    per_model: dict[str, dict[int, dict[int, dict]]],
    variance_fold: int,
    n_folds: int,
    md_out: Path,
) -> None:
    """per_model[model][fold][run] = score dict"""
    lines: list[str] = []
    lines.append(f"# NER evaluation — {n_folds}-fold stratified CV (sender x decade)")
    lines.append("")
    lines.append("Gold: 561 unique letters, ART_NOM merged into ART.")
    lines.append("Metrics: micro-averaged across letters within a fold; "
                 "means / CIs across folds (or across runs for LLM variance).")
    lines.append("")
    lines.append("## Table 1 — Fold variance (1 run per fold, all models)")
    lines.append("")
    lines.append(f"Mean lenient F1 ± 95% normal-approximation CI across "
                 f"{n_folds} folds.")
    lines.append("")
    lines.append("| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) | "
                 "PER strict F1 | PER lenient F1 |")
    lines.append("|---|---|---|---|---|")

    for model in sorted(per_model):
        fold_scores = per_model[model]
        # For fold variance: take run 0 in each fold that has run 0.
        strict = [fold_scores[k][0]["strict_micro"][2]
                  for k in sorted(fold_scores) if 0 in fold_scores[k]]
        lenient = [fold_scores[k][0]["lenient_micro"][2]
                   for k in sorted(fold_scores) if 0 in fold_scores[k]]
        per_strict = [fold_scores[k][0]["strict_per_label"].get("PER", (0, 0, 0))[2]
                      for k in sorted(fold_scores) if 0 in fold_scores[k]]
        per_lenient = [fold_scores[k][0]["lenient_per_label"].get("PER", (0, 0, 0))[2]
                       for k in sorted(fold_scores) if 0 in fold_scores[k]]

        ms, cs = normal_ci(strict)
        ml, cl = normal_ci(lenient)
        mps, _ = normal_ci(per_strict)
        mpl, _ = normal_ci(per_lenient)
        lines.append(f"| {model} | {ms:.3f} ± {cs:.3f} | {ml:.3f} ± {cl:.3f} "
                     f"| {mps:.3f} | {mpl:.3f} |")

    lines.append("")
    lines.append(f"## Table 2 — LLM run variance (fold {variance_fold}, "
                 "3 runs per model)")
    lines.append("")
    lines.append("Mean lenient F1 ± 95% normal-approximation CI across 3 runs "
                 f"on fold {variance_fold} letters (LLMs only).")
    lines.append("")
    lines.append("| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) |")
    lines.append("|---|---|---|")

    for model in sorted(per_model):
        runs = per_model[model].get(variance_fold, {})
        if len(runs) < 2:
            continue  # not a variance-run model
        strict = [runs[r]["strict_micro"][2] for r in sorted(runs)]
        lenient = [runs[r]["lenient_micro"][2] for r in sorted(runs)]
        ms, cs = normal_ci(strict)
        ml, cl = normal_ci(lenient)
        n = len(strict)
        lines.append(f"| {model} ({n} runs) | {ms:.3f} ± {cs:.3f} "
                     f"| {ml:.3f} ± {cl:.3f} |")

    lines.append("")
    lines.append("## Per-label breakdown (fold 0, run 0)")
    lines.append("")
    lines.append("| Model | label | Strict F1 | Lenient F1 |")
    lines.append("|---|---|---|---|")
    for model in sorted(per_model):
        s = per_model[model].get(0, {}).get(0)
        if not s:
            continue
        labels = sorted(set(s["strict_per_label"]) | set(s["lenient_per_label"]))
        for lbl in labels:
            sf = s["strict_per_label"].get(lbl, (0, 0, 0))[2]
            lf = s["lenient_per_label"].get(lbl, (0, 0, 0))[2]
            lines.append(f"| {model} | {lbl} | {sf:.3f} | {lf:.3f} |")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_out.relative_to(REPO)}")
    print("\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variance-fold", type=int, default=0,
                    help="Which fold holds the 3-run variance experiment.")
    ap.add_argument("--pred-dir", type=str, default=str(PRED_DIR),
                    help="Directory of prediction JSONL files (default: predictions/).")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    # Derive output filenames from the pred-dir name.  Default -> ner_final_*;
    # anything else -> ner_final_<suffix>.* so raw metrics are preserved.
    suffix = ""
    if pred_dir.name != PRED_DIR.name:
        suffix = "_" + pred_dir.name.replace("predictions_", "")
    csv_out = EVAL_DIR / f"ner_final_metrics{suffix}.csv"
    md_out  = EVAL_DIR / f"ner_final_summary{suffix}.md"

    gold = load_gold()
    folds, n_folds = load_folds()
    letters_by_fold: dict[int, set[str]] = defaultdict(set)
    for lid, k in folds.items():
        letters_by_fold[k].add(lid)

    if not pred_dir.exists():
        sys.exit(f"No predictions dir: {pred_dir}. Run the Colab notebooks first.")

    per_model: dict[str, dict[int, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict))
    rows: list[dict] = []

    for path in sorted(pred_dir.glob("*.jsonl")):
        m = FNAME_RE.match(path.name)
        if not m:
            print(f"  (skip) {path.name} does not match <model>_fold<k>_run<r>.jsonl")
            continue
        model, fold, run = m["model"], int(m["fold"]), int(m["run"])

        preds = load_predictions(path)
        letters_in_fold = letters_by_fold[fold]
        score = score_fold(gold, preds, letters_in_fold)
        per_model[model][fold][run] = score
        rows.extend(score_to_rows(model, fold, run, score))
        print(f"  {model:<25} fold {fold} run {run}  "
              f"strict={score['strict_micro'][2]:.3f}  "
              f"lenient={score['lenient_micro'][2]:.3f}")

    emit_csv(rows, csv_out)
    write_summary(per_model, args.variance_fold, n_folds, md_out)


if __name__ == "__main__":
    main()
