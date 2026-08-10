"""GLiNER2 zero-shot NER baseline evaluation on the 561-letter gold pool.

Pipeline:
  1. Load the four NER_707_*_final.json exports and pick one annotator per
     letter (tie-break: A2 -> A1 -> A3 -> A4, matching the
     dedup policy used in analyze_ner_iaa_707.ipynb §8).
  2. Save the gold pool as one JSONL row per letter for reproducibility.
  3. Run GLiNER2 (fastino/gliner2-multi-v1 by default) on each letter's text
     using the schema from ``entity_schema.ENTITY_SCHEMA``.
  4. Save predictions as JSONL.
  5. Compute strict and lenient F1 per label + micro overall, using the same
     matching semantics as ``calculate_ner_iaa.py`` so the numbers are
     directly comparable to the IAA table in the paper.

Outputs (under ``NER_pipeline/eval/``):
  - gold_561.jsonl              one gold letter per line
  - gliner_preds_561.jsonl      one prediction set per line
  - gliner_baseline_metrics.csv per-label + overall F1
  - gliner_baseline_summary.md  human-readable table (also printed)

Usage (from repo root):
    python NER_pipeline/gliner_baseline_eval.py
    python NER_pipeline/gliner_baseline_eval.py --model fastino/gliner2-multi-v1
    python NER_pipeline/gliner_baseline_eval.py --limit 10   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from calculate_ner_iaa import lenient_match, strict_match, _f1  # noqa: E402
from entity_schema import ENTITY_SCHEMA, build_schema, label_set  # noqa: E402
from gliner_predict import extract_to_ls_results                # noqa: E402

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="fastino/gliner2-multi-v1",
                    help="HuggingFace model id passed to GLiNER2.from_pretrained().")
    ap.add_argument("--chunk-chars", type=int, default=1500,
                    help="Max chars per inference chunk (default 1500).")
    ap.add_argument("--device", default=None,
                    help="'cpu', 'cuda', 'cuda:0' ... default: GLiNER2 decides.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap on number of letters (0 = all 561).")
    ap.add_argument("--out-dir", default=str(REPO / "NER_pipeline" / "eval"),
                    help="Output directory (default NER_pipeline/eval/).")
    return ap.parse_args()


# --------------------------------------------------------------------------
# Gold pool assembly
# --------------------------------------------------------------------------
NER_EXPORTS = {
    "A2":    REPO / "data" / "labelstudio_output" / "NER_707_A2_final.json",
    "A1": REPO / "data" / "labelstudio_output" / "NER_707_A1_final.json",
    "A3":  REPO / "data" / "labelstudio_output" / "NER_707_A3_final.json",
    "A4":  REPO / "data" / "labelstudio_output" / "NER_707_A4_final.json",
}
TIE_BREAK = ["A2", "A1", "A3", "A4"]  # matches §8 of the IAA notebook


def _is_annotated(task: dict) -> bool:
    return any(not a.get("was_cancelled") for a in (task.get("annotations") or []))


def _spans(task: dict) -> list[tuple[int, int, str]]:
    for a in (task.get("annotations") or []):
        if a.get("was_cancelled"):
            continue
        out = []
        for r in a.get("result") or []:
            if r.get("type") != "labels":
                continue
            v = r.get("value") or {}
            labs = v.get("labels") or []
            if not labs:
                continue
            try:
                s, e = int(v["start"]), int(v["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e > s:
                out.append((s, e, labs[0]))
        return out
    return []


def build_gold() -> list[dict]:
    """Return one dict per unique letter with picked annotator + text + spans."""
    picked: dict[str, dict] = {}
    for name in TIE_BREAK:
        for t in json.loads(NER_EXPORTS[name].read_text(encoding="utf-8")):
            if not _is_annotated(t):
                continue
            data = t.get("data") or {}
            lid = data.get("letter_id")
            if not lid or lid in picked:
                continue
            picked[lid] = {
                "letter_id":     lid,
                "annotator":     name,
                "source_sender": data.get("source", ""),
                "text":          data.get("text", "") or "",
                "gold_spans":    _spans(t),
            }
    return sorted(picked.values(), key=lambda r: r["letter_id"])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(gold: list[dict], preds: list[dict]) -> dict:
    """Aggregate strict + lenient F1 across all letters, plus per-label."""
    by_lid_gold = {r["letter_id"]: set(tuple(s) for s in r["gold_spans"]) for r in gold}
    by_lid_pred = {r["letter_id"]: set(tuple(s) for s in r["pred_spans"]) for r in preds}

    tot_s = [0, 0, 0]
    tot_l = [0, 0, 0]
    per_s: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_l: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for lid, gs in by_lid_gold.items():
        ps = by_lid_pred.get(lid, set())
        s_tp, s_fp, s_fn, s_per = strict_match(gs, ps)
        l_tp, l_fp, l_fn, l_per = lenient_match(gs, ps)
        tot_s[0] += s_tp; tot_s[1] += s_fp; tot_s[2] += s_fn
        tot_l[0] += l_tp; tot_l[1] += l_fp; tot_l[2] += l_fn
        for k, (tp, fp, fn) in s_per.items():
            per_s[k][0] += tp; per_s[k][1] += fp; per_s[k][2] += fn
        for k, (tp, fp, fn) in l_per.items():
            per_l[k][0] += tp; per_l[k][1] += fp; per_l[k][2] += fn

    return {
        "n_letters": len(by_lid_gold),
        "strict":  {"P": _f1(*tot_s)[0], "R": _f1(*tot_s)[1], "F1": _f1(*tot_s)[2],
                    "TP": tot_s[0], "FP": tot_s[1], "FN": tot_s[2]},
        "lenient": {"P": _f1(*tot_l)[0], "R": _f1(*tot_l)[1], "F1": _f1(*tot_l)[2],
                    "TP": tot_l[0], "FP": tot_l[1], "FN": tot_l[2]},
        "per_label_strict":  {k: tuple(v) for k, v in per_s.items()},
        "per_label_lenient": {k: tuple(v) for k, v in per_l.items()},
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _report_line(label: str, arr: Iterable[int]) -> str:
    tp, fp, fn = arr
    P, R, F = _f1(tp, fp, fn)
    return (f"| {label:<8} | {tp:>5} | {fp:>5} | {fn:>5} | "
            f"{P:.3f} | {R:.3f} | **{F:.3f}** |")


def write_summary(report: dict, model: str, elapsed_s: float, out_md: Path) -> str:
    labels = sorted(set(report["per_label_strict"]) | set(report["per_label_lenient"]))

    lines: list[str] = []
    lines.append(f"# GLiNER2 zero-shot NER baseline")
    lines.append("")
    lines.append(f"- Model : `{model}`")
    lines.append(f"- Letters evaluated : **{report['n_letters']}**")
    lines.append(f"- Total inference time : {elapsed_s:.1f} s "
                 f"({elapsed_s / max(report['n_letters'], 1):.2f} s / letter)")
    lines.append(f"- Reference : deduplicated gold pool "
                 f"(A2 -> A1 -> A3 -> A4 tie-break)")
    lines.append("")

    for mode in ("strict", "lenient"):
        lines.append(f"## {mode.upper()} F1")
        lines.append("")
        s = report[mode]
        lines.append(f"**Micro**: P = {s['P']:.3f} · R = {s['R']:.3f} · "
                     f"**F1 = {s['F1']:.3f}**  "
                     f"(TP = {s['TP']}, FP = {s['FP']}, FN = {s['FN']})")
        lines.append("")
        lines.append("| label | TP | FP | FN | P | R | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for lbl in labels:
            arr = report[f"per_label_{mode}"].get(lbl, (0, 0, 0))
            lines.append(_report_line(lbl, arr))
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def write_metrics_csv(report: dict, out_csv: Path) -> None:
    import csv
    labels = sorted(set(report["per_label_strict"]) | set(report["per_label_lenient"]))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "label", "TP", "FP", "FN", "P", "R", "F1"])
        for mode in ("strict", "lenient"):
            for lbl in labels:
                tp, fp, fn = report[f"per_label_{mode}"].get(lbl, (0, 0, 0))
                P, R, F = _f1(tp, fp, fn)
                w.writerow([mode, lbl, tp, fp, fn, f"{P:.4f}", f"{R:.4f}", f"{F:.4f}"])
            s = report[mode]
            w.writerow([mode, "MICRO", s["TP"], s["FP"], s["FN"],
                        f"{s['P']:.4f}", f"{s['R']:.4f}", f"{s['F1']:.4f}"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Gold pool -----------------------------------------------------
    print("Building gold pool ...")
    gold = build_gold()
    if args.limit:
        gold = gold[: args.limit]
    print(f"  {len(gold)} unique letters")
    with (out_dir / "gold_561.jsonl").open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # --- 2. Load GLiNER2 --------------------------------------------------
    print(f"Loading GLiNER2 model: {args.model}")
    from gliner2 import GLiNER2  # noqa: E402
    extractor = GLiNER2.from_pretrained(args.model)
    if args.device:
        try:
            extractor.to(args.device)
        except Exception as exc:                              # noqa: BLE001
            print(f"  (warning) could not move model to {args.device}: {exc}")
    schema = build_schema(extractor)
    print(f"Schema labels: {', '.join(label_set())}")

    # --- 3. Predict -------------------------------------------------------
    preds: list[dict] = []
    t0 = time.time()
    for i, g in enumerate(gold, 1):
        results, _ = extract_to_ls_results(
            extractor, schema, g["text"], args.chunk_chars,
        )
        pred_spans = [
            (int(r["value"]["start"]), int(r["value"]["end"]), r["value"]["labels"][0])
            for r in results
        ]
        preds.append({
            "letter_id": g["letter_id"],
            "pred_spans": pred_spans,
        })
        if i % 25 == 0 or i == len(gold):
            print(f"  [{i:>4}/{len(gold)}]  {g['letter_id']}  "
                  f"gold={len(g['gold_spans']):>3}  pred={len(pred_spans):>3}")
    elapsed = time.time() - t0
    print(f"\nInference: {elapsed:.1f} s total ({elapsed/len(gold):.2f} s / letter)")

    with (out_dir / "gliner_preds_561.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- 4. Evaluate ------------------------------------------------------
    report = evaluate(gold, preds)
    md = write_summary(report, args.model, elapsed,
                       out_dir / "gliner_baseline_summary.md")
    write_metrics_csv(report, out_dir / "gliner_baseline_metrics.csv")
    print("\n" + md)
    print(f"\nWrote:")
    print(f"  {(out_dir / 'gold_561.jsonl').relative_to(REPO)}")
    print(f"  {(out_dir / 'gliner_preds_561.jsonl').relative_to(REPO)}")
    print(f"  {(out_dir / 'gliner_baseline_metrics.csv').relative_to(REPO)}")
    print(f"  {(out_dir / 'gliner_baseline_summary.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
