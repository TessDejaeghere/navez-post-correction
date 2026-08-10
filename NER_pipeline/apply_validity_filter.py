"""Strip malformed predicted spans from per-model NER prediction JSONLs.

This is a readability test, not a plausibility test.  It removes spans that do
not denote a readable substring of the letter; a well-formed span that names an
entity which is not there survives untouched and is scored as a spurious
mention.  Three checks (imported from NER_pipeline/error_analysis.py so the
filter and the error analysis can never drift apart):
  * invalid_offsets  : start<0, end<=start, or end>len(text)
  * empty_content    : span has no alphabetic characters
  * tiny_midword     : span length <= 3 chars AND both edges strictly inside a word

Reads
-----
* ``NER_pipeline/eval/gold_561_merged.jsonl``   (for the letter text)
* ``NER_pipeline/eval/predictions/*.jsonl``

Writes
------
* ``NER_pipeline/eval/predictions_filtered/*.jsonl``
* Console summary of drops per model / per check

Usage
-----
    python NER_pipeline/apply_validity_filter.py

Downstream
----------
    python NER_pipeline/compute_ner_metrics.py \
        --pred-dir NER_pipeline/eval/predictions_filtered \
        --variance-fold 0
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from error_analysis import check_malformed  # noqa: E402

EVAL_DIR = REPO / "NER_pipeline" / "eval"
GOLD_PATH = EVAL_DIR / "gold_561_merged.jsonl"
PRED_DIR = EVAL_DIR / "predictions"
OUT_DIR = EVAL_DIR / "predictions_filtered"


def load_gold_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["letter_id"]] = r["text"]
    return out


def filter_file(src: Path, dst: Path, texts: dict[str, str]) -> dict[str, int]:
    """Filter one JSONL and return a count-by-reason dict."""
    counts: dict[str, int] = defaultdict(int)
    n_lines = 0
    n_in = 0
    n_out = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            lid = row["letter_id"]
            text = texts.get(lid, "")
            kept: list[list] = []
            for span in row.get("pred_spans", []):
                if len(span) != 3:
                    counts["malformed_row"] += 1
                    continue
                s, e, lbl = int(span[0]), int(span[1]), str(span[2])
                n_in += 1
                tag = check_malformed(text, s, e)
                if tag is None:
                    kept.append([s, e, lbl])
                    n_out += 1
                else:
                    counts[tag] += 1
            row["pred_spans"] = kept
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["_lines"] = n_lines
    counts["_spans_in"] = n_in
    counts["_spans_out"] = n_out
    return counts


def main() -> None:
    if not PRED_DIR.exists():
        raise SystemExit(f"No predictions dir at {PRED_DIR}")
    if not GOLD_PATH.exists():
        raise SystemExit(f"No gold file at {GOLD_PATH}")

    texts = load_gold_texts()
    print(f"Loaded {len(texts)} gold texts")

    per_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for src in sorted(PRED_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name
        counts = filter_file(src, dst, texts)
        # aggregate per model (filename prefix before _fold<k>_run<r>.jsonl)
        stem = src.stem
        # tolerate model names with underscores
        if "_fold" not in stem:
            continue
        model = stem.split("_fold")[0]
        for k, v in counts.items():
            per_model[model][k] += v

    # summary
    print(f"\n{'Model':<30} {'preds':>7} {'dropped':>8} {'invalid':>8} "
          f"{'empty':>7} {'midword':>8} {'kept':>7}")
    print("-" * 82)
    for m in sorted(per_model):
        c = per_model[m]
        n_in = c.get("_spans_in", 0)
        n_out = c.get("_spans_out", 0)
        dropped = n_in - n_out
        pct = f"({dropped / n_in:.1%})" if n_in else ""
        print(f"{m:<30} {n_in:>7} {dropped:>4} {pct:>5} "
              f"{c.get('invalid_offsets', 0):>8} "
              f"{c.get('empty_content', 0):>7} "
              f"{c.get('tiny_midword', 0):>8} "
              f"{n_out:>7}")

    print(f"\nFiltered predictions written to {OUT_DIR.relative_to(REPO)}")
    print("Next:  python NER_pipeline/compute_ner_metrics.py "
          f"--pred-dir {OUT_DIR.relative_to(REPO)} --variance-fold 0")


if __name__ == "__main__":
    main()
