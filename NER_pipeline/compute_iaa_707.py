"""Intermediate IAA + annotation stats for NER_707_*_final exports.

Steps:
  1. For each of the 4 annotator export files, filter to tasks whose
     `data.assignment == "IAA"` (the shared 10 calibration letters).
  2. Write the 4 filtered files to NER_pipeline/IAA_annots/NER_707_<name>_iaa10.json.
  3. Call NER_pipeline/calculate_ner_iaa.py on those 4 files.
  4. Compute annotation stats on the FULL exports:
       - # tasks total, # annotated, # cancelled
       - spans per label per annotator
       - mean/median text length (chars) of annotated letters
       - union of unique letter_ids per annotator
  5. Print & write the stats to NER_pipeline/IAA_annots/annotation_stats_707.md
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LS_OUT = REPO / "data" / "labelstudio_output"
IAA_DIR = REPO / "NER_pipeline" / "IAA_annots"
IAA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "A1": LS_OUT / "NER_707_A1_final.json",
    "A2":    LS_OUT / "NER_707_A2_final.json",
    "A3":  LS_OUT / "NER_707_A3_final.json",
    "A4":  LS_OUT / "NER_707_A4_final.json",
}


def load_tasks(p: Path) -> list[dict]:
    return json.loads(p.read_text(encoding="utf-8"))


def is_annotated(t: dict) -> bool:
    return any(not a.get("was_cancelled") for a in (t.get("annotations") or []))


def spans_of(t: dict) -> list[tuple[int, int, str]]:
    for a in (t.get("annotations") or []):
        if a.get("was_cancelled"):
            continue
        out = []
        for r in a.get("result") or []:
            if r.get("type") != "labels":
                continue
            v = r.get("value") or {}
            labels = v.get("labels") or []
            if not labels:
                continue
            try:
                s, e = int(v["start"]), int(v["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e > s:
                out.append((s, e, labels[0]))
        return out
    return []


# --- 1. filter to IAA tasks -----------------------------------------------
iaa_files: dict[str, Path] = {}
iaa_letter_ids_per: dict[str, set[str]] = {}

for name, src in FILES.items():
    tasks = load_tasks(src)
    iaa_tasks = [t for t in tasks
                 if (t.get("data") or {}).get("assignment") == "IAA"
                 and is_annotated(t)]
    out = IAA_DIR / f"NER_707_{name}_iaa10.json"
    out.write_text(json.dumps(iaa_tasks, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    iaa_files[name] = out
    iaa_letter_ids_per[name] = {
        (t.get("data") or {}).get("letter_id") for t in iaa_tasks
    }
    print(f"  wrote {out.relative_to(REPO)}  ({len(iaa_tasks)} IAA tasks)")

common_iaa = set.intersection(*iaa_letter_ids_per.values())
print(f"\nCommon IAA letters across all 4 annotators: {len(common_iaa)}")
for lid in sorted(common_iaa):
    print(f"  - {lid}")

# --- 2. run existing IAA script -------------------------------------------
print("\n" + "=" * 72)
print("Running calculate_ner_iaa.py on IAA-10 slices ...")
print("=" * 72)
iaa_report = IAA_DIR / "iaa_report_707.json"
cmd = [
    sys.executable,
    str(REPO / "NER_pipeline" / "calculate_ner_iaa.py"),
    *(str(p) for p in iaa_files.values()),
    "--out", str(iaa_report),
]
subprocess.run(cmd, check=True)
print(f"\nIAA report written to: {iaa_report.relative_to(REPO)}")

# --- 3. annotation stats on FULL exports ----------------------------------
print("\n" + "=" * 72)
print("Annotation stats (full exports)")
print("=" * 72)

stats_rows = []
label_counts_per: dict[str, Counter] = {}
per_letter_span_counts: dict[str, list[int]] = {}

for name, src in FILES.items():
    tasks = load_tasks(src)
    annotated = [t for t in tasks if is_annotated(t)]
    cancelled = sum(1 for t in tasks
                    if not is_annotated(t) and (t.get("annotations") or []))

    lengths = [len((t.get("data") or {}).get("text") or "") for t in annotated]
    label_counter: Counter = Counter()
    span_counts_this = []
    for t in annotated:
        sps = spans_of(t)
        span_counts_this.append(len(sps))
        for _, _, lbl in sps:
            label_counter[lbl] += 1
    label_counts_per[name] = label_counter
    per_letter_span_counts[name] = span_counts_this

    stats_rows.append({
        "annotator":    name,
        "n_total":      len(tasks),
        "n_annotated":  len(annotated),
        "n_cancelled":  cancelled,
        "n_spans":      sum(span_counts_this),
        "mean_len":     round(statistics.mean(lengths), 1) if lengths else 0,
        "median_len":   int(statistics.median(lengths)) if lengths else 0,
        "min_len":      min(lengths) if lengths else 0,
        "max_len":      max(lengths) if lengths else 0,
        "mean_spans":   round(statistics.mean(span_counts_this), 2) if span_counts_this else 0,
    })

# All labels present across the four
all_labels = sorted({lbl for c in label_counts_per.values() for lbl in c})

# --- 4. write Markdown report --------------------------------------------
md_path = IAA_DIR / "annotation_stats_707.md"
lines: list[str] = []
lines.append("# NER_707 — annotation stats & intermediate IAA")
lines.append("")
lines.append(f"Source: `data/labelstudio_output/NER_707_*_final.json`  "
             f"(A1, A2, A3, A4)")
lines.append("")

# 4a. per-annotator overview
lines.append("## Per-annotator overview (full export, not only IAA)")
lines.append("")
header = ["annotator", "n_total", "n_annotated", "n_cancelled",
          "n_spans", "mean_spans/text",
          "mean_len (chars)", "median_len", "min_len", "max_len"]
lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "|".join(["---"] * len(header)) + "|")
for r in stats_rows:
    lines.append("| " + " | ".join(str(x) for x in [
        r["annotator"], r["n_total"], r["n_annotated"], r["n_cancelled"],
        r["n_spans"], r["mean_spans"],
        r["mean_len"], r["median_len"], r["min_len"], r["max_len"],
    ]) + " |")
lines.append("")

# totals
total_tasks = sum(r["n_total"] for r in stats_rows)
total_annot = sum(r["n_annotated"] for r in stats_rows)
total_spans = sum(r["n_spans"] for r in stats_rows)
lines.append(f"**Totals across all four exports:** {total_tasks} tasks, "
             f"{total_annot} annotated, {total_spans} spans.")
lines.append("")

# 4b. per-label counts
lines.append("## Spans per label per annotator")
lines.append("")
lines.append("| label | " + " | ".join(FILES.keys()) + " | TOTAL |")
lines.append("|" + "|".join(["---"] * (len(FILES) + 2)) + "|")
for lbl in all_labels:
    row_counts = [label_counts_per[name].get(lbl, 0) for name in FILES]
    total = sum(row_counts)
    lines.append(f"| {lbl} | " + " | ".join(str(c) for c in row_counts)
                 + f" | {total} |")
totals_row = [sum(label_counts_per[name].values()) for name in FILES]
lines.append(f"| **TOTAL** | " + " | ".join(f"**{c}**" for c in totals_row)
             + f" | **{sum(totals_row)}** |")
lines.append("")

# 4c. IAA summary (loaded from iaa_report_707.json if produced)
lines.append("## Intermediate IAA (10 shared IAA letters)")
lines.append("")
try:
    rep = json.loads(iaa_report.read_text(encoding="utf-8"))
    # rep is a list of pairwise dicts under "pairs" or a list itself?
    if isinstance(rep, dict) and "pairs" in rep:
        pairs = rep["pairs"]
    else:
        pairs = rep
    if isinstance(pairs, list):
        lines.append("| pair | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in pairs:
            a, b = p["annotators"]
            s, l = p["strict"], p["lenient"]
            lines.append(f"| {a} vs {b} | {s['P']:.3f} | {s['R']:.3f} | {s['F1']:.3f} "
                         f"| {l['P']:.3f} | {l['R']:.3f} | {l['F1']:.3f} |")
        if len(pairs) > 1:
            avg_s = sum(p["strict"]["F1"] for p in pairs) / len(pairs)
            avg_l = sum(p["lenient"]["F1"] for p in pairs) / len(pairs)
            lines.append("")
            lines.append(f"**Macro-avg over {len(pairs)} pairs:** strict F1 = {avg_s:.3f}, "
                         f"lenient F1 = {avg_l:.3f}")
except Exception as e:
    lines.append(f"_Could not parse `iaa_report_707.json` for pairwise summary: {e}_")
    lines.append(f"_See `{iaa_report.relative_to(REPO)}` for full pairwise details._")
lines.append("")

md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nMarkdown stats written to: {md_path.relative_to(REPO)}")
