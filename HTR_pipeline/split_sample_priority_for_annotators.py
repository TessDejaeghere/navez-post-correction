#!/usr/bin/env python3
"""Split sample-priority HTR-correction tasks across N annotators.

Pools tasks from all matching input files, sorts by ms_key for a deterministic
order, and round-robin assigns them so each annotator gets:
  - an evenly-sized share (within +/-1),
  - a balanced mix of tomes/dates (since neighbouring ms_keys go to different
    annotators).

By default it reads every
`data/labelstudio_upload/sample_priority_HTR_to_correct/labelstudio_sample_priority*.json`
EXCEPT files matching `*_annotator_*.json` (its own previous outputs), and writes:

  labelstudio_sample_priority_annotator_1.json
  labelstudio_sample_priority_annotator_2.json
  labelstudio_sample_priority_annotator_3.json

into the same folder. Originals are left untouched.

Usage:
    python HTR_pipeline/split_sample_priority_for_annotators.py
    python HTR_pipeline/split_sample_priority_for_annotators.py --annotators 4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "data" / "labelstudio_upload" / "sample_priority_HTR_to_correct"
DEFAULT_GLOB = "labelstudio_sample_priority*.json"
OUT_STEM = "labelstudio_sample_priority_annotator_"


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("tasks", [])


def ms_key_of(task: dict) -> str:
    d = task.get("data", {}) if isinstance(task, dict) else {}
    for field in ("ms_key", "Ms_key", "letter_id", "name", "key", "id"):
        v = d.get(field)
        if v:
            return str(v)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="Folder containing the input LS JSON files and where outputs go.")
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help="Glob for input files inside --dir.")
    ap.add_argument("--annotators", type=int, default=3,
                    help="How many annotators to split across (default: 3).")
    ap.add_argument("--out-stem", default=OUT_STEM,
                    help="Output filename stem. Files will be <stem><i>.json (1-indexed).")
    args = ap.parse_args()

    args.dir = args.dir.resolve()
    if args.annotators < 1:
        raise SystemExit("--annotators must be >= 1")

    inputs = sorted(p for p in args.dir.glob(args.glob)
                    if "_annotator_" not in p.name)
    if not inputs:
        raise SystemExit(f"No input files matching '{args.glob}' in {args.dir}")

    pooled: list[tuple[str, str, dict]] = []   # (ms_key, source_filename, task)
    per_source: Counter[str] = Counter()
    seen_keys: set[str] = set()
    for p in inputs:
        tasks = load_tasks(p)
        per_source[p.name] = len(tasks)
        for t in tasks:
            k = ms_key_of(t)
            # Dedupe across files just in case (shouldn't happen but cheap insurance).
            if k and k in seen_keys:
                continue
            if k:
                seen_keys.add(k)
            pooled.append((k, p.name, t))

    print(f"Sources ({len(inputs)} file(s)):")
    for name, n in per_source.most_common():
        print(f"  {name:60s}  {n:>4} tasks")
    print(f"Pooled (deduped) : {len(pooled)} tasks")

    # Deterministic order: sort by ms_key (empty keys sink to the end but keep
    # original relative order via stable sort).
    pooled.sort(key=lambda x: (x[0] == "", x[0]))

    # Round-robin into N buckets.
    buckets: list[list[dict]] = [[] for _ in range(args.annotators)]
    source_per_bucket: list[Counter[str]] = [Counter() for _ in range(args.annotators)]
    for i, (_k, src, task) in enumerate(pooled):
        b = i % args.annotators
        buckets[b].append(task)
        source_per_bucket[b][src] += 1

    print(f"\nSplit across {args.annotators} annotator(s):")
    for i, (bucket, src_counts) in enumerate(zip(buckets, source_per_bucket), start=1):
        out_path = args.dir / f"{args.out_stem}{i}.json"
        out_path.write_text(json.dumps(bucket, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        src_summary = ", ".join(f"{n} from {name}" for name, n in src_counts.most_common())
        print(f"  annotator {i}: {len(bucket):>3} tasks  ->  {out_path.relative_to(REPO)}")
        print(f"               ({src_summary})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
