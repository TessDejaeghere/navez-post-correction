"""
Run GLiNER2 zero-shot NER over the Label Studio import tasks in
``data/labelstudio_upload/`` and emit prediction-augmented copies into
``data/preannots_GLiNER/`` (per-letter + combined).

Reads the entity schema from ``entity_schema.ENTITY_SCHEMA`` so the labels
can be tweaked in one place.

Usage (from the repo root):

    python NER_pipeline/gliner_predict.py
    python NER_pipeline/gliner_predict.py --limit 1   # smoke test
    python NER_pipeline/gliner_predict.py --model fastino/gliner2-large

The output is a Label Studio import file: each task carries its original
``data`` block plus a ``predictions`` block ready for upload.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# GLiNER2 prints emoji during model load; on Windows the default cp1252
# console will crash on those. Reconfigure stdout/stderr to UTF-8 early.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Make the sibling `entity_schema` module importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entity_schema import ENTITY_SCHEMA, build_schema, label_set  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/labelstudio_upload",
        help="Folder of *_ls_import.json task files to preannotate.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/preannots_GLiNER",
        help="Folder to write per-letter prediction-augmented JSONs into.",
    )
    parser.add_argument(
        "--combined-name",
        default="gliner_preannots.json",
        help="Filename for the combined import array inside --output-dir.",
    )
    parser.add_argument(
        "--model",
        default="fastino/gliner2-multi-v1",
        help="HF model id passed to GLiNER2.from_pretrained(). Use "
             "'fastino/gliner2-multi-v1' (multilingual, recommended for French), "
             "'fastino/gliner2-large-v1' (English, 0.5B), or "
             "'fastino/gliner2-base-v1' (English, 0.2B, fastest).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1500,
        help="Approximate max chars per chunk fed to the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N files (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="'cpu', 'cuda', 'cuda:0' ... Default: let GLiNER2 decide.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split ``text`` on newlines into chunks no larger than ``max_chars``.

    Returns a list of ``(offset_in_original, chunk_text)`` tuples so spans
    extracted from each chunk can be mapped back to the source text.
    Single lines longer than ``max_chars`` are emitted as-is (GLiNER2 will
    truncate internally) rather than being split mid-word.
    """
    if len(text) <= max_chars:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    buf_len = 0
    cursor = 0

    for line in text.splitlines(keepends=True):
        line_len = len(line)
        if buf and buf_len + line_len > max_chars:
            chunks.append((buf_start, "".join(buf)))
            buf = []
            buf_len = 0
            buf_start = cursor
        if not buf:
            buf_start = cursor
        buf.append(line)
        buf_len += line_len
        cursor += line_len

    if buf:
        chunks.append((buf_start, "".join(buf)))
    return chunks


# ---------------------------------------------------------------------------
# Prediction conversion
# ---------------------------------------------------------------------------

def extract_to_ls_results(
    extractor,
    schema,
    full_text: str,
    chunk_chars: int,
) -> tuple[list[dict], list[float]]:
    """Run GLiNER2 over (possibly chunked) text and convert to Label Studio
    prediction results. Returns ``(results, confidences)``."""

    results: list[dict] = []
    confidences: list[float] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for offset, chunk in chunk_text(full_text, chunk_chars):
        out = extractor.extract(
            chunk,
            schema,
            include_confidence=True,
            include_spans=True,
        )
        entities = (out or {}).get("entities", {}) or {}

        for label, items in entities.items():
            if not items:
                continue
            for item in items:
                # GLiNER2 may return either a dict (when confidence/spans are
                # requested) or a bare string; handle both defensively.
                if not isinstance(item, dict):
                    continue
                start = item.get("start")
                end = item.get("end")
                surface = item.get("text", "")
                if start is None or end is None or end <= start:
                    continue
                abs_start = offset + int(start)
                abs_end = offset + int(end)
                # Sanity-check: the span text must match the original slice
                # exactly. If not, fall back to the chunk-local slice from
                # the model output (still mapped to absolute coords).
                expected = full_text[abs_start:abs_end]
                if expected != surface:
                    # Try to recover by trusting the offsets over the surface.
                    surface = expected or surface

                key = (abs_start, abs_end, label)
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                confidence = float(item.get("confidence", 0.0))
                confidences.append(confidence)
                results.append(
                    {
                        "id": uuid.uuid4().hex[:10],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": abs_start,
                            "end": abs_end,
                            "text": surface,
                            "labels": [label],
                        },
                        "score": round(confidence, 4),
                    }
                )

    # Sort by character offset so reviewing in Label Studio reads naturally.
    results.sort(key=lambda r: (r["value"]["start"], r["value"]["end"]))
    return results, confidences


def build_prediction_block(results: list[dict], confidences: list[float], model_id: str) -> dict:
    score = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return {
        "model_version": f"gliner2-zeroshot::{model_id}",
        "score": score,
        "result": results,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_task_file(path: Path) -> dict:
    """Each *_ls_import.json holds a single-element list with a task dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"{path.name}: expected 1 task, got {len(data)}")
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f"{path.name}: unsupported top-level JSON type {type(data).__name__}")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(p for p in input_dir.glob("*_ls_import.json") if p.is_file())
    if args.limit:
        task_files = task_files[: args.limit]
    if not task_files:
        print(f"No *_ls_import.json files found under {input_dir.resolve()}")
        return

    print(f"Loading GLiNER2 model: {args.model}")
    from gliner2 import GLiNER2  # import here so --help works without the dep

    extractor = GLiNER2.from_pretrained(args.model)
    if args.device:
        try:
            extractor.to(args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  (warning) could not move model to {args.device}: {exc}")

    schema = build_schema(extractor)
    print(f"Schema labels: {', '.join(label_set())}")

    combined: list[dict] = []
    total_entities = 0

    for idx, path in enumerate(task_files, start=1):
        task = load_task_file(path)
        text = (task.get("data") or {}).get("text", "")
        if not text:
            print(f"  [{idx}/{len(task_files)}] {path.name}: empty text — skipping")
            continue

        results, confidences = extract_to_ls_results(
            extractor, schema, text, args.chunk_chars
        )
        total_entities += len(results)

        prediction = build_prediction_block(results, confidences, args.model)
        task_out = {"data": task["data"], "predictions": [prediction]}

        out_path = output_dir / path.name
        out_path.write_text(
            json.dumps([task_out], ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        combined.append(task_out)

        print(
            f"  [{idx}/{len(task_files)}] {path.name}: "
            f"{len(results)} entities (avg conf {prediction['score']:.2f})"
        )

    combined_path = output_dir / args.combined_name
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=== Summary ===")
    print(f"  tasks processed   : {len(combined)}")
    print(f"  total entities    : {total_entities}")
    print(f"  per-letter output : {output_dir.resolve()}")
    print(f"  combined output   : {combined_path.resolve()}")


if __name__ == "__main__":
    main()
