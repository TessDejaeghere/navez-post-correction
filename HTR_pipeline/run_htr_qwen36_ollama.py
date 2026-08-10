#!/usr/bin/env python3
"""
Run Qwen3.6 (Ollama, multimodal) HTR over the 10-letter pilot manifest,
talking to the Ollama endpoint that our research centre installed on
``<gpu-host>`` (default ``http://127.0.0.1:6767``).

Mirrors ``archive/iaa_pipeline/run_qwen35_htr_pilot.py``:
  * Same MANIFEST (kept in sync below).
  * Same SYSTEM_PROMPT / user_prompt.
  * Same image preprocessing (max 2048 px long edge, JPEG q=92).
  * Outputs land in ``notebooks/pilot_out/predicted_txt/<display-name>/`` so
    the evaluation cells in ``notebooks/htr_vlm_pilot.ipynb`` pick them up
    after adding the display name to ``MODELS``.

Differences vs the vLLM version:
  * Network client only — no local GPU work, no vLLM dependency.
  * Uses Ollama's native ``/api/chat`` endpoint because (a) ``think:false``
    works there as a top-level field (the OpenAI-compat surface routes
    qwen3.6's reasoning into a non-standard ``reasoning`` field and easily
    blows past ``max_tokens`` before any user-facing content is emitted),
    and (b) images go in a single ``images: [b64, ...]`` list on the user
    message, which is simpler than the multimodal-parts schema.

Pre-req — open the tunnel once per laptop session:

    ssh -fN -L 6767:127.0.0.1:6767 spark

Quick smoke test (single letter):

    python HTR_pipeline/run_htr_qwen36_ollama.py --only VI_Ms_II_70_1_74

Full 10-letter pilot:

    python HTR_pipeline/run_htr_qwen36_ollama.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image


# Windows default cp1252 console crashes on the arrow / accents we print.
# Reconfigure stdout/stderr to UTF-8 early so progress messages don't blow up.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Manifest — kept in sync with archive/iaa_pipeline/run_qwen35_htr_pilot.py
# and notebooks/htr_vlm_pilot.ipynb cell #VSC-db57075c.
# Paths are RELATIVE to --data-root (default: data/voorbeeldbrieven).
# ---------------------------------------------------------------------------
MANIFEST: list[dict] = [
    {"letter_id": "I_Ms_II_70_3_570",
     "image_paths": [
         "I/BE-KBR00_21245490_00000000_00_00_00_0_01_0001_31979564.jpeg",
         "I/BE-KBR00_21245490_00000000_00_00_00_0_01_0002_31979565.jpeg"],
     "sender": "Jean Victor Schnetz (1787-1870)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1863-02-20"},
    {"letter_id": "II_Ms_II_70_1_162",
     "image_paths": [
         "II/BE-KBR00_17132606_00000000_01_00_00_0_01_0001_31138636.jpeg",
         "II/BE-KBR00_17132606_00000000_01_00_00_0_01_0002_31138637.jpeg"],
     "sender": "Balthasar Paul Ommeganck (1755-1826)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1825-08-05"},
    {"letter_id": "III_Ms_II_70_1_185",
     "image_paths": [
         "III/BE-KBR00_17136458_00000000_01_00_00_0_01_0001_31162268.jpeg",
         "III/BE-KBR00_17136458_00000000_01_00_00_0_01_0002_31162269.jpeg"],
     "sender": "Jean François Portaels (1818-1895)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1844-12-20"},
    {"letter_id": "IV_Ms_II_70_2_11",
     "image_paths": [
         "IV/BE-KBR00_17402657_00000000_00_00_00_0_01_0001_31407114.jpeg",
         "IV/BE-KBR00_17402657_00000000_00_00_00_0_01_0002_31407115.jpeg"],
     "sender": "Annette Marotte / Marguerite Gabrielle Marotte-Willemin (1810-1866)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1840-09-24"},
    {"letter_id": "V_Ms_II_70_1_27",
     "image_paths": [
         "V/BE-KBR00_17104444_00000000_01_00_00_0_01_0001_30870694.jpeg",
         "V/BE-KBR00_17104444_00000000_01_00_00_0_01_0002_30870695.jpeg"],
     "sender": "Amélie Champein (1812-1898)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1857-08-06"},
    {"letter_id": "VI_Ms_II_70_1_74",
     "image_paths": [
         "VI/BE-KBR00_17113779_00000000_01_00_00_0_01_0001_30975451.jpeg"],
     "sender": "Fanny Geefs (1807-1883)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "unknown"},
    {"letter_id": "VII_Ms_II_70_1_108",
     "image_paths": [
         "VII/BE-KBR00_17118155_00000000_01_00_00_0_01_0001_31344242.jpeg",
         "VII/BE-KBR00_17118155_00000000_01_00_00_0_01_0002_31344243.jpeg"],
     "sender": "Louis Jéhotte (1803-1884)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1858-01-14"},
    {"letter_id": "VIII_Ms_II_70_1_122",
     "image_paths": [
         "VIII/BE-KBR00_17119774_00000000_01_00_00_0_01_0001_31189190.jpeg",
         "VIII/BE-KBR00_17119774_00000000_01_00_00_0_01_0002_31189191.jpeg"],
     "sender": "Jean-Baptiste De Landtsheer (1797-1874)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1825-09-30"},
    {"letter_id": "IX_Ms_II_70_6_49",
     "image_paths": [
         "IX/BE-KBR00_21360901_00000000_00_00_00_0_01_0001_31913451.jpeg",
         "IX/BE-KBR00_21360901_00000000_00_00_00_0_01_0002_31913452.jpeg"],
     "sender": "Joachim Heinrich Wilhelm Wagener (1782-1861)",
     "receiver": "François-Joseph Navez (1787-1869)",
     "date": "1844-10-30"},
    {"letter_id": "X_Ms_II_70_6_48",
     "image_paths": [
         "X/BE-KBR00_21360849_00000000_00_00_00_0_01_0001_31913458.tiff",
         "X/BE-KBR00_21360849_00000000_00_00_00_0_01_0002_31913459.jpeg"],
     "sender": "François-Joseph Navez (1787-1869)",
     "receiver": "Joachim Heinrich Wilhelm Wagener (1782-1861)",
     "date": "1844-10-26"},
]


# ---------------------------------------------------------------------------
# Prompts — kept identical to the vLLM pilot.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert archivist transcribing 19th-century French handwritten "
    "letters from the François-Joseph Navez correspondence (Belgian neo-classicist "
    "painter, 1787-1869). \n\n"
    "CRITICAL RULES:\n"
    "1. Preserve the EXACT orthography of the source. Do NOT modernise archaic "
    "forms (keep `étoit` as `étoit`, `avoit` as `avoit`, `tems` as `tems`). "
    "Do NOT archaize modern forms (do not change `était` to `étoit`). Whatever "
    "the writer wrote, copy it.\n"
    "2. Preserve line breaks as in the original.\n"
    "3. Preserve ligatures (`œ`, `æ`) and all diacritics.\n"
    "4. Do NOT translate. Do NOT summarise. Do NOT add commentary or notes.\n"
    "5. If a word is illegible, use [...] (three dots in brackets). Do not guess.\n"
    "6. Output ONLY the transcription. No preamble, no markdown, no JSON wrapper.\n\n"
    "PHYSICAL LAYOUT (bifolia):\n"
    "These letters are typically written on a single sheet of paper folded once "
    "(a bifolium = 4 written sides, like a small booklet). The scan order is "
    "NOT necessarily the reading order. In particular:\n"
    "- An ADDRESS PANEL (mostly blank, with a recipient name and a city, sometimes "
    "  postal stamps or seals) belongs at the END of the transcription as a block "
    "  prefixed `[Address:]`. Do NOT transcribe it as part of the letter body, "
    "  even if it appears on the first scan.\n"
    "- A POSTSCRIPT or text written sideways, in a margin, or crossed over earlier "
    "  text (cross-writing) belongs at the END of the body, prefixed `[PS:]` or "
    "  `[Margin:]`. Do NOT interleave it with the main text.\n"
    "- The main letter body should be transcribed in the order the writer "
    "  intended (greeting → body → closing → signature), reconstructing reading "
    "  order across pages even if the scans are out of order.\n"
    "- If you cannot determine the intended order, transcribe each visible block "
    "  separately and label them `[Page 1]`, `[Page 2]`, etc.\n\n"
    "Output structure:\n"
    "  <main letter body, in reading order>\n"
    "  [PS: ...]        (only if there is a postscript)\n"
    "  [Margin: ...]    (only if there is marginalia / cross-writing)\n"
    "  [Address: ...]   (only if an address panel is visible)"
)


def user_prompt(sender: str, receiver: str, date: str, n_pages: int) -> str:
    if n_pages == 1:
        page_line = (
            "Transcribe the attached letter image in full. Output only the "
            "transcription."
        )
    else:
        page_line = (
            f"This letter is attached as {n_pages} images, one per page, in "
            f"reading order (page 1 first). Transcribe ALL pages as ONE continuous "
            f"text in reading order. Do not repeat or label pages in the output. "
            f"Output only the transcription."
        )
    return (
        f"Letter metadata (for context only — do not include in output):\n"
        f"- Sender:   {sender}\n"
        f"- Receiver: {receiver}\n"
        f"- Date:     {date}\n"
        f"- Pages:    {n_pages}\n\n"
        f"{page_line}"
    )


# ---------------------------------------------------------------------------
# Image preprocessing — matches the vLLM pilot (max 2048 px long edge,
# JPEG q=92). Ollama wants raw base64 (no data: URI prefix) in `images`.
# ---------------------------------------------------------------------------
def image_to_b64(image_path: Path, max_dim: int) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_dim:
            scale = max_dim / max(im.size)
            im = im.resize(
                (int(im.size[0] * scale), int(im.size[1] * scale)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_ollama_payload(
    letter: dict,
    data_root: Path,
    max_dim: int,
    model: str,
    temperature: float,
    num_predict: int,
    think: bool,
    repeat_penalty: float,
    repeat_last_n: int,
    mirostat: int,
    mirostat_tau: float,
    mirostat_eta: float,
) -> dict:
    n_pages = len(letter["image_paths"])
    user_text = user_prompt(letter["sender"], letter["receiver"],
                            letter["date"], n_pages)
    images_b64: list[str] = []
    for rel in letter["image_paths"]:
        img_path = data_root / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        images_b64.append(image_to_b64(img_path, max_dim))

    options = {
        "temperature": temperature,
        "num_predict": num_predict,
        # Letters can be long; default ctx (256k for qwen3.6) is fine but
        # being explicit keeps behaviour predictable across model swaps.
        "num_ctx": 32768,
        # Anti-loop: greedy decoding (temp=0.0) on hard scans can fall
        # into multi-token repetition cycles. Ollama's default
        # repeat_last_n=64 is too narrow to catch a 5-token cycle;
        # bump to 256 and raise penalty slightly.
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
    }
    if mirostat:
        options["mirostat"] = mirostat
        options["mirostat_tau"] = mirostat_tau
        options["mirostat_eta"] = mirostat_eta

    return {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text, "images": images_b64},
        ],
        "options": options,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", type=Path,
                   default=Path("data/voorbeeldbrieven"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: notebooks/pilot_out/predicted_txt/<display-name>/")
    p.add_argument("--usage-log", type=Path,
                   default=Path("notebooks/pilot_out/usage_log.jsonl"))
    p.add_argument("--endpoint",
                   default="http://127.0.0.1:6767",
                   help="Ollama base URL (no trailing /). Tunnel from spark.")
    p.add_argument("--model", default="qwen3.6:latest")
    p.add_argument("--display-name", default="qwen3.6-ollama",
                   help="Tag used for the output sub-folder and usage_log entries.")
    p.add_argument("--max-image-dim", type=int, default=2048)
    p.add_argument("--num-predict", type=int, default=3000,
                   help="Max output tokens (Ollama's num_predict ≅ max_tokens). "
                        "Default 3000 — enough for any normal letter and caps "
                        "damage if the model falls into a repetition loop.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repeat-penalty", type=float, default=1.15,
                   help="Ollama anti-repetition penalty (default 1.15; ollama default 1.1).")
    p.add_argument("--repeat-last-n", type=int, default=256,
                   help="How many recent tokens to scan for repetition (default 256; ollama default 64).")
    p.add_argument("--mirostat", type=int, default=0, choices=[0, 1, 2],
                   help="0=off (default), 1=mirostat v1, 2=mirostat v2. "
                        "Use 2 for hard scans where the model gets stuck repeating a phrase.")
    p.add_argument("--mirostat-tau", type=float, default=5.0,
                   help="Target surprise (lower = more focused, higher = more diverse).")
    p.add_argument("--mirostat-eta", type=float, default=0.1,
                   help="Mirostat learning rate.")
    p.add_argument("--think", action="store_true",
                   help="Allow the reasoning phase (default: disabled — see note "
                        "in module docstring).")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="HTTP timeout per request, seconds.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional letter_id filter (subset for debugging).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = (args.output_dir or
               Path("notebooks/pilot_out/predicted_txt") / args.display_name).resolve()
    data_root = args.data_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.usage_log.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Endpoint:     {args.endpoint}")
    print(f"Model:        {args.model}")
    print(f"Display:      {args.display_name}")
    print(f"Think mode:   {args.think}")
    print(f"Data root:    {data_root}")
    print(f"Output dir:   {out_dir}")
    print(f"Usage log:    {args.usage_log.resolve()}")
    print("=" * 70)

    # Pre-flight: endpoint reachable and model available.
    try:
        r = requests.get(f"{args.endpoint}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot reach {args.endpoint}/api/tags: {e}", file=sys.stderr)
        print("Hint: open the SSH tunnel first:\n"
              "  ssh -fN -L 6767:127.0.0.1:6767 spark", file=sys.stderr)
        return 2
    served = {m["name"] for m in r.json().get("models", [])}
    if args.model not in served:
        print(f"ERROR: model {args.model!r} not served. Available: {sorted(served)}",
              file=sys.stderr)
        return 2

    # Select letters.
    todo = []
    for letter in MANIFEST:
        if args.only and letter["letter_id"] not in args.only:
            continue
        out_path = out_dir / f"{letter['letter_id']}.txt"
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            print(f"  [skip] {letter['letter_id']} (exists)")
            continue
        todo.append(letter)

    if not todo:
        print("Nothing to do.")
        return 0

    # Pre-flight images.
    missing = [str(data_root / rel) for l in todo for rel in l["image_paths"]
               if not (data_root / rel).exists()]
    if missing:
        print("ERROR: missing image files:\n  " + "\n  ".join(missing),
              file=sys.stderr)
        return 2

    chat_url = f"{args.endpoint}/api/chat"
    n_ok = 0
    with args.usage_log.open("a", encoding="utf-8") as ulog:
        for letter in todo:
            print(f"\n→ {letter['letter_id']} ({len(letter['image_paths'])} page(s))")
            payload = build_ollama_payload(
                letter, data_root, args.max_image_dim,
                args.model, args.temperature, args.num_predict, args.think,
                args.repeat_penalty, args.repeat_last_n,
                args.mirostat, args.mirostat_tau, args.mirostat_eta,
            )
            t0 = time.time()
            try:
                resp = requests.post(chat_url, json=payload, timeout=args.timeout)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [ERR] HTTP failure: {e}", file=sys.stderr)
                continue
            dt = time.time() - t0
            data = resp.json()
            text = (data.get("message") or {}).get("content", "").strip()
            done_reason = data.get("done_reason", "?")

            out_path = out_dir / f"{letter['letter_id']}.txt"
            out_path.write_text(text, encoding="utf-8")
            n_ok += 1

            usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns": data.get("load_duration"),
                "eval_duration_ns": data.get("eval_duration"),
            }
            ulog.write(json.dumps({
                "model_slug": args.model,
                "display": args.display_name,
                "provider": "ollama-spark",
                "letter_id": letter["letter_id"],
                "n_pages": len(letter["image_paths"]),
                "chars_out": len(text),
                "wall_time_s": round(dt, 2),
                "done_reason": done_reason,
                "usage": usage,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }) + "\n")
            print(f"  [ok]   {len(text):>5} chars in {dt:.1f}s "
                  f"(done_reason={done_reason}, "
                  f"out_tok={usage['completion_tokens']})")

    print(f"\nDone. {n_ok}/{len(todo)} predictions written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
