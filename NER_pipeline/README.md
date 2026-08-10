# NER_pipeline — GLiNER2 zero-shot preannotation

Reusable zero-shot NER preannotation pipeline for the Navez letters corpus
(19th-century French handwritten correspondence). Reads Label Studio
import tasks, runs [GLiNER2](https://github.com/fastino-ai/GLiNER2) with a
custom French/19c-art entity schema, and writes prediction-augmented copies
back out — ready for human correction in Label Studio.

## Label scheme

Defined in `entity_schema.py` (the single source of truth). Edit the
`ENTITY_SCHEMA` dict to refine descriptions or thresholds and re-run.

| Label     | Meaning                                                            |
|-----------|--------------------------------------------------------------------|
| `PER`     | Proper personal names (`"Monsieur David"`, `"le baron de Stassart"`) |
| `PER_NOM` | Nominal/indirect person references (`"ma nièce"`, `"le ministre"`)  |
| `ART`     | Explicit artwork titles (`"Agar dans le désert"`)                   |
| `ART_NOM` | Nominal artwork references (`"mon grand tableau"`, `"la toile"`)    |
| `EVT`     | Named events / salons (`"le Salon de 1814"`)                        |
| `ORG`     | Institutions, academies (`"l'Académie de Bruxelles"`)               |
| `LOC`     | Geographic locations (`"Rome"`, `"la rue de l'Hôpital"`)            |

The codes match the `<Labels>` block of the Label Studio `<View>` schema
used in `data/labelstudio_upload/*_ls_import.json` (note: `EVT`, not `EVE`).

## Install

GLiNER2 is already installed in the system Python on this machine
(`gliner2 1.2.4`). To install it into the project venv instead:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r NER_pipeline\requirements.txt
```

First run will download the model weights from HuggingFace (a few hundred MB).

### Available models

| Model id                         | Size | Notes                                                |
|----------------------------------|------|------------------------------------------------------|
| `fastino/gliner2-multi-v1`       | 0.3B | **Multilingual — default, best for French letters**  |
| `fastino/gliner2-large-v1`       | 0.5B | English; highest quality on English text             |
| `fastino/gliner2-base-v1`        | 0.2B | English; smallest / fastest                          |

## Run

From the repo root:

```bash
# Smoke test on 1 letter (uses the multilingual default)
python NER_pipeline/gliner_predict.py --limit 1

# Full run over everything in data/labelstudio_upload/
python NER_pipeline/gliner_predict.py

# Try the English large model (slower, lower recall on French)
python NER_pipeline/gliner_predict.py --model fastino/gliner2-large-v1

# Force CPU (default lets GLiNER2 pick)
python NER_pipeline/gliner_predict.py --device cpu
```

CLI flags:

| Flag              | Default                                 | What it does                              |
|-------------------|-----------------------------------------|-------------------------------------------|
| `--input-dir`     | `data/labelstudio_upload`               | Folder of `*_ls_import.json` tasks        |
| `--output-dir`    | `data/preannots_GLiNER`                 | Where to write prediction files           |
| `--combined-name` | `gliner_preannots.json`                 | Filename for the combined import array    |
| `--model`         | `fastino/gliner2-multi-v1`              | HuggingFace model id                      |
| `--chunk-chars`   | `1500`                                  | Approx max chars per inference chunk      |
| `--limit`         | `0` (all)                               | Cap on number of letter files processed   |
| `--device`        | (auto)                                  | `cpu`, `cuda`, `cuda:0`, ...              |

## Output format

Each output file is a standard Label Studio import: a one-element JSON array
containing the original `data` block plus a `predictions` block:

```json
[
  {
    "data": { "text": "...", "meta_date": "...", ... },
    "predictions": [
      {
        "model_version": "gliner2-zeroshot::fastino/gliner2-multi-v1",
        "score": 0.71,
        "result": [
          {
            "id": "a1b2c3d4e5",
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
              "start": 48,
              "end": 50,
              "text": "Je",
              "labels": ["PER_NOM"]
            },
            "score": 0.62
          }
        ]
      }
    ]
  }
]
```

The combined file `gliner_preannots.json` holds all tasks in one array — ideal
for a single Label Studio bulk import.

## Iterating on the scheme

1. Edit descriptions or per-label `threshold` values in
   [`entity_schema.py`](./entity_schema.py).
2. Re-run `gliner_predict.py` (it overwrites the output folder cleanly).
3. Re-import into Label Studio.

To change the label set itself (add/remove a label):
1. Edit `ENTITY_SCHEMA` in `entity_schema.py`.
2. Edit the `<Labels>` block in your Label Studio project settings so the
   codes match — predictions referencing an unknown label will be ignored.

## Notes on chunking & spans

Letters longer than `--chunk-chars` are split on newline boundaries (no
mid-word splits). Character offsets are mapped back into the original text,
and exact duplicates across chunks are de-duplicated. Spans where the
extracted surface form doesn't match `text[start:end]` are repaired by
trusting the offsets.

## Known caveats

* **Zero-shot ≠ gold.** Expect noticeable false positives on `PER` vs
  `PER_NOM` (the hardest contrast in this scheme). Tune `PER_NOM` threshold
  upwards if needed.
* **Old-French orthography** in handwritten letters (`temoigné`, `etoit`,
  `tres`, abbreviated `9bre` etc.) can cause the model to miss spans —
  reflected as recall, not precision.
* **Confidence scores** are model-internal and not calibrated; treat them
  as relative ranks rather than absolute probabilities.
