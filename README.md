# Measuring human labour for NER post-correction — code and evaluation artefacts

Anonymised code accompanying the CHR submission *"Agréez, Monsieur, mes entités
les plus distinguées: Measuring Human Labour for NER Post-Correction"*.

This repository contains the code that produced every number, table and figure
in the paper, together with the evaluation artefacts needed to re-run the
analysis without access to the image corpus.

Everything here is anonymised: the four annotators appear as `A1`–`A4`, and
author, institution and account names have been removed.

---

## What is where

```
NER_pipeline/          Sections 4.2-4.4: agreement, model evaluation, taxonomy, cost
  eval/                frozen gold standard, fold assignment, per-model predictions
  IAA_annots/          the two inter-annotator agreement rounds
HTR_pipeline/          Section 4.1: transcription, CER/WER, HTR post-correction
  evaluation/          the HTR correction report
```

## Reproducing the paper

Install the requirements (Python 3.11+), then run the steps below from the
repository root. Steps 1–5 need **no data beyond what is in this repository**.

```bash
pip install -r requirements.txt
```

### 1. Validity filter (Appendix A.2, Table 7)

Removes predicted spans that do not denote a readable substring of the letter.
Everything downstream is scored on the filtered output.

```bash
python NER_pipeline/apply_validity_filter.py
```

Writes `NER_pipeline/eval/predictions_filtered/` and prints the per-system drop
table reported in the appendix.

### 2. Accuracy (Section 4.3, Figure 2)

```bash
python NER_pipeline/compute_ner_metrics.py \
    --pred-dir NER_pipeline/eval/predictions_filtered --variance-fold 0
python NER_pipeline/plot_ner_results.py
```

Writes `ner_final_summary_filtered.md`, `ner_final_metrics_filtered.csv` and
`figures/fig_overall_filtered.pdf`.

### 3. Error taxonomy (Section 4.4, Figures 3 and 4)

```bash
python NER_pipeline/error_taxonomy.py \
    --pred-dir NER_pipeline/eval/predictions_filtered
python NER_pipeline/plot_error_taxonomy.py
```

Writes `error_taxonomy_filtered.{csv,md}`, the per-event CSV, and
`figures/fig_taxonomy_{burden,composition}_filtered.pdf`.

The run asserts, for every system, the identity stated in the paper:

```
events + matched imperfect pairs == strict FN + strict FP
```

If that assertion fails the taxonomy and the strict scores have drifted apart,
and no result should be trusted.

### 4. Qualitative error analysis (Section 4.4)

```bash
python NER_pipeline/error_analysis.py
```

### 5. Projected cost (supplementary)

```bash
python NER_pipeline/project_correction_cost.py
```

Prices each system's event profile using the coefficients fitted in step 6.
Defaults to the filtered taxonomy; pass `--taxonomy-csv` to price the
unfiltered counts instead. This table is supplementary and is not cited in the
paper.

### 6. Steps that need the annotation logs

The remaining scripts read the Label Studio exports and the page images, which
are **not** included here (see *Data availability*).

| script | produces |
|---|---|
| `NER_pipeline/analyze_correction_time.py` | the cost calibration of Section 3.2 |
| `NER_pipeline/build_ner_folds.py` | the frozen 561-letter pool and fold split |
| `NER_pipeline/calculate_ner_iaa.py` | agreement for one IAA round (Section 4.2) |
| `NER_pipeline/compute_iaa_707.py` | agreement for the second IAA round |
| `NER_pipeline/plot_label_counts.py` | span counts per label |
| `NER_pipeline/gliner_baseline_eval.py` | the zero-shot baseline |
| `HTR_pipeline/compute_htr_cer.py` | CER/WER of Section 4.1 |
| `HTR_pipeline/analyze_htr_correction_cost.py` | Figure 1 and the HTR tables |

Their outputs are included as `.md` / `.csv` under `NER_pipeline/eval/`,
`NER_pipeline/IAA_annots/` and `HTR_pipeline/evaluation/`, so the reported
numbers can be checked even though the inputs are withheld.

---

## Which script produced which result

| paper item | script |
|---|---|
| Table 1, corpus overview | `HTR_pipeline/build_letters_overview.py` |
| Table 2, the eight classes | `NER_pipeline/error_taxonomy.py` (`TAXONOMY`) |
| Tables 3–5, transcription quality | `HTR_pipeline/compute_htr_cer.py` |
| Figure 1, HTR edit classes | `HTR_pipeline/analyze_htr_correction_cost.py` |
| Table 6, inter-annotator agreement | `NER_pipeline/calculate_ner_iaa.py`, `compute_iaa_707.py` |
| Figure 2, strict and lenient F1 | `NER_pipeline/plot_ner_results.py` |
| Figures 3–4, burden and composition | `NER_pipeline/plot_error_taxonomy.py` |
| Table 7, validity filter drops | `NER_pipeline/apply_validity_filter.py` |
| Section 3.2, cost calibration | `NER_pipeline/analyze_correction_time.py` |
| Appendix A, class definitions | `NER_pipeline/error_taxonomy.py` |
| Appendix B, transcription prompt | `HTR_pipeline/run_htr_opus.py` |
| Appendix B, entity schema | `NER_pipeline/entity_schema.py` |

## Determinism

Fold assignment, sampling and example selection use a fixed seed (42) and
sorted iteration order. Prompted models were called at temperature 0. Steps 1–4
above are fully deterministic given the shipped prediction files.

## Data availability

The 561-letter gold standard is included in
`NER_pipeline/eval/gold_561_merged.jsonl` as letter text plus character-offset
spans, which is what every evaluation script consumes.

Not included:

* the page images (roughly 36 GB of archival scans), which are held by the
  holding library and are not ours to redistribute;
* the raw Label Studio exports containing per-annotator timing logs, which are
  withheld because they are personal working-time data. The aggregate
  per-annotator statistics used in the paper are included, under `A1`–`A4`.

## Anonymisation

This tree is generated from the working repository by a build script that
rewrites annotator names to `A1`–`A4` (in file contents, file names and JSON
keys), strips author, account and institution names, and refuses to copy any
file matching a credential pattern. The build fails if any identifying string
survives the rewrite.

One consequence worth knowing when reading the code: paths such as
`data/labelstudio_output/NER_707_A1_final.json` refer to files that are not
shipped, so the scripts in step 6 will not find their inputs. This is expected.
