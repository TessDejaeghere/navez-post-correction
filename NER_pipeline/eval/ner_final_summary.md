# NER evaluation — 5-fold stratified CV (sender x decade)

Gold: 561 unique letters, ART_NOM merged into ART.
Metrics: micro-averaged across letters within a fold; means / CIs across folds (or across runs for LLM variance).

## Table 1 — Fold variance (1 run per fold, all models)

Mean lenient F1 ± 95% normal-approximation CI across 5 folds.

| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) | PER strict F1 | PER lenient F1 |
|---|---|---|---|---|
| camembertav2 | 0.393 ± 0.026 | 0.560 ± 0.026 | 0.579 | 0.745 |
| claude-opus-4.8 | 0.561 ± 0.022 | 0.719 ± 0.019 | 0.718 | 0.869 |
| dalembert | 0.504 ± 0.028 | 0.661 ± 0.029 | 0.690 | 0.834 |
| gliner2 | 0.342 ± 0.008 | 0.384 ± 0.004 | 0.230 | 0.255 |
| gliner2-ft | 0.527 ± 0.057 | 0.649 ± 0.068 | 0.647 | 0.769 |
| qwen3-235b-a22b-2507 | 0.309 ± 0.018 | 0.406 ± 0.017 | 0.441 | 0.544 |
| qwen3-vl-30b-a3b-instruct | 0.305 ± 0.033 | 0.377 ± 0.033 | 0.481 | 0.570 |

## Table 2 — LLM run variance (fold 0, 3 runs per model)

Mean lenient F1 ± 95% normal-approximation CI across 3 runs on fold 0 letters (LLMs only).

| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) |
|---|---|---|
| claude-opus-4.8 (3 runs) | 0.544 ± 0.003 | 0.701 ± 0.002 |
| qwen3-235b-a22b-2507 (3 runs) | 0.315 ± 0.015 | 0.398 ± 0.014 |
| qwen3-vl-30b-a3b-instruct (3 runs) | 0.319 ± 0.017 | 0.391 ± 0.018 |

## Per-label breakdown (fold 0, run 0)

| Model | label | Strict F1 | Lenient F1 |
|---|---|---|---|
| camembertav2 | ART | 0.157 | 0.374 |
| camembertav2 | EVT | 0.219 | 0.556 |
| camembertav2 | LOC | 0.601 | 0.692 |
| camembertav2 | ORG | 0.203 | 0.502 |
| camembertav2 | PER | 0.570 | 0.721 |
| camembertav2 | PER_NOM | 0.182 | 0.335 |
| claude-opus-4.8 | ART | 0.264 | 0.425 |
| claude-opus-4.8 | EVT | 0.262 | 0.570 |
| claude-opus-4.8 | LOC | 0.718 | 0.828 |
| claude-opus-4.8 | ORG | 0.238 | 0.632 |
| claude-opus-4.8 | PER | 0.751 | 0.875 |
| claude-opus-4.8 | PER_NOM | 0.316 | 0.452 |
| dalembert | ART | 0.224 | 0.472 |
| dalembert | EVT | 0.275 | 0.500 |
| dalembert | LOC | 0.705 | 0.788 |
| dalembert | ORG | 0.285 | 0.555 |
| dalembert | PER | 0.697 | 0.798 |
| dalembert | PER_NOM | 0.248 | 0.418 |
| gliner2 | ART | 0.224 | 0.343 |
| gliner2 | EVT | 0.381 | 0.464 |
| gliner2 | LOC | 0.759 | 0.761 |
| gliner2 | ORG | 0.501 | 0.623 |
| gliner2 | PER | 0.247 | 0.269 |
| gliner2 | PER_NOM | 0.090 | 0.099 |
| gliner2-ft | ART | 0.366 | 0.556 |
| gliner2-ft | EVT | 0.464 | 0.609 |
| gliner2-ft | LOC | 0.823 | 0.878 |
| gliner2-ft | ORG | 0.414 | 0.668 |
| gliner2-ft | PER | 0.743 | 0.849 |
| gliner2-ft | PER_NOM | 0.361 | 0.488 |
| qwen3-235b-a22b-2507 | ART | 0.128 | 0.218 |
| qwen3-235b-a22b-2507 | EVT | 0.084 | 0.233 |
| qwen3-235b-a22b-2507 | LOC | 0.461 | 0.511 |
| qwen3-235b-a22b-2507 | ORG | 0.145 | 0.370 |
| qwen3-235b-a22b-2507 | PER | 0.482 | 0.553 |
| qwen3-235b-a22b-2507 | PER_NOM | 0.136 | 0.191 |
| qwen3-vl-30b-a3b-instruct | ART | 0.084 | 0.140 |
| qwen3-vl-30b-a3b-instruct | EVT | 0.043 | 0.138 |
| qwen3-vl-30b-a3b-instruct | LOC | 0.455 | 0.497 |
| qwen3-vl-30b-a3b-instruct | ORG | 0.105 | 0.287 |
| qwen3-vl-30b-a3b-instruct | PER | 0.504 | 0.574 |
| qwen3-vl-30b-a3b-instruct | PER_NOM | 0.117 | 0.169 |