# NER evaluation — 5-fold stratified CV (sender x decade)

Gold: 561 unique letters, ART_NOM merged into ART.
Metrics: micro-averaged across letters within a fold; means / CIs across folds (or across runs for LLM variance).

## Table 1 — Fold variance (1 run per fold, all models)

Mean lenient F1 ± 95% normal-approximation CI across 5 folds.

| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) | PER strict F1 | PER lenient F1 |
|---|---|---|---|---|
| camembertav2 | 0.416 ± 0.027 | 0.593 ± 0.027 | 0.596 | 0.767 |
| claude-opus-4.8 | 0.561 ± 0.022 | 0.719 ± 0.019 | 0.718 | 0.869 |
| dalembert | 0.509 ± 0.027 | 0.668 ± 0.028 | 0.694 | 0.839 |
| gliner2 | 0.342 ± 0.008 | 0.384 ± 0.004 | 0.230 | 0.255 |
| gliner2-ft | 0.527 ± 0.057 | 0.648 ± 0.068 | 0.647 | 0.768 |
| qwen3-235b-a22b-2507 | 0.310 ± 0.018 | 0.406 ± 0.017 | 0.441 | 0.544 |
| qwen3-vl-30b-a3b-instruct | 0.306 ± 0.034 | 0.379 ± 0.033 | 0.481 | 0.570 |

## Table 2 — LLM run variance (fold 0, 3 runs per model)

Mean lenient F1 ± 95% normal-approximation CI across 3 runs on fold 0 letters (LLMs only).

| Model | Strict F1 (mean±CI) | Lenient F1 (mean±CI) |
|---|---|---|
| claude-opus-4.8 (3 runs) | 0.544 ± 0.003 | 0.701 ± 0.002 |
| qwen3-235b-a22b-2507 (3 runs) | 0.315 ± 0.014 | 0.399 ± 0.014 |
| qwen3-vl-30b-a3b-instruct (3 runs) | 0.320 ± 0.018 | 0.392 ± 0.019 |

## Per-label breakdown (fold 0, run 0)

| Model | label | Strict F1 | Lenient F1 |
|---|---|---|---|
| camembertav2 | ART | 0.175 | 0.419 |
| camembertav2 | EVT | 0.230 | 0.585 |
| camembertav2 | LOC | 0.622 | 0.715 |
| camembertav2 | ORG | 0.220 | 0.541 |
| camembertav2 | PER | 0.594 | 0.749 |
| camembertav2 | PER_NOM | 0.197 | 0.361 |
| claude-opus-4.8 | ART | 0.264 | 0.425 |
| claude-opus-4.8 | EVT | 0.262 | 0.570 |
| claude-opus-4.8 | LOC | 0.718 | 0.828 |
| claude-opus-4.8 | ORG | 0.238 | 0.632 |
| claude-opus-4.8 | PER | 0.751 | 0.875 |
| claude-opus-4.8 | PER_NOM | 0.316 | 0.452 |
| dalembert | ART | 0.228 | 0.480 |
| dalembert | EVT | 0.284 | 0.515 |
| dalembert | LOC | 0.713 | 0.798 |
| dalembert | ORG | 0.291 | 0.567 |
| dalembert | PER | 0.701 | 0.802 |
| dalembert | PER_NOM | 0.252 | 0.423 |
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
| qwen3-235b-a22b-2507 | EVT | 0.085 | 0.236 |
| qwen3-235b-a22b-2507 | LOC | 0.461 | 0.511 |
| qwen3-235b-a22b-2507 | ORG | 0.145 | 0.370 |
| qwen3-235b-a22b-2507 | PER | 0.482 | 0.553 |
| qwen3-235b-a22b-2507 | PER_NOM | 0.136 | 0.191 |
| qwen3-vl-30b-a3b-instruct | ART | 0.085 | 0.141 |
| qwen3-vl-30b-a3b-instruct | EVT | 0.043 | 0.139 |
| qwen3-vl-30b-a3b-instruct | LOC | 0.456 | 0.498 |
| qwen3-vl-30b-a3b-instruct | ORG | 0.105 | 0.287 |
| qwen3-vl-30b-a3b-instruct | PER | 0.504 | 0.574 |
| qwen3-vl-30b-a3b-instruct | PER_NOM | 0.117 | 0.169 |