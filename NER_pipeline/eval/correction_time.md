# Measured post-correction time per error class

Estimated from the annotators' own edit log: every NER letter was pre-annotated with GLiNER2 and corrected in Label Studio, which records `lead_time` (active-tab seconds) and, for every span, whether it was accepted, edited or created by hand.

## 1. Sample

- 1128 submitted tasks from 4 annotators over 554 distinct letters.
- 908 of those carry the GLiNER2 suggestion object and can be diffed span by span (220 cannot: the export for one annotator dropped the prediction payload).
- Time on task: mean 7.7 min, median 4.7 min, p90 17.2 min.
- Total recorded effort: 145.4 hours.

| annotator | tasks | median min | mean min | accepted | edited | manual |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 206 | 6.6 | 9.3 | 1317 | 954 | 3051 |
| A2 | 249 | 6.6 | 10.1 | 1217 | 486 | 4864 |
| A3 | 310 | 4.1 | 7.0 | 1151 | 1286 | 3889 |
| A4 | 363 | 3.7 | 5.8 | 1827 | 81 | 5987 |

Annotator style differs sharply and is absorbed by fixed effects in every model below: some annotators edit a suggestion, others delete it and re-draw the span by hand, which moves the same correction from the `prediction-changed` column to the `manual` one.

## 2. What GLiNER2's output cost, by class

Excluded from this analysis: `F0` (malformed span) — 2 events. `F0` is not a taxonomy class: it is what the validity filter deletes before a corrector sees it, so it is not part of the cost model.

| class | error class | cost | events | % of spans touched |
|---|---|---|---:|---:|
| `A0` | accepted suggestion | (none) | 4245 | 16.3% |
| `M1` | redundant span | mechanical | 1041 | 4.0% |
| `M2` | affix drift | mechanical | 134 | 0.5% |
| `M3` | token drift | mechanical | 69 | 0.3% |
| `M4` | decidable label flip | mechanical | 1261 | 4.8% |
| `C1` | missed mention | cognitive | 14478 | 55.4% |
| `C2` | spurious mention | cognitive | 4554 | 17.4% |
| `C3` | semantic label confusion | cognitive | 274 | 1.0% |
| `C4` | mention segmentation | cognitive | 65 | 0.2% |

## 3. How long does each kind of correction take?

`minutes ~ event counts + letter length + annotator`, OLS with HC1 robust standard errors. Coefficients are absolute seconds per event; `chars_k` is seconds per 1,000 characters and absorbs reading. The number of accepted suggestions is deliberately left out of the primary model: it is almost collinear with the letter's total span count, and including it makes the remaining coefficients unidentifiable.

### Aggregate — absolute seconds per event

n = 908 tasks, adjusted R² = 0.539.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `n_deleted` | 5595 | -8.1 | [-17.1, 0.8] | 0.0736 |
| `n_mech_edit` | 1464 | 0.8 | [-4.3, 5.8] | 0.768 |
| `n_cog_edit` | 339 | 55.4 ** | [18.1, 92.7] | 0.00361 |
| `C1` | 14478 | 13.6 *** | [7.6, 19.6] | 8.55e-06 |
| `chars_k` | — | 89.8 *** | [56.1, 123.6] | 1.81e-07 |

Intercept: 104 s of fixed cost per letter (opening the task, orienting in the text).

### Per class — absolute seconds per event

n = 908 tasks, adjusted R² = 0.546.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `M1` | 1041 | -15.5 | [-35.8, 4.7] | 0.133 |
| `M2` | 134 | 36.8 | [-32.7, 106.3] | 0.299 |
| `M3` ⚠︎ | 69 | -103.5 ** | [-175.1, -31.8] | 0.00464 |
| `M4` | 1261 | 5.2 | [-4.0, 14.5] | 0.269 |
| `C1` | 14478 | 14.0 *** | [8.2, 19.8] | 2.24e-06 |
| `C2` | 4554 | -7.9 | [-17.3, 1.5] | 0.0979 |
| `C3` | 274 | 26.7 | [-16.7, 70.1] | 0.227 |
| `C4` ⚠︎ | 65 | 170.7 *** | [79.1, 262.3] | 0.000261 |
| `chars_k` | — | 91.6 *** | [57.0, 126.1] | 2.04e-07 |

Intercept: 92 s of fixed cost per letter (opening the task, orienting in the text).

### Robustness — with accepted suggestions in the model

n = 908 tasks, adjusted R² = 0.542.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `A0` | 4245 | -11.1 * | [-21.5, -0.8] | 0.0351 |
| `n_deleted` | 5595 | -6.2 | [-15.8, 3.5] | 0.209 |
| `n_mech_edit` | 1464 | 0.2 | [-5.1, 5.5] | 0.94 |
| `n_cog_edit` | 339 | 55.6 ** | [17.4, 93.9] | 0.00432 |
| `C1` | 14478 | 14.9 *** | [8.9, 20.9] | 1.05e-06 |
| `chars_k` | — | 95.0 *** | [60.3, 129.7] | 7.85e-08 |

Intercept: 111 s of fixed cost per letter (opening the task, orienting in the text).

⚠︎ = fewer than 100 events in the whole corpus; the estimate is not usable.

### Collinearity

Variance inflation factors. Above ~5 the coefficient is poorly identified and its interval should be read as indicative only. Classes with few events (`M3`, `C4`) are additionally imprecise regardless of their VIF.

**Aggregate model** — `C1` 6.3, `chars_k` 4.7, `n_deleted` 2.4, `n_cog_edit` 1.5, `n_mech_edit` 1.4

**Per class model** — `C1` 6.4, `chars_k` 4.8, `M4` 4.0, `M1` 4.0, `C2` 2.7, `C3` 1.6, `M2` 1.2, `C4` 1.2, `M3` 1.1

## 4. Where the time actually went

Seconds per event from the aggregate model, multiplied by the number of events of that kind in the corrected corpus. Reading is the `chars_k` coefficient applied to the total character count, and the fixed cost is the intercept applied to every task.

| component | events | estimated minutes | share of measured time |
|---|---:|---:|---:|
| reading the letters | — | 3288 | 49% |
| fixed cost per task | — | 1572 | 23% |
| deleting spurious suggestions † | 5595 | -759 | -11% |
| mechanical edits † | 1464 | 19 | 0% |
| cognitive edits | 339 | 313 | 5% |
| drawing missed mentions by hand | 14478 | 3286 | 49% |
| annotator differences (fixed effects) | — | -979 | -15% |
| **measured total** | | **6740** | 100% |

† coefficient not significant at the 5 % level; the component is reported for completeness but is not distinguishable from zero. Components are model-fitted and sum to the measured total up to rounding, since OLS residuals sum to zero.

The two columns tell different stories on purpose. A cognitive edit is by far the most expensive single action, but there are few of them; mentions GLiNER2 missed are cheap one by one and overwhelming in bulk. Optimising a system for this corpus therefore means optimising recall, not precision.

## 5. Interpretation

`chars_k` is seconds per 1,000 characters and captures reading: on a letter of average length the annotator spends roughly three and a half minutes simply reading, before any correction. The event coefficients are what one correction of that class adds on top of that. They are marginal effects in an observational sample, not stopwatch measurements; the aggregate model is the one to quote and the per-class model is indicative.

## 6. Caveats

- `lead_time` is active-tab time. Label Studio pauses after ~60 s of idleness, but a tab left open on a difficult passage still accrues time.
- Costs are estimated on GLiNER2's error mix. Applying them to another system's event profile assumes an event of class *c* costs the same whichever system produced it — plausible for the mechanical classes, less so for `C1`, whose cost depends on how many mentions are missing at once.
- Annotators worked in group settings with the domain expert present, so some cognitive cost was paid by conversation and never entered `lead_time`. These estimates are therefore a lower bound on the cognitive classes.
- Letters annotated by several annotators appear once per annotator; standard errors are robust but not clustered on the letter.
