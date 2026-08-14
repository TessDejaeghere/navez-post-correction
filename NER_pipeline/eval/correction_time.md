# Measured post-correction time per error class

Estimated from the annotators' own edit log: every NER letter was pre-annotated with GLiNER2 and corrected in Label Studio, which records `lead_time` (active-tab seconds) and, for every span, whether it was accepted, edited or created by hand.

## 1. Sample

- 651 submitted tasks from 4 annotators over 554 distinct letters.
- 519 of those carry the GLiNER2 suggestion object and can be diffed span by span (132 cannot: the export for one annotator dropped the prediction payload).
- Time on task: mean 7.1 min, median 4.2 min, p90 16.6 min.
- Total recorded effort: 77.6 hours.

| annotator | tasks | median min | mean min | accepted | edited | manual |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 124 | 6.4 | 9.0 | 749 | 556 | 1838 |
| A2 | 94 | 6.4 | 9.6 | 441 | 173 | 1766 |
| A3 | 189 | 3.7 | 6.8 | 707 | 761 | 2379 |
| A4 | 244 | 3.5 | 5.5 | 1173 | 50 | 3864 |

Annotator style differs sharply and is absorbed by fixed effects in every model below: some annotators edit a suggestion, others delete it and re-draw the span by hand, which moves the same correction from the `prediction-changed` column to the `manual` one.

## 2. What GLiNER2's output cost, by class

Excluded from this analysis: `F0` (malformed span) — 2 events. `F0` is not a taxonomy class: it is what the validity filter deletes before a corrector sees it, so it is not part of the cost model.

| class | error class | cost | events | % of spans touched |
|---|---|---|---:|---:|
| `A0` | accepted suggestion | (none) | 2343 | 16.2% |
| `M1` | redundant span | mechanical | 627 | 4.3% |
| `M2` | affix drift | mechanical | 65 | 0.5% |
| `M3` | token drift | mechanical | 28 | 0.2% |
| `M4` | decidable label flip | mechanical | 716 | 5.0% |
| `C1` | missed mention | cognitive | 7911 | 54.9% |
| `C2` | spurious mention | cognitive | 2576 | 17.9% |
| `C3` | semantic label confusion | cognitive | 128 | 0.9% |
| `C4` | mention segmentation | cognitive | 25 | 0.2% |

## 3. How long does each kind of correction take?

`minutes ~ event counts + letter length + annotator`, OLS with HC1 robust standard errors, except where the heading says otherwise. Coefficients are absolute seconds per event; `chars_k` is seconds per 1,000 characters and absorbs reading. The number of accepted suggestions is deliberately left out of the primary model: it is almost collinear with the letter's total span count, and including it makes the remaining coefficients unidentifiable.

### Aggregate — absolute seconds per event

n = 519 tasks, adjusted R² = 0.472.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `n_deleted` | 3203 | -4.0 | [-14.4, 6.5] | 0.457 |
| `n_mech_edit` | 809 | 0.7 | [-6.2, 7.6] | 0.845 |
| `n_cog_edit` | 153 | 50.5 | [-2.8, 103.9] | 0.0635 |
| `C1` | 7911 | 11.4 ** | [3.5, 19.4] | 0.00471 |
| `chars_k` | — | 87.6 *** | [43.8, 131.4] | 8.96e-05 |

Intercept: 121 s of fixed cost per letter (opening the task, orienting in the text).

### Per class — absolute seconds per event

n = 519 tasks, adjusted R² = 0.469.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `M1` | 627 | -6.1 | [-36.7, 24.5] | 0.697 |
| `M2` ⚠︎ | 65 | 52.7 | [-55.8, 161.3] | 0.341 |
| `M4` | 716 | 2.4 | [-11.2, 16.0] | 0.726 |
| `C1` | 7911 | 11.9 ** | [3.7, 20.1] | 0.00469 |
| `C2` | 2576 | -4.0 | [-13.7, 5.8] | 0.425 |
| `C3` | 128 | 35.5 | [-23.5, 94.6] | 0.238 |
| `chars_k` | — | 85.3 *** | [40.3, 130.4] | 0.000207 |

Intercept: 122 s of fixed cost per letter (opening the task, orienting in the text).

### Robustness — with accepted suggestions in the model

n = 519 tasks, adjusted R² = 0.476.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `A0` | 2343 | -13.4 | [-27.1, 0.3] | 0.0554 |
| `n_deleted` | 3203 | -1.7 | [-13.2, 9.7] | 0.767 |
| `n_mech_edit` | 809 | -0.0 | [-7.2, 7.1] | 0.993 |
| `n_cog_edit` | 153 | 50.1 | [-4.7, 104.9] | 0.073 |
| `C1` | 7911 | 13.0 ** | [5.2, 20.8] | 0.00106 |
| `chars_k` | — | 94.8 *** | [49.2, 140.4] | 4.58e-05 |

Intercept: 128 s of fixed cost per letter (opening the task, orienting in the text).

### Robustness — standard errors clustered by letter (457 clusters, 33 annotated more than once)

n = 519 tasks, adjusted R² = 0.472.

| term | events | seconds per event | 95% CI | p |
|---|---:|---:|---|---:|
| `n_deleted` | 3203 | -4.0 | [-14.3, 6.4] | 0.452 |
| `n_mech_edit` | 809 | 0.7 | [-5.9, 7.3] | 0.838 |
| `n_cog_edit` | 153 | 50.5 | [-0.7, 101.7] | 0.0531 |
| `C1` | 7911 | 11.4 ** | [3.6, 19.3] | 0.00439 |
| `chars_k` | — | 87.6 *** | [47.6, 127.6] | 1.77e-05 |

Intercept: 121 s of fixed cost per letter (opening the task, orienting in the text).

⚠︎ = fewer than 100 events in the whole corpus; the estimate is not usable.

### Collinearity

Variance inflation factors. Above ~5 the coefficient is poorly identified and its interval should be read as indicative only. Classes with few events (`M2`, `M3`, `C4`) are additionally imprecise regardless of their VIF.

**Aggregate model** — `C1` 6.1, `chars_k` 4.4, `n_deleted` 2.4, `n_cog_edit` 1.4, `n_mech_edit` 1.4

**Per class model** — `C1` 6.1, `chars_k` 4.6, `M1` 3.6, `M4` 3.6, `C2` 2.6, `C3` 1.5, `M2` 1.1

## 4. Where the time actually went

Seconds per event from the aggregate model, multiplied by the number of events of that kind in the corrected corpus. Reading is the `chars_k` coefficient applied to the total character count, and the fixed cost is the intercept applied to every task.

| component | events | estimated minutes | share of measured time |
|---|---:|---:|---:|
| reading the letters | — | 1761 | 50% |
| fixed cost per task | — | 1046 | 30% |
| deleting spurious suggestions † | 3203 | -211 | -6% |
| mechanical edits † | 809 | 9 | 0% |
| cognitive edits † | 153 | 129 | 4% |
| drawing missed mentions by hand | 7911 | 1508 | 43% |
| annotator differences (fixed effects) | — | -730 | -21% |
| **measured total** | | **3512** | 100% |

† coefficient not significant at the 5 % level; the component is reported for completeness but is not distinguishable from zero. Components are model-fitted and sum to the measured total up to rounding, since OLS residuals sum to zero.

The two columns tell different stories on purpose. A cognitive edit is by far the most expensive single action, but there are few of them; mentions GLiNER2 missed are cheap one by one and overwhelming in bulk. Optimising a system for this corpus therefore means optimising recall, not precision.

## 5. Interpretation

`chars_k` is seconds per 1,000 characters and captures reading: on a letter of average length the annotator spends roughly three and a half minutes simply reading, before any correction. The event coefficients are what one correction of that class adds on top of that. They are marginal effects in an observational sample, not stopwatch measurements; the aggregate model is the one to quote and the per-class model is indicative.

## 6. Caveats

- `lead_time` is active-tab time. Label Studio pauses after ~60 s of idleness, but a tab left open on a difficult passage still accrues time.
- Costs are estimated on GLiNER2's error mix. Applying them to another system's event profile assumes an event of class *c* costs the same whichever system produced it — plausible for the mechanical classes, less so for `C1`, whose cost depends on how many mentions are missing at once.
- Annotators worked in group settings with the domain expert present, so some cognitive cost was paid by conversation and never entered `lead_time`. These estimates are therefore a lower bound on the cognitive classes.
- Letters annotated by several annotators appear once per annotator; standard errors are robust but not clustered on the letter.
