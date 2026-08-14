# Projected post-correction cost per system

Each system's event profile on the 561-letter gold corpus (12476 gold spans, 1256k characters), priced with the per-event costs measured on the annotators' real GLiNER2 post-correction round.

Measured inputs: a cognitive edit costs 51 s, a mention the system missed 11 s to draw by hand, a mechanical edit 1 s and a deletion -4 s (the last two are not distinguishable from zero); reading costs 88 s per 1,000 characters and each task carries 121 s of fixed cost.

## Estimated effort to supervise 100 letters

### Pricing: *measured*

Coefficients that are not distinguishable from zero (deletions, mechanical edits) are priced at zero; the two significant event coefficients are used as fitted.

| system | family | correction only (h) | 95% CI | total incl. reading (h) | vs. from scratch |
|---|---|---:|---|---:|---:|
| `claude-opus-4.8` | prompted LLM | 3.7 | [1.4, 6.0] | 10.2 | +25% |
| `gliner2-ft` | fine-tuned | 3.8 | [1.6, 6.0] | 10.3 | +24% |
| `gliner2` | zero-shot | 4.9 | [2.1, 7.7] | 11.3 | +16% |
| `dalembert` | fine-tuned | 5.4 | [0.8, 10.0] | 11.9 | +12% |
| `qwen3-235b-a22b-2507` | prompted LLM | 5.7 | [2.3, 9.0] | 12.1 | +10% |
| `qwen3-vl-30b-a3b-instruct` | prompted LLM | 5.8 | [2.3, 9.3] | 12.3 | +9% |
| `camembertav2` | fine-tuned | 6.7 | [0.7, 12.8] | 13.2 | +2% |
| **manual from scratch** | baseline | 7.1 | [2.2, 12.0] | 13.5 | — |

### Pricing: *conservative*

As *measured*, but cognitive edits are priced at the cheaper missed-mention rate, giving a lower bound for systems whose segmentation-error count is far outside the range observed in the gliner2 round.

| system | family | correction only (h) | 95% CI | total incl. reading (h) | vs. from scratch |
|---|---|---:|---|---:|---:|
| `dalembert` | fine-tuned | 2.0 | [0.6, 3.4] | 8.5 | +37% |
| `claude-opus-4.8` | prompted LLM | 2.3 | [0.7, 3.9] | 8.8 | +35% |
| `camembertav2` | fine-tuned | 2.4 | [0.7, 4.0] | 8.8 | +35% |
| `gliner2-ft` | fine-tuned | 2.6 | [0.8, 4.5] | 9.1 | +33% |
| `gliner2` | zero-shot | 4.0 | [1.2, 6.7] | 10.4 | +23% |
| `qwen3-235b-a22b-2507` | prompted LLM | 4.8 | [1.5, 8.2] | 11.3 | +17% |
| `qwen3-vl-30b-a3b-instruct` | prompted LLM | 5.1 | [1.6, 8.6] | 11.5 | +15% |
| **manual from scratch** | baseline | 7.1 | [2.2, 12.0] | 13.5 | — |

### Pricing: *unconstrained*

Every fitted coefficient used as-is, including the non-significant negative ones; shown to expose how much the ranking depends on them.

| system | family | correction only (h) | 95% CI | total incl. reading (h) | vs. from scratch |
|---|---|---:|---|---:|---:|
| `gliner2-ft` | fine-tuned | 3.3 | [1.4, 5.3] | 9.8 | +28% |
| `claude-opus-4.8` | prompted LLM | 3.5 | [1.3, 5.6] | 9.9 | +27% |
| `dalembert` | fine-tuned | 4.6 | [-0.2, 9.4] | 11.1 | +18% |
| `gliner2` | zero-shot | 4.7 | [2.0, 7.4] | 11.2 | +17% |
| `qwen3-235b-a22b-2507` | prompted LLM | 5.3 | [2.5, 8.2] | 11.8 | +13% |
| `camembertav2` | fine-tuned | 5.5 | [-1.0, 12.1] | 12.0 | +11% |
| `qwen3-vl-30b-a3b-instruct` | prompted LLM | 5.5 | [2.5, 8.6] | 12.0 | +11% |
| **manual from scratch** | baseline | 7.1 | [2.2, 12.0] | 13.5 | — |

Figures are per 100 letters of this corpus (scaled from 561 letters, factor 0.178). *Correction only* is the part attributable to the system; *total* adds the reading and per-task overhead that every scenario pays, including annotating from scratch. The from-scratch row prices each of the 12476 gold spans as one hand-drawn mention.

## What the ranking says

- Cheapest to supervise under the measured pricing: `claude-opus-4.8` (3.7 h of correction per 100 letters).
- Most expensive: `camembertav2` (6.7 h), 1.8× the cheapest.
- The ordering is **not** stable across pricings: it is dominated by how segmentation errors are priced, which is exactly the coefficient with the weakest empirical support. Only the claims below survive all three pricings.

### Claims that hold under every pricing

| system | measured (h) | conservative (h) | unconstrained (h) |
|---|---|---|---|
| `claude-opus-4.8` | 3.7 | 2.3 | 3.5 |
| `gliner2-ft` | 3.8 | 2.6 | 3.3 |
| `dalembert` | 5.4 | 2.0 | 4.6 |
| `gliner2` | 4.9 | 4.0 | 4.7 |
| `camembertav2` | 6.7 | 2.4 | 5.5 |
| `qwen3-235b-a22b-2507` | 5.7 | 4.8 | 5.3 |
| `qwen3-vl-30b-a3b-instruct` | 5.8 | 5.1 | 5.5 |

- Always among the three cheapest to supervise: `claude-opus-4.8`.
- Always among the three most expensive: `qwen3-vl-30b-a3b-instruct`.
- Every system beats annotating from scratch on total time under every pricing, but the spread between systems is comparable to the gap between the best system and no system at all, so which model is chosen matters about as much as whether one is used.

## Caveats

- The prices come from GLiNER2's error mix and are applied to other systems' event profiles. The load-bearing assumption is that an event of a given class costs the same whoever produced it.
- **Extrapolation warning.** The cognitive-edit price is estimated on 339 events in the GLiNER2 round, of which only 65 were segmentation errors. CamemBERTaV2 and d'AlemBERT generate 2,026 and 1,500 segmentation events respectively, far outside that support. The *conservative* pricing exists precisely to bound this: it re-prices cognitive edits at the much cheaper missed-mention rate.
- `C1` is the dominant term for most systems, and its per-event cost is the one most likely to be non-linear: an annotator who has to add ten mentions to one letter re-reads it once, not ten times. Systems with very high `C1` counts (both Qwen variants) are therefore likely *over*-priced here.
- Intervals reflect coefficient uncertainty only. They do not include uncertainty in the event counts themselves, nor the fold-to-fold variation reported in `error_taxonomy.md`, and imposed prices contribute no variance at all.
- Annotators worked alongside the domain expert, so part of the cognitive cost never entered `lead_time`; all figures are lower bounds on the cognitive classes.
