# NER post-correction error taxonomy

Scored on all 561 gold letters (12445 gold spans), pooled over the 5 cross-validation test folds (run 0).

Every event below is one action a corrector has to take. Events cover the whole strict error budget but are fewer than it, because a wrong boundary or a wrong label on the right characters costs one repair while strict scoring charges it twice (`events + aligned repairs == strict FN + strict FP`).

## 1. The taxonomy

| code | error class | detectability | locality | search scope | cost | repair |
|---|---|---|---|---|---|---|
| `F0` | malformed span | evident | in_span | bounded | **mechanical** | deleted by a validity rule (offsets / alphabetic content / word edges) |
| `M1` | redundant span | evident | in_span | bounded | **mechanical** | deduplicated by an overlap rule |
| `M2` | affix drift | evident | in_span | bounded | **mechanical** | normalised by a strip rule (whitespace, punctuation, article, honorific) |
| `M3` | token drift | semi | in_text | bounded | **mechanical** | one drag of the span handle in Label Studio |
| `M4` | decidable label flip | semi | in_text | bounded | **mechanical** | one keystroke; PER/PER_NOM recoverable from capitalisation |
| `C1` | missed mention | silent | external | unbounded | **cognitive** | re-read the letter (and the scan) to find what is not there |
| `C2` | spurious mention | silent | in_text | bounded | **cognitive** | read the context and adjudicate against the guidelines |
| `C3` | semantic label confusion | silent | external | bounded | **cognitive** | resolve metonymy / consult prosopography |
| `C4` | mention segmentation | silent | external | bounded | **cognitive** | decide who or what the mention denotes, then re-cut the span |

*Detectability*: `evident` = demonstrably wrong without reading the letter; `semi` = visible, but the surrounding sentence is needed; `silent` = the interface shows nothing that signals a problem.  *Locality*: `in_span` = repairable from the span string; `in_text` = from the letter; `external` = requires the scan, the guidelines or prosopographical reference works.  *Search scope*: `bounded` = an on-screen span triggers the action, so the work is proportional to the number of predictions; `unbounded` = nothing triggers the action, so the whole letter has to be re-read.

## 2. Post-correction burden per system

| system | family | events | mechanical | cognitive | % cognitive | mech./letter | cogn./letter | unbounded/letter | cognitive load ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gliner2` | zero-shot | 10176 | 2142 | 8034 | 79% | 3.82 | 14.32 | 11.59 | 0.65 |
| `gliner2-ft` | fine-tuned | 8841 | 1677 | 7164 | 81% | 2.99 | 12.77 | 7.21 | 0.58 |
| `dalembert` | fine-tuned | 9778 | 2476 | 7302 | 75% | 4.41 | 13.02 | 3.30 | 0.59 |
| `camembertav2` | fine-tuned | 12848 | 3694 | 9154 | 71% | 6.58 | 16.32 | 3.36 | 0.74 |
| `claude-opus-4.8` | prompted LLM | 7488 | 2147 | 5341 | 71% | 3.83 | 9.52 | 5.89 | 0.43 |
| `qwen3-235b-a22b-2507` | prompted LLM | 11424 | 1130 | 10294 | 90% | 2.01 | 18.35 | 14.45 | 0.83 |
| `qwen3-vl-30b-a3b-instruct` | prompted LLM | 11304 | 853 | 10451 | 92% | 1.52 | 18.63 | 15.29 | 0.84 |

*unbounded/letter* is the subset of the cognitive events that no on-screen artefact points at, i.e. the mentions the corrector can only recover by re-reading the letter.

The *cognitive load ratio* normalises the cognitive events by the 12445 gold spans (22.2 per letter) that a from-scratch pass would have to create by hand. It is a rough proxy for the share of the manual burden that survives post-correction; it is not bounded by 1, because a system can also add spurious mentions the annotator never had to consider.

### Fold variance of the per-letter rates

Mean over the 5 test folds ± half-width of a 95% normal-approximation CI (same convention as the F1 tables; with k = 5 the interval is wide and indicates fold sensitivity rather than a population estimate).

| system | mechanical / letter | cognitive / letter | unbounded / letter |
|---|---|---|---|
| `gliner2` | 3.82 ± 0.29 | 14.32 ± 1.15 | 11.59 ± 0.76 |
| `gliner2-ft` | 2.99 ± 0.66 | 12.77 ± 1.99 | 7.22 ± 2.86 |
| `dalembert` | 4.41 ± 0.49 | 13.01 ± 1.73 | 3.30 ± 0.56 |
| `camembertav2` | 6.58 ± 0.60 | 16.32 ± 2.50 | 3.36 ± 0.38 |
| `claude-opus-4.8` | 3.83 ± 0.39 | 9.52 ± 1.10 | 5.89 ± 0.80 |
| `qwen3-235b-a22b-2507` | 2.01 ± 0.21 | 18.35 ± 1.43 | 14.45 ± 1.33 |
| `qwen3-vl-30b-a3b-instruct` | 1.52 ± 0.22 | 18.63 ± 1.54 | 15.29 ± 1.52 |

## 3. Event composition per system

Classes with no events in this run are omitted from the two tables below: `F0` (malformed span). `F0` is not a taxonomy class: it is what the validity filter deletes, so it is empty by construction whenever that filter has run over the predictions being scored.

| system | `M1` | `M2` | `M3` | `M4` | `C1` | `C2` | `C3` | `C4` |
|---|---|---|---|---|---|---|---|---|
| `gliner2` | 2 | 306 | 140 | 1694 | 6500 | 1061 | 321 | 152 |
| `gliner2-ft` | 123 | 1222 | 295 | 37 | 4047 | 2514 | 288 | 315 |
| `dalembert` | 553 | 1416 | 459 | 48 | 1854 | 3708 | 444 | 1296 |
| `camembertav2` | 1517 | 1218 | 910 | 49 | 1884 | 5001 | 588 | 1681 |
| `claude-opus-4.8` | 199 | 1581 | 344 | 23 | 3303 | 1302 | 368 | 368 |
| `qwen3-235b-a22b-2507` | 99 | 742 | 222 | 67 | 8109 | 1761 | 212 | 212 |
| `qwen3-vl-30b-a3b-instruct` | 111 | 538 | 151 | 53 | 8577 | 1490 | 202 | 182 |

### Share of each system's events (row-normalised)

| system | `M1` | `M2` | `M3` | `M4` | `C1` | `C2` | `C3` | `C4` |
|---|---|---|---|---|---|---|---|---|
| `gliner2` | 0% | 3% | 1% | 17% | 64% | 10% | 3% | 1% |
| `gliner2-ft` | 1% | 14% | 3% | 0% | 46% | 28% | 3% | 4% |
| `dalembert` | 6% | 14% | 5% | 0% | 19% | 38% | 5% | 13% |
| `camembertav2` | 12% | 9% | 7% | 0% | 15% | 39% | 5% | 13% |
| `claude-opus-4.8` | 3% | 21% | 5% | 0% | 44% | 17% | 5% | 5% |
| `qwen3-235b-a22b-2507` | 1% | 6% | 2% | 1% | 71% | 15% | 2% | 2% |
| `qwen3-vl-30b-a3b-instruct` | 1% | 5% | 1% | 0% | 76% | 13% | 2% | 2% |

### Boundary repairs hidden inside label events

Charging one event per aligned pair is what makes `events + aligned repairs == strict FN + strict FP` hold, and it has a cost: a prediction wrong in both its label and its boundary is charged to the label class (`M4` or `C3`) and the boundary repair rides along uncounted. The columns below give the class that repair would have fallen into, so the table bounds how much this understates the burden.

| system | label events | hidden repair | share | `M2` | `M3` | `C4` | % of all events |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gliner2` | 2015 | 506 | 25% | 376 | 73 | 57 | 5% |
| `gliner2-ft` | 325 | 134 | 41% | 52 | 68 | 14 | 2% |
| `dalembert` | 492 | 242 | 49% | 92 | 113 | 37 | 2% |
| `camembertav2` | 637 | 353 | 55% | 82 | 193 | 78 | 3% |
| `claude-opus-4.8` | 391 | 208 | 53% | 75 | 105 | 28 | 3% |
| `qwen3-235b-a22b-2507` | 279 | 152 | 54% | 49 | 72 | 31 | 1% |
| `qwen3-vl-30b-a3b-instruct` | 255 | 156 | 61% | 56 | 69 | 31 | 1% |
| **all systems** | **4394** | **1751** | **40%** | **782** | **693** | **276** | — |

84% of the hidden repairs would themselves be mechanical, so the bias understates mechanical labour and is conservative with respect to the mechanical/cognitive split reported above.

## 4. Cognitive events on the identity-critical labels (PER, LOC)

These are the events that propagate into the entity-clustering step.

| system | missed PER/LOC | spurious PER/LOC | segmentation | semantic label | total | per letter |
|---|---:|---:|---:|---:|---:|---:|
| `gliner2` | 2623 | 120 | 76 | 211 | 3030 | 5.40 |
| `gliner2-ft` | 1751 | 558 | 142 | 86 | 2537 | 4.52 |
| `dalembert` | 409 | 884 | 467 | 193 | 1953 | 3.48 |
| `camembertav2` | 497 | 1351 | 692 | 281 | 2821 | 5.03 |
| `claude-opus-4.8` | 670 | 376 | 173 | 84 | 1303 | 2.32 |
| `qwen3-235b-a22b-2507` | 4071 | 224 | 98 | 77 | 4470 | 7.97 |
| `qwen3-vl-30b-a3b-instruct` | 4099 | 273 | 91 | 78 | 4541 | 8.09 |

## 5. Sub-types

### `C1` — missed mention

| system | `other labels` | `identity-critical (PER/LOC)` |
|---|---|---|
| `gliner2` | 3877 | 2623 |
| `gliner2-ft` | 2296 | 1751 |
| `dalembert` | 1445 | 409 |
| `camembertav2` | 1387 | 497 |
| `claude-opus-4.8` | 2633 | 670 |
| `qwen3-235b-a22b-2507` | 4038 | 4071 |
| `qwen3-vl-30b-a3b-instruct` | 4478 | 4099 |

### `C4` — mention segmentation

| system | `split` | `merge` | `identity-altering boundary` |
|---|---|---|---|
| `gliner2` | 41 | 71 | 40 |
| `gliner2-ft` | 146 | 88 | 81 |
| `dalembert` | 1095 | 97 | 104 |
| `camembertav2` | 1312 | 143 | 226 |
| `claude-opus-4.8` | 101 | 169 | 98 |
| `qwen3-235b-a22b-2507` | 77 | 75 | 60 |
| `qwen3-vl-30b-a3b-instruct` | 84 | 47 | 51 |

### `C3` — semantic label confusion

| system | `PER->PER_NOM` | `PER->LOC` | `LOC->ORG` | `ART->PER` | `LOC->PER` | `PER->ART` | `PER_NOM->PER` | `PER_NOM->ART` |
|---|---|---|---|---|---|---|---|---|
| `gliner2` | 2 | 2 | 4 | 38 | 11 | 0 | 64 | 39 |
| `gliner2-ft` | 45 | 27 | 6 | 3 | 23 | 38 | 5 | 12 |
| `dalembert` | 39 | 42 | 23 | 25 | 43 | 33 | 16 | 15 |
| `camembertav2` | 41 | 105 | 22 | 36 | 59 | 30 | 5 | 29 |
| `claude-opus-4.8` | 44 | 6 | 82 | 30 | 8 | 21 | 11 | 12 |
| `qwen3-235b-a22b-2507` | 29 | 3 | 24 | 22 | 7 | 9 | 19 | 8 |
| `qwen3-vl-30b-a3b-instruct` | 27 | 4 | 20 | 19 | 20 | 18 | 21 | 8 |

### `M4` — decidable label flip

| system | `PER_NOM->PER` | `PER->PER_NOM` |
|---|---|---|
| `gliner2` | 1686 | 8 |
| `gliner2-ft` | 12 | 25 |
| `dalembert` | 31 | 17 |
| `camembertav2` | 30 | 19 |
| `claude-opus-4.8` | 13 | 10 |
| `qwen3-235b-a22b-2507` | 48 | 19 |
| `qwen3-vl-30b-a3b-instruct` | 50 | 3 |

## 6. Worked examples

### `M1` redundant span — mechanical (evident / in_span / bounded)

2604 events across all systems. Repair: deduplicated by an overlap rule.

- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/2/4` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «Navez» `PER`
  - context: Paris ce 5 Septembre 1844. Mon cher Monsieur **[Navez]** mon ancien et excellent camarade Je profite de l'occas…
- `dalembert` · letter `Ms. II 70/1/62` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «Péri» `PER`
  - context: … les pays et à toutes les époques, depuis le siècle de **[Péri]**clès jusqu'à celui de Louis XIII et de Louis XIV.  Il s…
- `camembertav2` · letter `Ms. II 70/1/177` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «Ange» `PER`
  - context: …tine! En présence de l’œuvre de ce grand homme Michel, **[Ange]** on se trouve si petit, qu’on a presque honte de peindr…

### `M2` affix drift — mechanical (evident / in_span / bounded)

7023 events across all systems. Repair: normalised by a strip rule (whitespace, punctuation, article, honorific).

- `claude-opus-4.8` · letter `Ms. II 70/4/37`
  - gold: «Facos» `PER`
  - predicted: «Facos[?]» `PER`
  - context: … existent et quelques sculptures quelques en marbre de **[Facos[?]]**, vous voyez que c'est un petit Vatican; dans la second…
- `dalembert` · letter `Ms. II 70/6/sup./110`
  - gold: «Navez» `PER`
  - predicted: «Monsieur Navez» `PER`
  - context: …nde de l'ami Navez  E. Prévinaire  [Address:] Monsieur **[Monsieur Navez]** rue Royale neuve à Bruxelles.
- `gliner2-ft` · letter `Ms. II 70/4/37`
  - gold: «Gérard » `PER`
  - predicted: «Gérard» `PER`
  - context: …n de sérieux. nous verrons sortir quelque jour de chez **[Gérard]** un jugement de Paris, je suis sur que ce sera le penda…

### `M3` token drift — mechanical (semi / in_text / bounded)

2521 events across all systems. Repair: one drag of the span handle in Label Studio.

- `qwen3-235b-a22b-2507` · letter `Ms. II 70/2/172`
  - gold: «P. et Dedreux» `PER`
  - predicted: «Dedreux» `PER`
  - context: …enu .  Adieu cher ami et camarade —  tout à toi  P. et **[Dedreux]**  [Address:] A Monsieur Mr. Navez Peintre chez Mr. Dehe…
- `camembertav2` · letter `Ms. II 70/1/6`
  - gold: «monsieur⏎Navez» `PER`
  - predicted: «onsieur⏎Navez» `PER`
  - context: … la promesse qui m'est faite.  Encore une fois, merci m**[onsieur Navez]**, et croyez moi bien, je vous prie, votre très dévoué s…
- `claude-opus-4.8` · letter `Ms. II 70/1/183`
  - gold: «J. Sturm» `PER`
  - predicted: «notre malheureux ami J. Sturm» `PER`
  - context: …de plaisir qu’on va ériger un monument à la mémoire de **[notre malheureux ami J. Sturm]**. de notre côté nous lui faisons élever un souvenir à l…

### `M4` decidable label flip — mechanical (semi / in_text / bounded)

1971 events across all systems. Repair: one keystroke; PER/PER_NOM recoverable from capitalisation.

- `gliner2` · letter `Ms. II 70/2/55` · *PER_NOM->PER*
  - gold: «Jacquet » `PER`
  - predicted: «Jacquet» `PER_NOM`
  - context: …le fait que deux de plus je les ai toutes  prises chez **[Jacquet]** là au moins l'on sait que l’on a  du bon il ne mettra …
- `gliner2` · letter `Ms. II 70/2/50` · *PER_NOM->PER*
  - gold: «Robert» `PER`
  - predicted: «Robert» `PER_NOM`
  - context: …gulier personnage vous m'adressez  un petit tableau de **[Robert]** qui a  déjà été exposé à la société des amis  des Grec…
- `gliner2` · letter `Ms. II 70/4/61` · *PER_NOM->PER*
  - gold: «Monsieur Van Regemorter» `PER`
  - predicted: «Monsieur Van Regemorter» `PER_NOM`
  - context: Monsieur,  C'est par l'obligeance de **[Monsieur Van Regemorter]** que nous avons quelques rapports indirects avec Vous, …

### `C1` missed mention — cognitive (silent / external / unbounded)

34274 events across all systems. Repair: re-read the letter (and the scan) to find what is not there.

- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/2/27` · *identity-critical (PER/LOC)*
  - gold: «monsieur Navez» `PER`
  - predicted: — *(no prediction)*
  - context: …ls me seront toujours agréables  [Address:] à monsieur **[monsieur Navez]** chez Monsieur Dehollain peintre rue des Sœurs à Bruxel…
- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/1/182` · *identity-critical (PER/LOC)*
  - gold: «le champ⏎de Booz» `LOC`
  - predicted: — *(no prediction)*
  - context: …depuis hier le pendant sera Ruth lorsq’ ci elle quitte **[le champ de Booz]** emportant sur les épaules le graine qu’elle vient de r…
- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/3/536` · *identity-critical (PER/LOC)*
  - gold: «Navez» `PER`
  - predicted: — *(no prediction)*
  - context: …e prendre pour antique le Corrège ou le guide Mon cher **[Navez]** cette lettre était comme depuis quelques jours j'étais…

### `C2` spurious mention — cognitive (silent / in_text / bounded)

16837 events across all systems. Repair: read the context and adjudicate against the guidelines.

- `gliner2-ft` · letter `Ms. II 70/2/312bis` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «geure» `LOC`
  - context: …de très bonnes choses mais rien de grand ni d'élevé du **[geure]**  Quant a moi pour la gloire artistique de la Belgique …
- `dalembert` · letter `Ms. II 70/5/401` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «Bordereau» `PER`
  - context: …iteur Meeus vandermaelen  Bruxelles le 6 octbre 1894.  **[Bordereau]** un billet de banque No. 1131 — flo. 250. un do. d ——— …
- `camembertav2` · letter `Ms. II 70/1/414` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «Bus» `PER`
  - context: … [...] de Rome]  [PS: Mes Complement a Mor le Baron du **[Bus]** a Mons le Chent Rettelot et sa femme ]  [Address:] ITA…

### `C3` semantic label confusion — cognitive (silent / external / bounded)

2423 events across all systems. Repair: resolve metonymy / consult prosopography.

- `camembertav2` · letter `Ms. II 70/3/436` · *PER->LOC*
  - gold: «Carrare» `LOC`
  - predicted: «Carrare» `PER`
  - context: …de te faire ses salutations empressées. Tenerani est a **[Carrare]**. Terling a Terni. Je n'ai pas vu encore toutes les per…
- `camembertav2` · letter `Ms. II 70/2/113` · *EVT->LOC*
  - gold: «grand⏎salon carré» `LOC`
  - predicted: «grand⏎salon» `EVT`
  - context: …e portrait fait un fort bon effet il est placé dans le **[grand salon]** carré il m'a paru faire plaisir il est surtout du très…
- `claude-opus-4.8` · letter `Ms. II 70/3/620` · *PER_NOM->LOC*
  - gold: «mon frère» `LOC`
  - predicted: «mon frère» `PER_NOM`
  - context: …e ton dévoué de cœur  [Margin bottom Page 1:] Desplace **[mon frère]** te fait mille amitiés quand on fait te dire je ne sais…

### `C4` mention segmentation — cognitive (silent / external / bounded)

4206 events across all systems. Repair: decide who or what the mention denotes, then re-cut the span.

- `camembertav2` · letter `Ms. II 70/1/388` · *identity-altering boundary*
  - gold: «Snets» `PER`
  - predicted: «Mons Snets» `PER`
  - context: …armis les fransais nous avons les Vernets pere et fils **[Mons Snets]** votre bon Ami ; et Mons Granet tous des artistes du pr…
- `camembertav2` · letter `Ms. II 70/2/172` · *identity-altering boundary*
  - gold: «P. et Dedreux» `PER`
  - predicted: «dreux» `PER`
  - context: …u .  Adieu cher ami et camarade —  tout à toi  P. et De**[dreux]**  [Address:] A Monsieur Mr. Navez Peintre chez Mr. Dehe…
- `gliner2-ft` · letter `Ms. II 70/2/59` · *identity-altering boundary*
  - gold: «Mme. de Chateaubrian» `PER`
  - predicted: «Mme. de Chateaubriand» `PER`
  - context: …re disposition.  vous saivez que le conclave a donné à **[Mme. de Chateaubriand]**  le chat du dernier pape heureux animal te  voila l'ob…
