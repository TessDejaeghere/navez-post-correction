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
| `gliner2` | zero-shot | 10178 | 2144 | 8034 | 79% | 3.82 | 14.32 | 11.59 | 0.65 |
| `gliner2-ft` | fine-tuned | 8844 | 1680 | 7164 | 81% | 2.99 | 12.77 | 7.21 | 0.58 |
| `dalembert` | fine-tuned | 10075 | 2773 | 7302 | 72% | 4.94 | 13.02 | 3.30 | 0.59 |
| `camembertav2` | fine-tuned | 14595 | 5439 | 9156 | 63% | 9.70 | 16.32 | 3.36 | 0.74 |
| `claude-opus-4.8` | prompted LLM | 7491 | 2150 | 5341 | 71% | 3.83 | 9.52 | 5.89 | 0.43 |
| `qwen3-235b-a22b-2507` | prompted LLM | 11446 | 1152 | 10294 | 90% | 2.05 | 18.35 | 14.45 | 0.83 |
| `qwen3-vl-30b-a3b-instruct` | prompted LLM | 11368 | 917 | 10451 | 92% | 1.63 | 18.63 | 15.29 | 0.84 |

*unbounded/letter* is the subset of the cognitive events that no on-screen artefact points at, i.e. the mentions the corrector can only recover by re-reading the letter.

The *cognitive load ratio* normalises the cognitive events by the 12445 gold spans (22.2 per letter) that a from-scratch pass would have to create by hand. It is a rough proxy for the share of the manual burden that survives post-correction; it is not bounded by 1, because a system can also add spurious mentions the annotator never had to consider.

### Fold variance of the per-letter rates

Mean over the 5 test folds ± half-width of a 95% normal-approximation CI (same convention as the F1 tables; with k = 5 the interval is wide and indicates fold sensitivity rather than a population estimate).

| system | mechanical / letter | cognitive / letter | unbounded / letter |
|---|---|---|---|
| `gliner2` | 3.82 ± 0.29 | 14.32 ± 1.15 | 11.59 ± 0.76 |
| `gliner2-ft` | 2.99 ± 0.66 | 12.77 ± 1.99 | 7.22 ± 2.86 |
| `dalembert` | 4.94 ± 0.61 | 13.01 ± 1.73 | 3.30 ± 0.56 |
| `camembertav2` | 9.69 ± 0.99 | 16.32 ± 2.50 | 3.36 ± 0.38 |
| `claude-opus-4.8` | 3.83 ± 0.40 | 9.52 ± 1.10 | 5.89 ± 0.80 |
| `qwen3-235b-a22b-2507` | 2.05 ± 0.22 | 18.35 ± 1.43 | 14.45 ± 1.33 |
| `qwen3-vl-30b-a3b-instruct` | 1.63 ± 0.19 | 18.63 ± 1.54 | 15.29 ± 1.52 |

## 3. Event composition per system

| system | `F0` | `M1` | `M2` | `M3` | `M4` | `C1` | `C2` | `C3` | `C4` |
|---|---|---|---|---|---|---|---|---|---|
| `gliner2` | 2 | 2 | 306 | 140 | 1694 | 6500 | 1061 | 321 | 152 |
| `gliner2-ft` | 3 | 123 | 1222 | 295 | 37 | 4047 | 2514 | 288 | 315 |
| `dalembert` | 297 | 553 | 1416 | 459 | 48 | 1854 | 3708 | 444 | 1296 |
| `camembertav2` | 1745 | 1517 | 1218 | 910 | 49 | 1884 | 5003 | 588 | 1681 |
| `claude-opus-4.8` | 3 | 199 | 1581 | 344 | 23 | 3303 | 1302 | 368 | 368 |
| `qwen3-235b-a22b-2507` | 22 | 99 | 742 | 222 | 67 | 8109 | 1761 | 212 | 212 |
| `qwen3-vl-30b-a3b-instruct` | 64 | 111 | 538 | 151 | 53 | 8577 | 1490 | 202 | 182 |

### Share of each system's events (row-normalised)

| system | `F0` | `M1` | `M2` | `M3` | `M4` | `C1` | `C2` | `C3` | `C4` |
|---|---|---|---|---|---|---|---|---|---|
| `gliner2` | 0% | 0% | 3% | 1% | 17% | 64% | 10% | 3% | 1% |
| `gliner2-ft` | 0% | 1% | 14% | 3% | 0% | 46% | 28% | 3% | 4% |
| `dalembert` | 3% | 5% | 14% | 5% | 0% | 18% | 37% | 4% | 13% |
| `camembertav2` | 12% | 10% | 8% | 6% | 0% | 13% | 34% | 4% | 12% |
| `claude-opus-4.8` | 0% | 3% | 21% | 5% | 0% | 44% | 17% | 5% | 5% |
| `qwen3-235b-a22b-2507` | 0% | 1% | 6% | 2% | 1% | 71% | 15% | 2% | 2% |
| `qwen3-vl-30b-a3b-instruct` | 1% | 1% | 5% | 1% | 0% | 75% | 13% | 2% | 2% |

### Boundary repairs hidden inside label events

Charging one event per aligned pair is what makes `events + aligned repairs == strict FN + strict FP` hold, and it has a cost: a prediction wrong in both its label and its boundary is charged to the label class (`M4` or `C3`) and the boundary repair rides along uncounted. The columns below give the class that repair would have fallen into, so the table bounds how much this understates the burden.

| system | label events | hidden repair | share | `M2` | `M3` | `C4` | % of all events |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gliner2` | 2015 | 506 | 25% | 376 | 73 | 57 | 5% |
| `gliner2-ft` | 325 | 134 | 41% | 52 | 68 | 14 | 2% |
| `dalembert` | 492 | 242 | 49% | 92 | 113 | 37 | 2% |
| `camembertav2` | 637 | 353 | 55% | 82 | 193 | 78 | 2% |
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
| `camembertav2` | 497 | 1352 | 692 | 281 | 2822 | 5.03 |
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

### `F0` malformed span — mechanical (evident / in_span / bounded)

2136 events across all systems. Repair: deleted by a validity rule (offsets / alphabetic content / word edges).

- `camembertav2` · letter `Ms. II 70/5/632` · *tiny_midword*
  - gold: — *(nothing in the gold)*
  - predicted: «IE» `LOC`
  - context: …52]  [Margin: PARIS 21 JANV 52]  [Margin: ENTREE PAR QU**[IE]**VRAIN 22 JANV.] 
- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/5/406` · *empty_content*
  - gold: — *(nothing in the gold)*
  - predicted: «1858» `LOC`
  - context: …, car je suis très enrhumé  [...]  [Margin: Mell. Maes **[1858]**.]
- `camembertav2` · letter `Ms. II 70/2/105` · *tiny_midword*
  - gold: — *(nothing in the gold)*
  - predicted: «u» `PER`
  - context: …ient posément maintenant comme il convient dans l'age m**[u]**r Raphael dans la dispute du saint sairement a-t-il don…

### `M1` redundant span — mechanical (evident / in_span / bounded)

2604 events across all systems. Repair: deduplicated by an overlap rule.

- `claude-opus-4.8` · letter `Ms. II 70/2/26` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «l'irascible Dupavillon» `PER`
  - context: …ir égard à son grand âge que sa tête s'affoiblit [...] **[l'irascible Dupavillon]** prétend lui qu'il a encore toute sa tête et qu'il n'es…
- `camembertav2` · letter `Ms. II 70/2/28` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «Romee» `LOC`
  - context: …rgeait, à onze heures du soir ils rentrèrent tous deux **[Romee]** (c'était à Paques qu'il s'était donné ce beau repas) b…
- `camembertav2` · letter `Ms. II 70/2/69` · *overlapping same-label span*
  - gold: — *(nothing in the gold)*
  - predicted: «né» `PER`
  - context: … temps que vous n'avez eu des nouvelles des Robert l'ai**[né]** est à Venise il veut y faire le carnaval de même grand…

### `M2` affix drift — mechanical (evident / in_span / bounded)

7023 events across all systems. Repair: normalised by a strip rule (whitespace, punctuation, article, honorific).

- `dalembert` · letter `Ms. II 70/2/103`
  - gold: «Rénéry » `PER`
  - predicted: «Rénéry» `PER`
  - context: …z-vous d’avoir vu à l’atelier un certain Italien nommé **[Rénéry]** qui donna à Poisson et à moi des leçons de perspective…
- `camembertav2` · letter `Ms. II 70/2/107`
  - gold: «Rome » `LOC`
  - predicted: «Rome» `LOC`
  - context: …s figures valent mieux que celles de la mere abbesse à **[Rome]** ou du baptême du Duc de Chartres dans ces deux tableau…
- `camembertav2` · letter `Ms. II 70/1/201`
  - gold: «Monsieur⏎M F Navez» `PER`
  - predicted: «M F Navez» `PER`
  - context: Monsieur **[M F Navez]** Directeur de l’academie royale des beaux arts, rue roy…

### `M3` token drift — mechanical (semi / in_text / bounded)

2521 events across all systems. Repair: one drag of the span handle in Label Studio.

- `claude-opus-4.8` · letter `Ms. II 70/4/37`
  - gold: «Navez» `PER`
  - predicted: «mon cher Navez» `PER`
  - context: …nd la provision est épuisée on la renouvelle. la votre **[mon cher Navez]**, a été si abondante et si bien faite que vous ne pouve…
- `qwen3-235b-a22b-2507` · letter `Ms. II 70/3/416`
  - gold: «Navez» `PER`
  - predicted: «Mon cher Navez» `PER`
  - context: Rome le 24 Novem 1821 –  **[Mon cher Navez]**  Nous avons tous été bien peinés quand nous avons appr…
- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/2/61`
  - gold: «votre voitiex Trotrot tailleur» `PER`
  - predicted: «Trotrot» `PER`
  - context: …otre hôtel vous pouvez le remettre chez  votre voitiex **[Trotrot]** tailleur au numéro  38 même [??]  notre dame des victo…

### `M4` decidable label flip — mechanical (semi / in_text / bounded)

1971 events across all systems. Repair: one keystroke; PER/PER_NOM recoverable from capitalisation.

- `gliner2` · letter `Ms. II 70/2/69` · *PER_NOM->PER*
  - gold: «Navez» `PER`
  - predicted: «Navez» `PER_NOM`
  - context: Paris 27 mai 1832  Vraiment mon cher **[Navez]** vous avez été bien inspiré de ne point envoyer vos tab…
- `gliner2` · letter `Ms. II 70/5/540` · *PER_NOM->PER*
  - gold: «Monsieur F. J. Navez» `PER`
  - predicted: «F. J. Navez» `PER_NOM`
  - context: …tions nombreux  Ledit.  [Address:] a Monsieur Monsieur **[F. J. Navez]** Chez Monsieur de Hemptinne Rue des frippiers Bruxelles
- `gliner2` · letter `Ms. II 70/1/62` · *PER_NOM->PER*
  - gold: «Napoléon» `PER`
  - predicted: «Napoléon» `PER_NOM`
  - context: … a été mise en vente à Paris. C'est le couronnement de **[Napoléon]** qui avait été destiné pour Milan, dont le pareil est à…

### `C1` missed mention — cognitive (silent / external / unbounded)

34274 events across all systems. Repair: re-read the letter (and the scan) to find what is not there.

- `qwen3-235b-a22b-2507` · letter `Ms. II 70/4/41` · *identity-critical (PER/LOC)*
  - gold: «Paris» `LOC`
  - predicted: — *(no prediction)*
  - context: **[Paris]** ce 16 décembre 1826  Monsieur Navez, j'ai été bien dis…
- `qwen3-vl-30b-a3b-instruct` · letter `Ms. II 70/2/99` · *identity-critical (PER/LOC)*
  - gold: «David» `PER`
  - predicted: — *(no prediction)*
  - context: …elles productions qu'à faites sous l'empire l'école de **[David]** vous ne sauriez croire le plaisir que j’ai eu à contem…
- `camembertav2` · letter `Ms. II 70/3/413` · *identity-critical (PER/LOC)*
  - gold: «trinité du mont» `LOC`
  - predicted: — *(no prediction)*
  - context: …sions) ont su le toucher, vous n’avez plus à naples de **[trinité du mont]** pour cuillir des fleurs et plus le petit ange pour à p…

### `C2` spurious mention — cognitive (silent / in_text / bounded)

16839 events across all systems. Repair: read the context and adjudicate against the guidelines.

- `camembertav2` · letter `Ms. II 70/3/493` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «Projet» `PER`
  - context: …et tout en m'en bercant Je vois bien qu'il est loin de **[Projet]** a la Chose  Beaucamp est revenu de la Campagne que ver…
- `camembertav2` · letter `Ms. II 70/1/57` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «e» `LOC`
  - context: … Sur l'effet d'un tableau.  Je suis avide de voir Lazar**[e]** aussi me procurerai-je ce plaisir dans les premiers jo…
- `dalembert` · letter `Ms. II 70/1/180` · *identity-critical (PER/LOC)*
  - gold: — *(nothing in the gold)*
  - predicted: «Joseph» `LOC`
  - context: …rès. C’est l’origine l’ange du seigneur apparait à st. **[Joseph]** en Egypte pendant son sommeil il vient lui annoncer qu…

### `C3` semantic label confusion — cognitive (silent / external / bounded)

2423 events across all systems. Repair: resolve metonymy / consult prosopography.

- `qwen3-235b-a22b-2507` · letter `Ms. II 70/2/42` · *ART->PER*
  - gold: « gerardi» `PER`
  - predicted: «bella gerardi» `ART`
  - context: … infecté sa servante maîtresse la madone santissima la **[bella gerardi]** parceque Schnetz ne l'a point approchée mais n'importe…
- `camembertav2` · letter `Ms. II 70/2/158` · *LOC->PER*
  - gold: «hercule» `PER`
  - predicted: «ule» `LOC`
  - context: …ouffrant de la blessure qu'il reçut d'une flêche d'herc**[ule]**, il évente son pied douloureux avec une aile d'aigle —…
- `qwen3-235b-a22b-2507` · letter `Ms. II 70/1/177` · *ART->LOC*
  - gold: «Chapelle Sixtine» `LOC`
  - predicted: «Chapelle Sixtine» `ART`
  - context: …un grand sentiment de respect en d’admiration c’est la **[Chapelle Sixtine]**! En présence de l’œuvre de ce grand homme Michel, Ange…

### `C4` mention segmentation — cognitive (silent / external / bounded)

4206 events across all systems. Repair: decide who or what the mention denotes, then re-cut the span.

- `gliner2-ft` · letter `Ms. II 70/1/36` · *merge*
  - gold: «Van Reeth» `PER`
  - predicted: «Verheyden⏎Van Reeth» `PER`
  - context: …ls d'après son dessin fait une gravure d'un tableau de **[Verheyden Van Reeth]** grave un petit tableau de Dekeyser d'après le dessin q…
- `camembertav2` · letter `Ms. II 70/2/21` · *merge*
  - gold: «Vatican» `LOC`
  - predicted: «Vatican Roma» `LOC`
  - context: …Address:] al Signor Navez via dei Greci N° 26 al primo **[Vatican Roma]** P.P. PAYÉ — Italia — 8 MAR
- `camembertav2` · letter `Ms. II 70/2/107` · *split*
  - gold: «La Dubufe» `PER`
  - predicted: «Dubufe» `PER`
  - context: …rtrait de Mr. Gisors Architecte mais il est fort bien. **[La Dubufe]** ce peintre à la mode jadis pour les portrait commence …
