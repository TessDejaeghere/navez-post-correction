# HTR post-correction assessment

364 manually corrected letters (358 distinct shelfmarks, 750 pages, 784,549 characters of Claude Opus 4.8 output), diffed against the annotators' submitted transcriptions.

## 1. Measured correction time

Time on task is Label Studio's `lead_time` (active-tab seconds, paused after ~60 s idle), capped at 60 min to contain tasks left open.

| | mean | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| minutes per letter | 11.0 | 6.9 | 13.6 | 23.9 | 60.0 |
| minutes per page | 5.3 | 3.4 | 6.6 | 11.4 | 30.0 |
| minutes per 1,000 characters | 6.8 | 4.6 | 6.8 | 11.2 | 348.8 |

Total recorded correction effort: **66.5 hours** for 364 letters.

| annotator | letters | pages | median min | mean min | min/page | edits | sec/edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 63 | 129 | 8.3 | 14.4 | 4.2 | 1157 | 47.0 |
| A2 | 49 | 100 | 8.5 | 12.7 | 3.9 | 620 | 60.3 |
| A3 | 122 | 248 | 7.4 | 11.8 | 3.7 | 2088 | 41.4 |
| A4 | 130 | 273 | 5.3 | 7.9 | 2.4 | 771 | 79.5 |

The mean is roughly twice the median: a minority of letters (faint ink, dense cursive) absorbs a disproportionate share of the effort, which is the same long tail the CER table reports.

## 2. Proper-noun error rate

Measured against the NER gold standard rather than a spelling heuristic. 357 of these letters were subsequently annotated for named entities, so we know exactly which character ranges a human considered to be a person, place or institution. Claude's raw output is aligned to that annotated text; a mention counts as an error if any of its characters had to be changed.

| entity label | mentions | edited | mention error rate | characters | character error rate |
|---|---:|---:|---:|---:|---:|
| **PER** | 2,753 | 343 | 12.5% | 27,172 | 4.4% |
| **LOC** | 1,758 | 107 | 6.1% | 14,013 | 3.2% |
| **ORG** | 584 | 29 | 5.0% | 9,312 | 2.0% |
| ART | 1,062 | 72 | 6.8% | 18,310 | 2.3% |
| EVT | 259 | 5 | 1.9% | 3,917 | 1.4% |
| PER_NOM | 1,205 | 93 | 7.7% | 16,511 | 2.8% |
| **all proper names (PER+LOC+ORG)** | 5,095 | 479 | **9.4%** | 50,497 | 3.6% |
| non-entity text | — | — | — | 649,063 | 1.6% |

`PER_NOM` is the control: those mentions (« mon père », « le ministre ») are ordinary common nouns that a human annotator marked as referring to a person. Named persons are corrected 1.6x as often as nominal references (12.5% against 7.7%), which isolates the difficulty of the *name* from the difficulty of the passage it sits in.

At character level, 3.6% of the text inside a proper name had to be corrected against 1.6% of the text outside any entity — a factor of 2.2. This is the number that matters for the downstream network: a mis-transcribed common noun costs a reader nothing, whereas a mis-transcribed surname silently creates or destroys a node.

### Cross-check against the spelling heuristic

A purely orthographic detector — a token counts as a proper noun if it is capitalised, at least three characters long, and not a sentence opener, month, weekday or honorific — gives the following, over all 364 corrections including the letters without NER annotations.

| | tokens | edited | error rate |
|---|---:|---:|---:|
| capitalised tokens | 8,385 | 768 | 9.2% |
| all tokens | 136,589 | 4,240 | 3.1% |

The heuristic puts the ratio at 3.0x. It is reported only as a sanity check; the gold-grounded figures above are the ones to quote, since they rest on a human judgement of what is a name rather than on capitalisation in a 19th-century hand.

### Proper-name mentions Claude got wrong

Left is Claude Opus 4.8, right is the annotated gold mention.

- `Ms. II 70/1/61` `LOC`: Votre ancien élève, rue des fossés du temple → **rue des fossés du temple No 68**
- `Ms. II 70/1/67` `PER`: Navez → **Poisson**
- `Ms. II 70/1/389` `ORG`: du Directeur de L'Acad: d'F: → **L'Acad du Fr**
- `Ms. II 70/2/256` `PER`: Vte Antoinette [...] → **Get. Antoinette**
- `Ms. II 70/2/47` `PER`:  → **M Schneider**
- `Ms. II 70/3/402` `ORG`: Nicol des bras → **l’ecole des beaux arts**
- `Ms. II 70/3/402` `LOC`:  → **Rue Jacob 36**
- `Ms. II 70/3/402` `PER`:  → **Poisson**
- `Ms. II 70/3/493` `PER`: H. Sané → **[…]**
- `Ms. II 70/3/517` `PER`: Thomas → **Romain**
- `Ms. II 70/3/517` `PER`: s de ma part à Mde Davie → **Mde David**
- `Ms. II 70/3/517` `PER`: à un ami de cœur. Mr S. d'Albe désire → **Mr […]**
- `Ms. II 70/3/517` `PER`: de Mr Thevenin comme tu connois la peinture de Mr Land → **Mr […]**
- `Ms. II 70/3/517` `ORG`: l'atelier et il m → **son atelier**

## 3. Did corrections happen where they were pre-annotated?

`char coverage` is the share of the transcription each highlight class covers; `hit rate` is the share of edits that fall inside it; `lift` is hit rate over coverage, i.e. how much better than pointing at random the highlight is; `span precision` is the share of highlighted spans that contained at least one edit.

| highlight | spans | char coverage | edit hit rate | lift | span precision |
|---|---:|---:|---:|---:|---:|
| Suspicious French | 6,960 | 3.7% | 18.8% | 5.09x | 51.0% |
| Person Name | 2,141 | 2.5% | 8.6% | 3.40x | 13.1% |
| Format Anomaly | 90 | 0.9% | 1.5% | 1.68x | 72.2% |
| any | 9,191 | 7.1% | 28.3% | 3.98x | 42.4% |

Taken together the three overlays cover 7% of the text and catch 28% of the corrections, a lift of 3.98x over chance. A lift above 1 means the overlay is informative; the gap between hit rate and 100% is the share of corrections the annotator found unaided, which is the part of the workload no interface currently supports.

## 4. Transcription cost

OpenRouter usage records for Claude Opus 4.8, converted at 1 USD = 0.92 EUR.

| | letters | pages | cost (USD) | cost (EUR) | EUR / letter | EUR / page |
|---|---:|---:|---:|---:|---:|---:|
| whole digitised corpus | 1,559 | 3,257 | $107.54 | €98.93 | €0.063 | €0.030 |
| the corrected subset | 358 | 736 | $25.14 | €23.12 | €0.065 | €0.031 |

Those same letters took 67 hours of human correction, so the inference is worth €0.35 of corrector time per hour: transcription would only stop being the cheap part of this pipeline if an annotator could be paid €0.35 an hour. The binding cost is human attention, not tokens, which is the premise of the whole paper stated as a budget line.

## 5. Mechanical / cognitive split of the edit log

4,636 edits over 364 letters (13 per letter), touching 63,317 characters.

| class | cost | edits | % of edits | characters | % of characters |
|---|---|---:|---:|---:|---:|
| whitespace / hyphenation / stray punctuation | mechanical | 740 | 16.0% | 16,454 | 26.0% |
| diacritic restoration | mechanical | 126 | 2.7% | 129 | 0.2% |
| capitalisation | mechanical | 204 | 4.4% | 210 | 0.3% |
| single-character substitution | mechanical | 1,420 | 30.6% | 1,490 | 2.4% |
| model meta-commentary | cognitive | 5 | 0.1% | 7,922 | 12.5% |
| silent omission restored | cognitive | 18 | 0.4% | 1,788 | 2.8% |
| proper-noun misreading | cognitive | 218 | 4.7% | 6,376 | 10.1% |
| illegible / uncertain passage | cognitive | 194 | 4.2% | 20,352 | 32.1% |
| other word-level substitution | cognitive | 1,711 | 36.9% | 8,596 | 13.6% |
| **mechanical total** | | **2,490** | **53.7%** | **18,283** | **28.9%** |
| **cognitive total** | | **2,146** | **46.3%** | **45,034** | **71.1%** |

Counted per edit and per character the split tells different stories, and both belong in the paper: mechanical fixes are numerous but tiny, cognitive ones are fewer and rewrite whole passages.

### Per annotator

| annotator | edits | % mechanical | % cognitive | proper-noun edits |
|---|---:|---:|---:|---:|
| A1 | 1,157 | 50.1% | 49.9% | 19 |
| A2 | 620 | 54.0% | 46.0% | 60 |
| A3 | 2,088 | 43.1% | 56.9% | 130 |
| A4 | 771 | 87.5% | 12.5% | 9 |

### Sample proper-noun corrections

Widened to whole words; left is Claude Opus 4.8, right is the annotator.

- `Ms. II 70/1/67`: devoué Navez [Address:] → **devoué Poisson [Address:]**
- `Ms. II 70/1/71`: son père artiste peintre Belge] → **son père]**
- `Ms. II 70/1/72`: d'histoire, Bruxelles Monsieur [Margin: Fran] → **d’Histoire Bruxelles]**
- `Ms. II 70/1/77`: de perspective artist Belge [...] → **de perspective]**
- `Ms. II 70/1/330`: [Address:] Vandenberghe 2 Mars 1848 artiste français peintre Monsieur → **[Address:] Monsieur Navez**
- `Ms. II 70/2/211`: empressées. F. D. Paris → **empressées. Ledit Paris**
- `Ms. II 70/2/47`: la laisserai auquel je → **la lesserai à M Schneider auquel**
- `Ms. II 70/2/47`: adresse à M. Schnei der je le → **adresse je le**
- `Ms. II 70/2/47`: de la Légion de → **du lion de**
- `Ms. II 70/2/47`: royale de à → **Royale Pays-Bas à**
- `Ms. II 70/3/402`: vais peut être le terminer Devigne le → **70 […] le**
- `Ms. II 70/3/402`: bon Camarade tes vieil → **tous camarade Ton vieil**
- `Ms. II 70/3/402`: ami J. Vollon t'embrasse de cœur → **ami de 53**
- `Ms. II 70/3/402`: 60 ans] [Address:] → **53 ans Poisson Rue Jacob 36]**

## Caveats

- `lead_time` is active-tab time, not effort, and is capped here at 60 min per task.
- These figures measure *residual* work after Claude's output, so they are a lower bound on its true error rate, not a CER.
- The gold-grounded entity rates inherit the NER gold standard's own limits: outside the 30 inter-annotator-agreement letters it is single-annotator, and mention boundaries carry the annotators' residual disagreement (lenient F1 0.94 for `PER`, 0.90 for `LOC`).
- The capitalisation heuristic reported as a cross-check will miss lowercased names in careless hands and over-count capitalised common nouns in 19th-century orthography.
- Edit classes are assigned by string tests in a fixed priority order; `other word-level substitution` is the residual class and is counted as cognitive because resolving it requires the scan.
