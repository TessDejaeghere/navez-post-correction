# NER_707 — annotation stats & intermediate IAA

Source: `data/labelstudio_output/NER_707_*_final.json`  (A1, A2, A3, A4)

## Per-annotator overview (full export, not only IAA)

| annotator | n_total | n_annotated | n_cancelled | n_spans | mean_spans/text | mean_len (chars) | median_len | min_len | max_len |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 127 | 127 | 0 | 3406 | 26.82 | 2406.3 | 1721 | 272 | 10764 |
| A2 | 96 | 96 | 0 | 2530 | 26.35 | 2697.1 | 1844 | 261 | 8887 |
| A3 | 190 | 190 | 0 | 3877 | 20.41 | 2136.5 | 1403 | 228 | 18313 |
| A4 | 246 | 246 | 0 | 5143 | 20.91 | 2340.0 | 1781 | 183 | 10913 |

**Totals across all four exports:** 659 tasks, 659 annotated, 14956 spans.

## Spans per label per annotator

| label | A1 | A2 | A3 | A4 | TOTAL |
|---|---|---|---|---|---|
| ART | 30 | 52 | 58 | 56 | 196 |
| ART_NOM | 544 | 339 | 461 | 596 | 1940 |
| EVT | 139 | 107 | 140 | 201 | 587 |
| LOC | 613 | 464 | 775 | 1192 | 3044 |
| ORG | 267 | 174 | 240 | 401 | 1082 |
| PER | 922 | 799 | 1430 | 2024 | 5175 |
| PER_NOM | 891 | 595 | 773 | 673 | 2932 |
| **TOTAL** | **3406** | **2530** | **3877** | **5143** | **14956** |

## Intermediate IAA (10 shared IAA letters)

| pair | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 |
|---|---|---|---|---|---|---|
| NER_707_A1_iaa10_u22 vs NER_707_A2_iaa10_u24 | 0.486 | 0.471 | 0.479 | 0.811 | 0.785 | 0.798 |
| NER_707_A1_iaa10_u22 vs NER_707_A3_iaa10_u23 | 0.654 | 0.634 | 0.644 | 0.768 | 0.743 | 0.755 |
| NER_707_A1_iaa10_u22 vs NER_707_A4_iaa10_u26 | 0.625 | 0.524 | 0.570 | 0.856 | 0.717 | 0.781 |
| NER_707_A2_iaa10_u24 vs NER_707_A3_iaa10_u23 | 0.465 | 0.465 | 0.465 | 0.778 | 0.778 | 0.778 |
| NER_707_A2_iaa10_u24 vs NER_707_A4_iaa10_u26 | 0.500 | 0.432 | 0.464 | 0.912 | 0.789 | 0.846 |
| NER_707_A3_iaa10_u23 vs NER_707_A4_iaa10_u26 | 0.581 | 0.503 | 0.539 | 0.844 | 0.730 | 0.783 |

**Macro-avg over 6 pairs:** strict F1 = 0.527, lenient F1 = 0.790
