# NER error analysis — fold 0

Fold 0 contains 113 letters with 2572 gold spans.
Snippets: `[SPAN]` is the gold entity, surrounded by ±60 chars of context (line-breaks stripped).

## Aggregate per-model status of gold spans

| Model | strict hit | boundary only | label only | boundary+label | missed |
|---|---:|---:|---:|---:|---:|
| `gliner2` | 697 (27%) | 76 (3%) | 313 (12%) | 84 (3%) | 1402 (55%) |
| `gliner2-ft` | 1606 (62%) | 344 (13%) | 76 (3%) | 44 (2%) | 502 (20%) |
| `dalembert` | 1377 (54%) | 452 (18%) | 86 (3%) | 93 (4%) | 564 (22%) |
| `camembertav2` | 1214 (47%) | 563 (22%) | 96 (4%) | 106 (4%) | 593 (23%) |
| `claude-opus-4.8` | 1283 (50%) | 373 (15%) | 62 (2%) | 51 (2%) | 803 (31%) |
| `qwen3-235b-a22b-2507` | 597 (23%) | 161 (6%) | 38 (1%) | 30 (1%) | 1746 (68%) |
| `qwen3-vl-30b-a3b-instruct` | 570 (22%) | 128 (5%) | 35 (1%) | 35 (1%) | 1804 (70%) |

## Malformed spans — per-model validity-check counts

Counted over ALL predicted spans (gold-matched or not). These checks ask whether a span denotes a readable substring of the letter, not whether it names a real entity: a well-formed span that invents an entity is counted as spurious, not here. Checks: `invalid_offsets` (start<0, end<=start, or end>len(text) — LLM-only failure mode); `empty_content` (no alphabetic characters in the span); `tiny_midword` (<=3 chars, both edges strictly inside a word).

| Model | total preds | invalid_offsets | empty_content | tiny_midword | total malformed |
|---|---:|---:|---:|---:|---:|
| `gliner2` | 1455 | 1 | 0 | 0 | 1 (0.1%) |
| `gliner2-ft` | 2847 | 0 | 0 | 0 | 0 (0.0%) |
| `dalembert` | 3214 | 0 | 47 | 25 | 72 (2.2%) |
| `camembertav2` | 3887 | 0 | 12 | 379 | 391 (10.1%) |
| `claude-opus-4.8` | 2159 | 0 | 0 | 0 | 0 (0.0%) |
| `qwen3-235b-a22b-2507` | 1313 | 0 | 3 | 0 | 3 (0.2%) |
| `qwen3-vl-30b-a3b-instruct` | 1175 | 0 | 7 | 0 | 7 (0.6%) |

## A. Fine-tuning saved us — caught by ≥ 2 fine-tuned systems, missed / mislabelled by ALL prompted LLMs

*197 candidate spans in total; showing up to 2 per label (seed=42).*

| Label | PER | LOC | ORG | EVT | ART | PER_NOM | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 37 | 40 | 28 | 14 | 38 | 40 | 0 |

### PER  (37 spans, showing 2)

**Letter `Ms. II 70/1/182`**
- gold: `PER` at chars [3924:3930]
- context: …sir que cela ma fuit, car nous n'ignorez pas que je porte a **[Robert]** la plus franche amitié. Ce succes l’a beaucoup encouragé. P…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/168`**
- gold: `PER` at chars [333:344]
- context: …onsieur, d'agréer l'expression de mes sentimens d'estime.;  **[f. Paelinck]** née Horgniésr Mercredi matin.
- per-model status of this gold span:
    - `gliner2`: label_only
    - `gliner2-ft`: strict_hit
    - `dalembert`: missed
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### LOC  (40 spans, showing 2)

**Letter `Ms. II 70/2/69`**
- gold: `LOC` at chars [2834:2848]
- context: …e carnaval de même grandeur environ que sa moisson dans les **[marais Pontins]** je suis faché que vous n'ayez point vu ce tableau il vous e…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: boundary_and_label
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/2/3`**
- gold: `LOC` at chars [432:441]
- context: …dresse deux gravures d'après Jacquand, pour l’Exposition de **[Bruxelles]**, représentant l'une Rubens recevant dans son atelier la vis…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### ORG  (28 spans, showing 2)

**Letter `Ms. II 70/2/2/176bis`**
- gold: `ORG` at chars [403:416]
- context: …confiance Monsieur de vous prier de vouloir l'accepter dans **[votre atelier]** je ne puis que vous donner des rapports satisfaisants sur s…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/153`**
- gold: `ORG` at chars [1753:1770]
- context: … contient l'expression de leur satisfaction des etudes de l'**[académie de Namur]** et les felicitations qu’ils croient devoir adresser a mr le…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### EVT  (14 spans, showing 2)

**Letter `Ms. II 70/2/76`**
- gold: `EVT` at chars [1438:1443]
- context: …ti depuis quelque temps à peine si j'ai une idée confusé du **[salon]** tant j'y suis resté peu de temps je sais que vous êtes parf…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: boundary_only
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/13`**
- gold: `EVT` at chars [1296:1306]
- context: …s motifs, pour ainsi me priver de l'honneur  de figurer à l'**[exposition]**, outre que cela est fort désagréable, c'est en Certaine Sor…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### ART  (38 spans, showing 2)

**Letter `Ms. II 70/5/433`**
- gold: `ART` at chars [107:133]
- context: …i reçu votre lettre en datte du 18 8t. qui m'apprend que le **[portrait de mon petit fils]** est terminé, je suis très désireuse d'aller le voir chez vo…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: boundary_only
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: boundary_and_label
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/126`**
- gold: `ART` at chars [2766:2777]
- context: …u'à tous mes vrais camarades d'Atelier. Je vous aimez avoir **[vos dessins]** de suite faites moi l'honner m'ecrire un mot et en peu de j…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### PER_NOM  (40 spans, showing 2)

**Letter `Ms. II 70/5/487`**
- gold: `PER_NOM` at chars [4506:4512]
- context: …s ; le palais qu'on eleve pour le Roi à Bruxelles, celui du **[Prince]** à Tervueren demanderont des embellissemens de tout genre et…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: label_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/2/69`**
- gold: `PER_NOM` at chars [4050:4058]
- context: …per l'épidémie a commencé dans notre maison par la bonne de **[ma fille]** et a prise avec une force épouvantable ensuite la femme de …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

## B. Boundary drift on fine-tuned systems — lenient hit but not strict hit for ≥ 2 fine-tuned systems (illustrates the strict/lenient gap the paper reports)

*483 candidate spans in total; showing up to 2 per label (seed=42).*

| Label | PER | LOC | ORG | EVT | ART | PER_NOM | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 101 | 47 | 92 | 40 | 106 | 97 | 0 |

### PER  (101 spans, showing 2)

**Letter `Ms. II 70/1/190`**
- gold: `PER` at chars [2378:2382]
- context: …vernement provisoire qui mangerait la barque temporelle. Le **[pape]** n'a plus aucune force sinon la force morale et a sujet il v…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: strict_hit
    - `dalembert`: missed
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `camembertav2`: `PER` [2380:2382] = 'pe'
    - `gliner2-ft`: `PER` [2378:2382] = 'pape'

**Letter `Ms. II 70/1/162`**
- gold: `PER` at chars [1132:1139]
- context: …la plus haute Considération. B. P. Ommeganck vice President **[Willems]** secret
- per-model status of this gold span:
    - `gliner2`: boundary_and_label
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: strict_hit
    - `qwen3-vl-30b-a3b-instruct`: strict_hit
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `PER` [1132:1137] = 'Wille'
    - `camembertav2`: `PER` [1132:1139] = 'Willems'
    - `gliner2-ft`: `PER` [1132:1146] = 'Willems secret'

### LOC  (47 spans, showing 2)

**Letter `Ms. II 70/1/196`**
- gold: `LOC` at chars [2538:2564]
- context: …e suis avec respect votre tout devoué élève. Albert Roberti **[rue de la chaussée d’Antin]** N° 23
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: boundary_only
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `LOC` [2538:2570] = 'rue de la chaussée d’Antin N° 23'
    - `camembertav2`: `LOC` [2538:2570] = 'rue de la chaussée d’Antin N° 23'
    - `gliner2-ft`: `LOC` [2538:2564] = 'rue de la chaussée d’Antin'

**Letter `Ms. II 70/2/34`**
- gold: `LOC` at chars [2408:2414]
- context: … est aussi un des bailleurs de fonds est parti mais c'est de**[ Paris]** qu'il se rend dans ce pays. il utilisera son voyage en se p…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: boundary_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `LOC` [2409:2414] = 'Paris'
    - `camembertav2`: `LOC` [2409:2414] = 'Paris'
    - `gliner2-ft`: `LOC` [2409:2414] = 'Paris'

### ORG  (92 spans, showing 2)

**Letter `Ms. II 70/1/389`**
- gold: `ORG` at chars [2364:2377]
- context: …rochain deux de mes tableaux comme vous etes ami des Mrs de **[la Commission]** je voudrais etre asseré par un lettre de vous qu'on placera…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: missed
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: boundary_and_label
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `camembertav2`: `ORG` [2367:2377] = 'Commission'
    - `gliner2-ft`: `ORG` [2367:2377] = 'Commission'

**Letter `Ms. II 70/4/119`**
- gold: `ORG` at chars [18:29]
- context: Mon cher Navez!  L**['université]** accepte votre offre obligeante avec la plus grande reconnai…
- per-model status of this gold span:
    - `gliner2`: boundary_only
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: missed
    - `claude-opus-4.8`: boundary_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `ORG` [17:29] = 'L'université'
    - `gliner2-ft`: `ORG` [19:29] = 'université'

### EVT  (40 spans, showing 2)

**Letter `Ms. II 70/5/489`**
- gold: `EVT` at chars [1130:1149]
- context: …iste de preferera  Je crois déja vous avoir marqué que dans **[le dernier concours]** laissé au choix de la Reine elle avoit hesité entre Hector …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: missed
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `camembertav2`: `EVT` [1131:1149] = 'e dernier concours'
    - `gliner2-ft`: `EVT` [1133:1149] = 'dernier concours'

**Letter `Ms. II 70/1/13`**
- gold: `EVT` at chars [1450:1462]
- context: … Pour vous Convaincre, Monsieur, quand au dessin envoyé à l'**[expo- sition]**, J'ai l'honneur de vous adresser par chemin de fer de l'Éta…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: boundary_only
    - `camembertav2`: strict_hit
    - `claude-opus-4.8`: boundary_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `EVT` [1449:1462] = ''expo- sition'
    - `camembertav2`: `EVT` [1450:1462] = 'expo- sition'

### ART  (106 spans, showing 2)

**Letter `Ms. II 70/2/99`**
- gold: `ART` at chars [1470:1480]
- context: …vous dispenser de venir à l’exposé tion prochaine car ou les**[ tableaux ]**qui y seront vous avez le musée de Versailles qu’un artise s…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `ART` [1471:1479] = 'tableaux'
    - `camembertav2`: `ART` [1471:1479] = 'tableaux'
    - `gliner2-ft`: `ART` [1471:1479] = 'tableaux'

**Letter `Ms. II 70/1/196`**
- gold: `ART` at chars [1369:1387]
- context: … un des tableaux qui fait le plus de plaisir, surtout comme **[tableau d’Histoire]**, tout le monde lui en fait compliment; les éloges donnés pa…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `ART` [1369:1378] = 'tableau d'
    - `camembertav2`: `ART` [1369:1379] = 'tableau d’'
    - `gliner2-ft`: `ART` [1369:1387] = 'tableau d’Histoire'

### PER_NOM  (97 spans, showing 2)

**Letter `Ms. II 70/5/487`**
- gold: `PER_NOM` at chars [4479:4484]
- context: …heureux fruits de vos etudes ; le palais qu'on eleve pour le**[ Roi ]**à Bruxelles, celui du Prince à Tervueren demanderont des emb…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `PER_NOM` [4480:4483] = 'Roi'
    - `camembertav2`: `PER_NOM` [4478:4483] = 'e Roi'
    - `gliner2-ft`: `PER_NOM` [4480:4483] = 'Roi'

**Letter `Ms. II 70/4/60`**
- gold: `PER_NOM` at chars [1481:1488]
- context: …aire ce tableau ou du moins pour prendre apresent la tête de**[ Madame]** & pour faire le reste aussitot que vous le pourrez Nos enfa…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: boundary_only
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_and_label
    - `claude-opus-4.8`: boundary_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- fine-tuned systems' actual predicted spans:
    - `dalembert`: `PER_NOM` [1482:1488] = 'Madame'
    - `gliner2-ft`: `PER_NOM` [1482:1488] = 'Madame'

## C. Label collisions — ≥ 2 systems have the exact boundaries but the wrong label (typically PER ↔ PER_NOM, LOC ↔ ORG)

*90 candidate spans in total; showing up to 2 per label (seed=42).*

| Label | PER | LOC | ORG | EVT | ART | PER_NOM | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 18 | 5 | 9 | 2 | 17 | 39 | 0 |

### PER  (18 spans, showing 2)

**Letter `Ms. II 70/5/487`**
- gold: `PER` at chars [2138:2142]
- context: …emporté le prix etoit une grande et mauvaise Composition de **[Bato]** figure pied dans les halle, et formant une marche de grenad…
- per-model status of this gold span:
    - `gliner2`: label_only
    - `gliner2-ft`: strict_hit
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `ART`
    - `gliner2`: predicted `PER_NOM`
    - `camembertav2`: predicted `ART`

**Letter `Ms. II 70/5/365`**
- gold: `PER` at chars [43:62]
- context: Mon cher Monsieur Navez,   Vos deux petits **[Roger Van der Weyde]** n'ont pas été vendus à la dernière vente, ils n'auraient pa…
- per-model status of this gold span:
    - `gliner2`: label_only
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: strict_hit
- systems' wrong labels at the exact boundaries:
    - `gliner2`: predicted `PER_NOM`
    - `qwen3-235b-a22b-2507`: predicted `ART`

### LOC  (5 spans, showing 2)

**Letter `Ms. II 70/2/76`**
- gold: `LOC` at chars [258:272]
- context: …morts. il y a peu de jours encore  que j’ai j’ai conduit au **[père la Chaise]** un jeune membre de cette famille tous ce malheurs et peut-ê…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `camembertav2`: predicted `PER_NOM`

**Letter `Ms. II 70/5/486`**
- gold: `LOC` at chars [415:419]
- context: …idence epistolaire. Faber a egalement obtenu une medaille à **[Gand]** pour ses porcelaine blanches, fabrique nationale Gulsbenels…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: strict_hit
    - `dalembert`: strict_hit
    - `camembertav2`: label_only
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `camembertav2`: predicted `ART`
    - `qwen3-235b-a22b-2507`: predicted `EVT`

### ORG  (9 spans, showing 2)

**Letter `Ms. II 70/2/75`**
- gold: `ORG` at chars [255:261]
- context: …ient brisées quand j'ai vu cela  je suis allé de suite chez **[Vallee]**  pour le prier de me donner deux  hommes pour les ouvrir et…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: label_only
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: label_only
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `camembertav2`: predicted `LOC`
    - `qwen3-vl-30b-a3b-instruct`: predicted `PER`
    - `qwen3-235b-a22b-2507`: predicted `PER`
    - `gliner2-ft`: predicted `PER`
    - `claude-opus-4.8`: predicted `PER`

**Letter `Ms. II 70/1/12`**
- gold: `ORG` at chars [1992:2007]
- context: … Subside, ni du gouvernement, ni de la province, ni même de **[ma Ville Natale]**, pour faire mes études, tandis que Je sais fort bien que l'…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: boundary_only
    - `camembertav2`: boundary_only
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `claude-opus-4.8`: predicted `LOC`
    - `qwen3-235b-a22b-2507`: predicted `LOC`

### EVT  (2 spans, showing 2)

**Letter `Ms. II 70/1/153`**
- gold: `EVT` at chars [630:636]
- context: …a régence de Namur, faits juger mes concours à l'académie d'**[anvers]** ou j'ai toujours obtenue, des bons resultats; pour pourrez …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: label_only
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `claude-opus-4.8`: predicted `LOC`
    - `gliner2-ft`: predicted `LOC`

**Letter `Ms. II 70/2/28`**
- gold: `EVT` at chars [5784:5790]
- context: …ze heures du soir ils rentrèrent tous deux Romee (c'était à **[Paques]** qu'il s'était donné ce beau repas) bras dessus bras dessous…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: strict_hit
    - `qwen3-vl-30b-a3b-instruct`: strict_hit
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `LOC`
    - `camembertav2`: predicted `LOC`

### ART  (17 spans, showing 2)

**Letter `Ms. II 70/2/54`**
- gold: `ART` at chars [1146:1155]
- context: …rer. Castor et Pollux, Isis Paris, Antinous, Flore, Thalie, **[Melpomene]**   Therpsicore, Clio et Calliope, Minerve, l'Amour grec, Ado…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: label_only
    - `dalembert`: label_only
    - `camembertav2`: missed
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `gliner2-ft`: predicted `PER`

**Letter `Ms. II 70/2/54`**
- gold: `ART` at chars [1189:1196]
- context: …, Flore, Thalie, Melpomene   Therpsicore, Clio et Calliope, **[Minerve]**, l'Amour grec, Adonis et enfin la Venus de Médicis. il faud…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: label_only
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: strict_hit
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `camembertav2`: predicted `LOC`
    - `gliner2-ft`: predicted `LOC`

### PER_NOM  (39 spans, showing 2)

**Letter `Ms. II 70/1/190`**
- gold: `PER_NOM` at chars [4924:4932]
- context: … longtems sans vous donner de mes nouvelles. Mon ami intime **[Portaels]** étant chargé de me rappeler tres souvent à votre bon souven…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: label_only
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `camembertav2`: predicted `PER`
    - `claude-opus-4.8`: predicted `PER`
    - `gliner2-ft`: predicted `PER`

**Letter `Ms. II 70/1/200`**
- gold: `PER_NOM` at chars [76:86]
- context: …40. Mon cher Maître!  Je profite de l’occasion du départ de **[M. Purtoit]** architecte des hospices de Bruxelles, pour me rapeler à vot…
- per-model status of this gold span:
    - `gliner2`: strict_hit
    - `gliner2-ft`: label_only
    - `dalembert`: label_only
    - `camembertav2`: label_only
    - `claude-opus-4.8`: label_only
    - `qwen3-235b-a22b-2507`: label_only
    - `qwen3-vl-30b-a3b-instruct`: label_only
- systems' wrong labels at the exact boundaries:
    - `dalembert`: predicted `PER`
    - `camembertav2`: predicted `PER`
    - `qwen3-vl-30b-a3b-instruct`: predicted `PER`
    - `qwen3-235b-a22b-2507`: predicted `PER`
    - `gliner2-ft`: predicted `PER`
    - `claude-opus-4.8`: predicted `PER`

## D. Universally missed — no system caught the gold span (diagnostic for annotation quirks or genuinely hard cases)

*153 candidate spans in total; showing up to 2 per label (seed=42).*

| Label | PER | LOC | ORG | EVT | ART | PER_NOM | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 10 | 9 | 16 | 10 | 39 | 69 | 0 |

### PER  (10 spans, showing 2)

**Letter `Ms. II 70/3/493`**
- gold: `PER` at chars [2944:2947]
- context: …dieu Mon Cher Navez Je Suis tout à vous D'amitié            **[[…]]**  [Address:] A Monsieur Le Chevalier Navez A Bruxelles
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/60`**
- gold: `PER` at chars [512:517]
- context: … par le porteur s'il vous plait. Je vous Salue amicalement. **[[...]]** Bruxelles le 27. 7.bre 1825.  [...]
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### LOC  (9 spans, showing 2)

**Letter `Ms. II 70/1/132`**
- gold: `LOC` at chars [0:26]
- context: **[Sourcelles, près Charleroi]**, le 25 7bre 1840 Mon cher et ancien collegue Moné Leclercq …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/3/457`**
- gold: `LOC` at chars [1493:1507]
- context: … que je suis chargé de l'execution d'un œil de bœuf dans la **[cour du Louvre]**, mais ce travail ne sera payé qu'en 1824 ainsi tu vois que …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### ORG  (16 spans, showing 2)

**Letter `Ms. II 70/2/123`**
- gold: `ORG` at chars [2352:2366]
- context: …ritses? pourrons¬ nous. jamais parvenir à nous organiser un **[gouver¬ nement]** régulier et capable d'inspirer de la confience à nous mêmes…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/2/26`**
- gold: `ORG` at chars [3923:3934]
- context: …au Code. Louis XVIII [...] a voulu que l'on le peignit dans **[son cabinet]**, mais ce à quoi tient le plus c'est à ce qu'on reconnaisse …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### EVT  (10 spans, showing 2)

**Letter `Ms. II 70/1/190`**
- gold: `EVT` at chars [50:76]
- context: Rome le 2 Mai 1848  Mon Cher Maitre,  A la veille **[des plus grands évenements]** et d'une guerre civille qui vont compromettre la délivrance…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/1/157`**
- gold: `EVT` at chars [372:382]
- context: …a Mr. Vandertinden et qu’il vient de me communiquer. que la **[nomination]** de mon mari comme peintre du Roi, est du 25 Juin 1815 Croye…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### ART  (39 spans, showing 2)

**Letter `Ms. II 70/1/126`**
- gold: `ART` at chars [451:464]
- context: …uvoir vairier chaque semaine la grandeur ou effet de toutes **[leurs figures]**, et tant des choses qu'on ne sait dire; nous chez Monsieur …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/4/82`**
- gold: `ART` at chars [1241:1267]
- context: … douane &. Si cela se peut chez vous, je vous renverrois la **[Disposition du St Philippe]** qui est bonne en vous priant de me faire passer un autre Su…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

### PER_NOM  (69 spans, showing 2)

**Letter `Ms. II 70/6/sup./79bis`**
- gold: `PER_NOM` at chars [605:653]
- context: …r n'est pas seulement un artiste de mérite: c'est encore un **[homme qui jouit à Rome d'une juste considération]**. Investi par le gouvernement de fonctions qui équivalent sa…
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

**Letter `Ms. II 70/2/116`**
- gold: `PER_NOM` at chars [833:841]
- context: …evrais dire voir pr- endre des barricades car je vous jure, **[mon cher]**, que je nʼai tué âme qui vive comment lʼaurais-je fait: je …
- per-model status of this gold span:
    - `gliner2`: missed
    - `gliner2-ft`: missed
    - `dalembert`: missed
    - `camembertav2`: missed
    - `claude-opus-4.8`: missed
    - `qwen3-235b-a22b-2507`: missed
    - `qwen3-vl-30b-a3b-instruct`: missed

## E. Universally spurious — ≥ 2 fine-tuned systems agree on a span that is NOT in the gold (possible gold-annotation gap; label is the *predicted* label, not a gold label)

*401 candidate spans in total; showing up to 2 per label (seed=42).*

| Label | PER | LOC | ORG | EVT | ART | PER_NOM | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 108 | 37 | 35 | 27 | 73 | 121 | 0 |

### PER  (108 spans, showing 2)

**Letter `Ms. II 70/3/408`**
- predicted (not in gold): `PER` at chars [1780:1788]
- agreeing models: `dalembert`, `gliner2-ft`
- context: … Roman  [Margin: Bremay aout 1842 artist statuaire français **[Mesentin]** de [...] /visiterit [...] Mengeard. Navez Monsieur Navez à …

**Letter `Ms. II 70/1/196`**
- predicted (not in gold): `PER` at chars [410:416]
- agreeing models: `camembertav2`, `dalembert`
- context: …oire ils sont en petit nombre, Ziegeler, à fait le prophéte **[Daniel]** dans la fosse aux liens, c'est un beau tableau grandement p…

### LOC  (37 spans, showing 2)

**Letter `Ms. II 70/6/sup./79bis`**
- predicted (not in gold): `LOC` at chars [1768:1772]
- agreeing models: `camembertav2`, `dalembert`, `gliner2-ft`
- context: … de la capitale.  [Address, potlood:] Laboureur Statuaire à **[Rome]**

**Letter `Ms. II 70/2/54`**
- predicted (not in gold): `LOC` at chars [1189:1196]
- agreeing models: `camembertav2`, `gliner2-ft`
- context: …, Flore, Thalie, Melpomene   Therpsicore, Clio et Calliope, **[Minerve]**, l'Amour grec, Adonis et enfin la Venus de Médicis. il faud…

### ORG  (35 spans, showing 2)

**Letter `Ms. II 70/3/428`**
- predicted (not in gold): `ORG` at chars [7725:7731]
- agreeing models: `camembertav2`, `dalembert`
- context: …e tu veux bien lui dire d'aimable dans ta lettre — tous ces **[musées]** te salue. Schetz Beauvoir Dupré réussi le […] — Calet est a…

**Letter `Ms. II 70/2/28`**
- predicted (not in gold): `ORG` at chars [7384:7395]
- agreeing models: `dalembert`, `gliner2-ft`
- context: … que quand j'étais avec Schnetz et les Robert, ou seul dans **[mon atelier]**, quels énormes progrès Schnetz a fait quelle différence de …

### EVT  (27 spans, showing 2)

**Letter `Ms. II 70/2/76`**
- predicted (not in gold): `EVT` at chars [1646:1657]
- agreeing models: `dalembert`, `gliner2-ft`
- context: …x les oies sont dans le petit salon d'entrée qui précède le **[salon carré]** où lʼon voit à langle à droite en entrant votre sommeil de …

**Letter `Ms. II 70/1/168`**
- predicted (not in gold): `EVT` at chars [170:178]
- agreeing models: `camembertav2`, `gliner2-ft`
- context: … d’être persuadé de ses regrét d’être privé si longtems des **[réunions]** auxquelles il attachait tant de prix. Je saisis avec empres…

### ART  (73 spans, showing 2)

**Letter `Ms. II 70/1/194`**
- predicted (not in gold): `ART` at chars [226:228]
- agreeing models: `camembertav2`, `dalembert`
- context: … Je vous en remercie du fond de mon cœur. Je vous joins ici **[le]** croquis que j'ai composé et que je soumet à vos conseils, l…

**Letter `Ms. II 70/2/28`**
- predicted (not in gold): `ART` at chars [9183:9191]
- agreeing models: `camembertav2`, `gliner2-ft`
- context: … rocher, des troncs d'arbres d'après Mrs Pier et Paul et de **[tableaux]** de son frère il va le soir à la villa dessiner d'après le n…

### PER_NOM  (121 spans, showing 2)

**Letter `Ms. II 70/3/428`**
- predicted (not in gold): `PER_NOM` at chars [3441:3460]
- agreeing models: `dalembert`, `gliner2-ft`
- context: …son église d'Assisi — j'ai fini un tableau aussi grand d'un **[chanteur napolitain]** — j'ai voulu tirer parti de la toile de mon tableau de Cori…

**Letter `Ms. II 70/2/133`**
- predicted (not in gold): `PER_NOM` at chars [4642:4650]
- agreeing models: `camembertav2`, `dalembert`, `gliner2-ft`
- context: …vous rappe lez-vous de quelle ignoble manière il parlait de **[son père]** il faut  lui rendre la justice sur un point c’est qu’il n’o…

## F. Malformed spans — spurious spans that also fail a validity check (see the aggregate table above for definitions)

*469 spurious ill-formed spans in total; showing up to 2 per (type × model) cell (seed=42).*

| Model | invalid_offsets | empty_content | tiny_midword |
|---|---:|---:|---:|
| `gliner2` | 1 | 0 | 0 |
| `gliner2-ft` | 0 | 0 | 0 |
| `dalembert` | 0 | 46 | 25 |
| `camembertav2` | 0 | 12 | 375 |
| `claude-opus-4.8` | 0 | 0 | 0 |
| `qwen3-235b-a22b-2507` | 0 | 3 | 0 |
| `qwen3-vl-30b-a3b-instruct` | 0 | 7 | 0 |

### invalid_offsets  (1 spans)

#### `gliner2`  (1 spans, showing 1)

**Letter `Ms. II 70/2/3`**
- predicted (not in gold): `PER_NOM` at chars [1946:1956]
- model: `gliner2`
- validity check failed: `invalid_offsets`
- context: …arad Allais. et tembrasse de tous cœur. tous a toi de cœur. **[Duval L C]**

### empty_content  (68 spans)

#### `dalembert`  (46 spans, showing 2)

**Letter `Ms. II 70/2/133`**
- predicted (not in gold): `LOC` at chars [4478:4479]
- model: `dalembert`
- validity check failed: `empty_content`
- context: … et en font un commerce assez étendu avec long le terre et l**[']**almagne + toutes les fois que le nom de ce peintre me passé …

**Letter `Ms. II 70/1/169`**
- predicted (not in gold): `ART` at chars [718:719]
- model: `dalembert`
- validity check failed: `empty_content`
- context: …e pas ajouter que je desirerais beaucoup voir le volume de l**[']**Accadémie, je prévois que cela seroit difficile peut-être, e…

#### `camembertav2`  (12 spans, showing 2)

**Letter `Ms. II 70/2/28`**
- predicted (not in gold): `ART` at chars [2850:2851]
- model: `camembertav2`
- validity check failed: `empty_content`
- context: …oici le portrait fidèle et non la caricature de la réunion l**[']**Epra qui de votre tems était si aimable pour ceux qui me con…

**Letter `Ms. II 70/2/26`**
- predicted (not in gold): `PER` at chars [1606:1607]
- model: `camembertav2`
- validity check failed: `empty_content`
- context: …lle eut en ces passant par la bouche des signore S... C... M**[.]**.. E... chut me criez vous aussitôt, soyez tranquil mon cher…

#### `qwen3-235b-a22b-2507`  (3 spans, showing 2)

**Letter `Ms. II 70/2/265`**
- predicted (not in gold): `EVT` at chars [2287:2291]
- model: `qwen3-235b-a22b-2507`
- validity check failed: `empty_content`
- context: …oujours Votre bien dévoué ami Granet  paris, ce 23 Decembre **[1836]**.  [Address: granet 1836 artiste françois Directeur des Musé…

**Letter `Ms. II 70/6/14`**
- predicted (not in gold): `EVT` at chars [484:488]
- model: `qwen3-235b-a22b-2507`
- validity check failed: `empty_content`
- context: …dor Hildebrandt Professor d. Académie  Düsseldorf 18 August **[1849]**.  [Margin: Directeur de l'académie de Düsseldorf 1849 artis…

#### `qwen3-vl-30b-a3b-instruct`  (7 spans, showing 2)

**Letter `Ms. II 70/4/119`**
- predicted (not in gold): `PER_NOM` at chars [165:170]
- model: `qwen3-vl-30b-a3b-instruct`
- validity check failed: `empty_content`
- context: …en écrirai officiellement aussitôt mon retour de Bruges; je **[[...]]** dans une heure. — Veuillez vous mettre en rapport avec Mr V…

**Letter `Ms. II 70/4/96`**
- predicted (not in gold): `ART` at chars [410:415]
- model: `qwen3-vl-30b-a3b-instruct`
- validity check failed: `empty_content`
- context: …J'approuve en ratifie d'avance tout ce que votre expérience **[[...]]** laissera une entreprise pour la faire admettre en réussir. …

### tiny_midword  (400 spans)

#### `dalembert`  (25 spans, showing 2)

**Letter `Ms. II 70/2/69`**
- predicted (not in gold): `EVT` at chars [3745:3747]
- model: `dalembert`
- validity check failed: `tiny_midword`
- context: …n d'Aurel je vous dirai mon cher que depuis l'invasion du ch**[ol]**éra j'ai eu à supporter une foule de tribulations qui m'ont …

**Letter `Ms. II 70/2/147`**
- predicted (not in gold): `EVT` at chars [672:675]
- model: `dalembert`
- validity check failed: `tiny_midword`
- context: …te remercierai en t'embrassant.  Je te prie d'envoyer au déb**[arc]**adère soit un garçon d'hôtel soit n'importe qui pour m'indiq…

#### `camembertav2`  (375 spans, showing 2)

**Letter `Ms. II 70/5/487`**
- predicted (not in gold): `ART` at chars [2165:2166]
- model: `camembertav2`
- validity check failed: `tiny_midword`
- context: …rande et mauvaise Composition de Bato figure pied dans les h**[a]**lle, et formant une marche de grenadiers, du reste bien dess…

**Letter `Ms. II 70/2/28`**
- predicted (not in gold): `ORG` at chars [9933:9934]
- model: `camembertav2`
- validity check failed: `tiny_midword`
- context: …onnaire, il expose sa grande œuvre dans la sacristie de la t**[r]**inité dumont, il y a mis un blanc d'œuf qui lui a fait décou…
