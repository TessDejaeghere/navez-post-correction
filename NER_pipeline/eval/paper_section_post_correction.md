# Draft prose for the paper — NER post-correction assessment

Paste-ready text for the empty *NER post-correction assessment* heading, plus a
short insert for *Methodology* and two paragraphs for *Discussion*.
Counts are produced by `NER_pipeline/error_taxonomy.py --pred-dir
NER_pipeline/eval/predictions_filtered` (report in
`NER_pipeline/eval/error_taxonomy_filtered.md`, long-form counts in the matching
`.csv`); the figures are drawn by `NER_pipeline/plot_error_taxonomy.py` and
carry a `_filtered` suffix. The unsuffixed artefacts are the pre-filter run,
kept only for the counts of filter-removed spans quoted below.

**Framing.** The claim being made is *not* that recall matters more than
precision for network construction — that is well understood. It is that the
post-correction labour a system imposes can be **measured**, decomposed by the
kind of repair it demands, and calibrated in minutes against a real annotation
log. The recall findings are an output of applying the instrument, not the
contribution.

## Figure key (placeholders used below)

| placeholder | file | shows |
|---|---|---|
| **Figure [BURDEN]** | `fig_taxonomy_burden_filtered.png` | correction events per letter, mechanical vs cognitive |
| **Figure [COMPOSITION]** | `fig_taxonomy_composition_filtered.png` | % of the system's correction events, by class |
| **Figure [UNBOUNDED]** | `fig_taxonomy_unbounded_filtered.png` | missed mentions per letter, mean over 5 folds ± 95 % CI |
| **Figure [COUNTS]** | `fig_taxonomy_counts_filtered.png` | raw event counts, models × class (appendix) |

---

## Insert for Methodology (after the mechanical/cognitive definition)

To make this distinction usable we need an instrument that takes a system's
output and returns not its distance from the gold standard but an estimate of
the work required to repair it. Accuracy metrics answer *how wrong is this
output*; a project that will post-correct the output needs to know *what will
this cost, and what kind of cost is it*. The two questions have different
answers whenever error types differ in how much human attention they demand,
which in our corpus they do by two orders of magnitude.

We operationalise this distinction into eight error classes, each defined by its
position on three axes. *Detectability* asks whether the error is demonstrably
wrong without reading the letter (evident), visible but only judgeable in
context (semi), or invisible in the correction interface (silent).
*Locality* asks whether the repair can be made from the span string alone
(in-span), from the letter text (in-text), or only by consulting the source
scan, the annotation guidelines or prosopographical reference works (external).
A third axis emerged from the data rather than from the design: *search scope*
distinguishes errors that are triggered by something on the screen (bounded —
the work grows with the number of predictions) from errors that nothing points
at (unbounded — the corrector has to re-read the whole letter to find what is
not there). Classes that are evident or semi-detectable and span- or text-local
are **mechanical**; classes that are silent and/or require external resolution
are **cognitive**.

The unit of measurement is the *edit event*: one action a corrector must
perform. A deterministic validity filter runs first over the predictions and
deletes spans that do not denote a readable substring of the letter: impossible
offsets, spans containing no alphabetic character, and fragments of three
characters or fewer cut out of the middle of a word. The test is one of
readability, not of plausibility: a prediction that quotes the letter correctly
while naming an entity that is not there passes untouched and is counted as a
spurious mention (C2), a cognitive repair. Over the spans that survive, every
gold span that
is not strictly recovered and every predicted span that is not aligned to a gold
span produces exactly one event. Gold and predicted spans are aligned
label-agnostically by greedy best-IoU one-to-one matching, so that a correct
span carrying a wrong label is counted once as a label error rather than twice
as a miss plus a spurious span. Events therefore cover the whole strict error
budget but are fewer than it, and the identity
`events + aligned repairs = strict FN + strict FP` is asserted at run time. The
gap is itself informative: between 1,126 and 4,446 events per system, depending
on the system, are double-charged by strict scoring but cost the corrector a
single keystroke. Charging one event per aligned pair also means that a
prediction wrong in both its label and its boundary is charged to the label
class alone: this happens in 1,751 of 4,394 label events (39.8 %), and 84 % of
the uncounted repairs would themselves be mechanical, so the instrument
understates mechanical labour by 1.3–5.0 % of a system's total events.

The mechanical/cognitive assignment is a claim about labour, so we test it
against labour actually performed. Our NER annotation round was itself a
post-correction task: every letter was pre-annotated with GLiNER2 and corrected
in Label Studio, which records the time on task and, for every span, whether it
was accepted, edited or drawn by hand. Diffing the suggestion against the
submission gives 26,121 classified edits over the 908 submissions whose
suggestion payload survived the export, drawn from 1,128 submissions by four
annotators totalling 145 hours of recorded work. Regressing time on task on the
same eight classes (OLS, annotator fixed effects, robust standard errors,
adjusted R² = 0.54) prices them. A cognitive edit costs 55 s (95 % CI 18–93),
a mention the system missed 14 s (8–20) to draw by hand, and a mechanical edit
0.8 s (−4 to 6) — not distinguishable from zero, as is deleting a spurious
suggestion. Reading the letter costs 90 s per 1,000 characters regardless. The
classification is therefore not merely a convenient grouping: the two cost tiers it posits are
separated by roughly two orders of magnitude in measured seconds. These prices
are estimated on GLiNER2's error mix and transfer to other systems only under
the assumption that a repair of a given class costs the same whoever caused it.

---

## §2.2.x NER post-correction assessment

Figure [BURDEN] recasts the seven systems' output as post-correction work on all
561 letters (12,445 gold spans, 22.2 per letter, pooled over the five test
folds), and Figure [COMPOSITION] shows how each system's events are distributed
over the taxonomy's eight classes. All counts are taken after the validity
filter, so they describe the work that reaches a corrector rather than the raw
output. Read as accuracy, these systems have
already been ranked
in §[X]; read as labour, they rank differently, and the three observations below
follow from the decomposition rather than from any single quantity. Several of
those quantities are individually recoverable from a sufficiently detailed
accuracy report — the missed-mention rate is recall in different units — and we
make no claim otherwise. What the instrument adds is that the components can be
counted in a common unit, compared against each other, and priced.

**The cheapest output to correct is not the most accurate output.** Claude Opus
4.8 leads on lenient F1 (0.72) and also produces the smallest total burden
(13.3 events per letter, of which 9.5 cognitive). But the two fine-tuned French
encoders, which trail it by 0.05–0.13 F1, more than halve the *unbounded*
burden (Figure [UNBOUNDED]): d'AlemBERT leaves 3.30 ± 0.56 and CamemBERTaV2
3.36 ± 0.38 mentions per letter that the corrector can only recover by
re-reading, against 5.89 ± 0.80 for Claude and 14.45 ± 1.33 and 15.29 ± 1.52 for
the two Qwen variants (means over the five folds, 95 % normal-approximation CI).
The intervals for the encoders and for Claude do not overlap. Expressed this
way the quantity is recall in different units — with 22.2 gold spans per letter,
a missed-mention rate of 3.30 is exactly a recall of 85.1 % — and we do not
claim it as a new measurement. What the taxonomy adds is its position in the
budget: missed mentions are 14.7 % of CamemBERTaV2's total repair work and
19.0 % of d'AlemBERT's, but 44.1 % of Claude's and 75.9 % of Qwen3-VL-30B's, a
decomposition no accuracy score yields because it requires counting the other
seven classes as well. It also puts the misses on the same scale as every other
repair, so they can be added: of Claude's 13.3 actions per letter, 5.89 are of
the kind that has no on-screen trigger. Because unbounded work scales with
letter length rather than with the number of predictions, this is the component
of the burden that a correction interface cannot absorb: a reviewer can be shown
a candidate span and asked to reject it, but cannot be shown an absence.

**A low false-positive count can be a symptom rather than a virtue.** The two
Qwen variants generate by far the fewest mechanical events (1.52 and 2.01 per
letter, against 6.58 for CamemBERTaV2) and would read as the tidiest systems in
a precision-oriented table. They are tidy because they barely predict, and
Figure [COMPOSITION] makes the point directly: missed mentions alone account for
71 % and 76 % of their correction events, and their mechanical blocks are the
thinnest in the panel. On the identity-critical labels that feed our clustering
step, they leave 7.97 and 8.09 cognitive events per letter, against 2.32 for
Claude and 3.48 for d'AlemBERT (Table [Y]). What the aggregate score reports as
restraint, the correction interface experiences as silence.

**Conversely, a system can be noisy and cheap.** CamemBERTaV2 produces the
largest number of events of any system (22.9 per letter) and the largest
mechanical block of any system (6.58 per letter), but the smallest cognitive
share (71 %, no longer separable from Claude's). Redundant overlapping spans are
12 % of its events, affix drift a further 9 % and token drift 7 % — the three
widest mechanical blocks in Figure [COMPOSITION], and none of them work that
needs a historian. What the figure does *not* show is the class that dominated
its raw output: roughly one event in eight before filtering was a malformed
span, and the validity filter removes every one of them before a corrector sees
them, which is precisely why that filter improved its lenient F1 by 3.3 points.
No filter touches the class that remains largest, however: 39 % of its surviving
events are spurious mentions, well-formed spans naming things the gold standard
does not, each costing an adjudication. Once the filter has run, its residual
profile is close to
d'AlemBERT's. Zero-shot GLiNER2 shows the same effect in a different class: 17 %
of its events are a single systematic confusion, predicting `PER_NOM` where the
gold label is `PER` (1,686 of 1,694 flips in one direction), which a
capitalisation rule recovers without opening the letter — it is the only system
in Figure [COMPOSITION] with a visible `M4` block. A model that fails
*systematically* is cheaper to repair than a model that fails *rarely but
unpredictably*.

The finer sub-types locate where the remaining cognitive work sits, and the two
system families fail in structurally different ways. The encoders' segmentation
errors are dominated by splits — d'AlemBERT cuts one gold mention into several
disjoint predictions 1,095 times against only 97 merges, CamemBERTaV2 1,312
against 143 — which is the expected signature of sliding-window token
classification with letter-level aggregation. Claude inverts the ratio (169
merges against 101 splits): it reads coordinated or apposed names as a single
referent, which is a reading decision rather than a windowing artefact and
correspondingly harder to undo. The distinction matters downstream, because a
split mention usually re-forms at the clustering step, where the two fragments
share a surface form, whereas a merged mention arrives as a single string that
no clustering step can separate. In the semantic label confusions, Claude's
largest single cell is predicting `LOC` where the gold label is `ORG` (82 cases,
against 4–24 for every other system) — the metonymy our annotators also
disagreed on (« l'académie de Namur » as a place or as an institution), which
the IAA table already flagged as the residual difficulty of the `ORG` class.
That the strongest system reproduces the annotators' own hardest judgement is a
result about the schema, not only about the model.

Finally, the taxonomy reframes the strict/lenient gap the previous section
reported. Between 3 % and 21 % of each system's events are affix drift (M2),
i.e. a leading article, honorific or stray punctuation that leaves the
referential core of the mention intact, and a further 1–7 % are token drift (M3)
that leaves the capitalised identity key unchanged (Figure [COMPOSITION]). These
are exactly the disagreements lenient F1 forgives and strict F1 punishes, and
for a downstream step that consumes surface strings they cost a normalisation
script, not a historian. The boundary disagreements that do change the identity
key are separated out as segmentation errors (C4) and remain cognitive: over the
seven systems, 9,544 boundary repairs are mechanical against 660 that alter
which entity the mention denotes.

---

## Optional subsection — what the instrument shows about optimising for recall

*Cut if space is short; it strengthens the Discussion but the section stands
without it. Numbers from the same event log; see the note at the end on how the
entity key is defined.*

That recall matters more than precision for a project which will post-correct
its own output is not a new observation, and the burden decomposition above
restates it in units of work. The instrument does, however, say something less
obvious about *which* recall: measured over the unit the downstream network
actually consumes, the metric and the objective disagree about which system to
deploy.

Our 7,023 identity-critical gold mentions correspond to only 5,253 distinct
per-letter entities, 1.34 mentions each: missing one mention of a correspondent
named three times in a letter costs nothing, since the node and its edge
survive, whereas missing the only mention destroys both. Mention-level recall
weights the two equally.

The second problem is that boundary-tolerant recall counts a mention as
recovered when the string it yields cannot be resolved to a person. On
`Ms. II 70/1/153`, two systems return a single `PER` span « J.V. Marinus Namur »,
fusing the correspondent with the city he writes from; its IoU with the gold
mention is 0.67, so lenient recall scores it as a hit, but the surface string
entering the blocking step matches no other mention of Marinus in the corpus.
Elsewhere « Silvie Odevaere » is returned as « Odevaere » (IoU 0.53, also a hit),
losing the forename that distinguishes her from her relatives. The size of this
effect is the distance between the two recall conventions: measured with any
overlap, d'AlemBERT recovers 96.9 % of identity-critical mentions and
CamemBERTaV2 95.7 %; requiring IoU ≥ 0.5 drops them to 86.5 % and 82.7 %, against
93.9 % and 85.8 % for Claude. Ten to thirteen points of the encoders' apparent
recall consist of mentions that are present but mis-cut, and a further 467 and
692 mentions respectively are segmentation events whose identity key has
actually changed, against 173 for Claude.

Recomputing recall over entities, and requiring that at least one surviving
mention retain its identity key, inverts the ranking:

| system | mention recall | entity recall | usable entity recall |
|---|---:|---:|---:|
| d'AlemBERT | **94.1 %** | 93.8 % | 86.6 % |
| CamemBERTaV2 | 92.9 % | 92.6 % | 82.0 % |
| Claude Opus 4.8 | 90.4 % | 92.2 % | **89.4 %** |
| GLiNER2 (fine-tuned) | 75.1 % | 75.9 % | 73.5 % |
| GLiNER2 (zero-shot) | 62.6 % | 72.2 % | 71.0 % |
| Qwen3-235B-A22B | 42.0 % | 48.5 % | 47.1 % |
| Qwen3-VL-30B-A3B | 41.6 % | 51.4 % | 49.9 % |

The distortion runs in both directions. For the sparse systems, mention recall
understates performance, because repetition protects them (GLiNER2 zero-shot
rises from 62.6 % to 72.2 %). For the high-recall encoders it overstates,
because their splits corrupt entities that recall counts as successes. The two
biases do not cancel, and they do not cancel equally across systems: d'AlemBERT
leads mention recall by 3.7 points and trails Claude by 2.8 on the quantity the
network actually consumes. The point is not that recall is the wrong thing to
care about, but that "recall" has to be defined over the unit the pipeline
consumes before it can arbitrate a deployment decision — and identifying that
unit is exactly what an analysis of downstream labour forces one to do.

The entity key is the capitalised-token normalisation our blocking step already
applies, so two people sharing a surname within one letter are conflated; this
slightly understates entity counts, uniformly across systems. "Usable" treats
every segmentation event as identity-destroying, which is right for merges but
pessimistic for splits that re-form at clustering, so the true value lies
between the last two columns. *Mention recall* in the table is computed after
the one-to-one alignment used throughout this section, so it is marginally
stricter than the raw any-overlap figures quoted above (94.1 % against 96.9 %
for d'AlemBERT): a gold mention counts as recovered only if a prediction is
assigned to it, not merely if some prediction touches it.

---

## Insert for Discussion

*System x produces this type of error, system y produces this type of error, for
a post-correction workflow z would be best.*

Read through the taxonomy, the choice between our system families is a choice
about which kind of labour to buy. The prompted LLMs — and Claude in particular
— produce output whose errors are few and semantically motivated, but
concentrated in the one class no interface can surface: mentions that are simply
absent. The fine-tuned encoders produce output that is noisier on the surface
but whose errors are overwhelmingly attached to something visible, and a
substantial share of which a deterministic filter removes before the historian
sees them (8 % of d'AlemBERT's raw events, 22 % of CamemBERTaV2's).
For a workflow in which recall is the binding constraint — ours, since missed
mentions silently truncate the correspondence network — d'AlemBERT's profile
looks preferable to Claude's despite the F1 ranking. That conclusion has to be
qualified, however, by the entity-level figures above: the encoders' recall
advantage is bought with segmentation errors that leave the mention present but
unresolvable, so on usable entities the ordering reverses again. The safe
reading is not that one system wins, but that the two fail in complementary
ways: their misses barely overlap, so a union of a prompted LLM with one
fine-tuned encoder cuts the permanently lost mentions from 409 to 193 (97.3 %
against 94.2 % mention recall). Whether that union also improves *usable*
entity recall depends on how overlapping spans from the two systems are
reconciled, and we have not tested it. The general point is not that one of
these systems is better, nor that recall should be preferred to precision, but
that the comparison can be conducted in units of human labour at all: given a
gold standard and a set of predictions, the procedure above returns an estimate
of what supervising each system would cost and what kind of work that cost
consists of. That is a question a project has to answer before it commits
annotator time, and one that a single accuracy score cannot answer, because the
systems are not making the same kind of mistake.

*What should a post-correction interface take into account?*

The three axes translate fairly directly into interface requirements. Errors
that are evident and in-span should never reach the historian: the validity
filter and the M1–M2 normalisation rules removed or repaired between 3 % and
31 %
of each system's raw events in our data (Figure [COMPOSITION]), and running them
is
a precondition for the interface, not a feature of it. Errors that are bounded
should be presented as decisions rather than as edits — a reviewer confronted
with a candidate span and its sentence can accept, reject or re-label it, and
the cost of that queue is predictable from the number of predictions. Errors
that are unbounded need a different affordance altogether, because there is
nothing to queue; this is the argument for the kind of pre-annotation overlay we
used in the HTR step, where independent signals (a second model, a gazetteer, a
name-shaped-string heuristic) are drawn on the text to give the reader's eye
somewhere to land. Designing for the aggregate F1 optimises the first two
categories, which are the cheap ones.

---

## Insert for Limitations

Outside the 30 letters used for inter-annotator agreement, the gold standard is
single-annotator. Spurious mentions (C2) are therefore scored against one
reader's judgement, and an unknown share of them are annotation gaps rather than
model errors. This is one reason the class is classified as cognitive:
adjudicating it requires the guidelines and, sometimes, the domain expert. It is
also the reason the validity filter cannot be extended to cover it: the filter
decides readability, which is a property of the span, whereas C2 asks whether
something is an entity, which is a property of the schema.
Second, our schema permits *nested* mentions — an `ART` span such as « portrait
de Charlotte Delavault » properly contains a `PER` span « Charlotte Delavault »
— so a prediction that correctly recovers the inner mention necessarily overlaps
two gold spans. Segmentation errors are therefore counted only when the gold
spans a prediction covers are mutually disjoint; scoring the naive overlap
instead inflates the merge count roughly fourfold, and any evaluation over a
nested schema has to make this choice explicit. Third, the taxonomy measures the
*number* of actions, not their duration; the mechanical/cognitive split is a
claim about the kind of labour each action requires, grounded in the repair
procedure, not a measured time cost. A timed post-correction study on a
subsample would convert these counts into minutes and is the obvious next step.
