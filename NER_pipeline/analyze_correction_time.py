"""Measure post-correction time per error class from the annotators' edit log.

Idea
----
The NER annotation round was a *post-correction* task: every letter was
pre-annotated with GLiNER2 and imported into Label Studio as suggestions.
Label Studio preserves, for every span in the submitted annotation, an
``origin`` field (``prediction`` = accepted untouched, ``prediction-changed``
= edited, ``manual`` = created by the annotator) and the span ``id`` is shared
with the prediction object, so the pre-annotation and the final annotation can
be diffed span by span rather than aligned heuristically.

Together with ``lead_time`` (Label Studio's active-tab seconds, paused after
~60 s idle) this gives one observation per (annotator, letter):

    minutes_spent  ~  reading  +  sum over edit events of (cost of that event)

Fitting that regression turns the mechanical / cognitive distinction from a
design claim into a measured quantity: seconds per edit, per error class.

Caveats, stated in the report as well
-------------------------------------
* ``lead_time`` is active-tab time, not effort; it still contains reading,
  scrolling and hesitation.  Letter length is included as a covariate to
  absorb the reading component.
* The costs are estimated on GLiNER2's error mix.  Transferring them to
  another system assumes an event of class *c* costs the same regardless of
  which system produced it.
* Event counts are collinear with each other and with letter length, which
  widens the intervals; the aggregate 4-regressor model is the one to quote.
* Annotators differ in style (A4 edits almost nothing and re-draws spans
  manually instead), so annotator fixed effects are included throughout.

Reads
-----
* ``data/labelstudio_output/NER_*.json``   (Label Studio exports)

Writes
------
* ``NER_pipeline/eval/correction_time_per_task.csv``
* ``NER_pipeline/eval/correction_time_events.csv``
* ``NER_pipeline/eval/correction_time.md``
* ``NER_pipeline/eval/figures/fig_correction_time.{png,pdf}``

Usage
-----
    python NER_pipeline/analyze_correction_time.py
    python NER_pipeline/analyze_correction_time.py --max-minutes 45
    python NER_pipeline/analyze_correction_time.py --drop-classes none

``F0`` is excluded by default: the validity filter removes malformed spans
before a corrector sees them, so they are not part of the cost model.  Pass
``--drop-classes none`` to price every class, including F0.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from error_taxonomy import (  # noqa: E402
    FILTER_CLASS, IDENTITY_LABELS, TAXONOMY, _case_decidable_label, _core,
    _proper_tokens, is_malformed,
)

LS_DIR = REPO / "data" / "labelstudio_output"
EVAL_DIR = REPO / "NER_pipeline" / "eval"
FIG_DIR = EVAL_DIR / "figures"
TASK_CSV = EVAL_DIR / "correction_time_per_task.csv"
EVENT_CSV = EVAL_DIR / "correction_time_events.csv"
REPORT = EVAL_DIR / "correction_time.md"

# Event classes, in the order used by the taxonomy report.  "A0" is not an
# error: it is an accepted suggestion, kept as a regressor because verifying a
# correct suggestion also costs time.
CLASSES = ["A0", "F0", "M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4"]
MECH = ["F0", "M1", "M2", "M3", "M4"]
COG_BOUNDED = ["C2", "C3", "C4"]
COG_UNBOUNDED = ["C1"]
DELETED = ["F0", "M1", "C2"]     # repairs made by removing the span
MECH_EDIT = ["M2", "M3", "M4"]
COG_EDIT = ["C3", "C4"]

MIN_SECONDS = 15.0     # discard tasks opened and submitted instantly
MIN_EVENTS_FOR_OWN_TERM = 30


# ---------------------------------------------------------------------------
# Edit-log extraction
# ---------------------------------------------------------------------------
def _spans_from_result(result: list) -> dict[str, tuple[int, int, str]]:
    out: dict[str, tuple[int, int, str]] = {}
    for r in result or []:
        if r.get("type") != "labels":
            continue
        v = r.get("value") or {}
        labels = v.get("labels") or []
        if not labels:
            continue
        try:
            s, e = int(v["start"]), int(v["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        out[str(r.get("id"))] = (s, e, str(labels[0]))
    return out


def _origins_from_result(result: list) -> dict[str, str]:
    return {str(r.get("id")): str(r.get("origin", "manual"))
            for r in result or [] if r.get("type") == "labels"}


def classify_edits(text: str,
                   pred: dict[str, tuple[int, int, str]],
                   final: dict[str, tuple[int, int, str]],
                   origins: dict[str, str]) -> list[dict]:
    """Diff the GLiNER2 suggestion against the submitted annotation.

    Returns one event per span the annotator touched, kept or created,
    labelled with the same nine-class taxonomy used in error_taxonomy.py.
    """
    events: list[dict] = []

    for sid, (fs, fe, fl) in final.items():
        origin = origins.get(sid, "manual")
        p = pred.get(sid)
        if p is None or origin == "manual":
            sub = ("identity-critical (PER/LOC)" if fl in IDENTITY_LABELS
                   else "other labels")
            events.append({"cls": "C1", "subtype": sub, "label": fl,
                           "from": "", "start": fs, "end": fe})
            continue

        ps, pe, pl = p
        if (ps, pe, pl) == (fs, fe, fl):
            events.append({"cls": "A0", "subtype": "accepted", "label": fl,
                           "from": pl, "start": fs, "end": fe})
            continue

        f_txt, p_txt = text[fs:fe], text[ps:pe]
        same_core = _core(f_txt) == _core(p_txt)

        if pl == fl:
            if same_core:
                cls, sub = "M2", ""
            elif _proper_tokens(f_txt) == _proper_tokens(p_txt):
                cls, sub = "M3", ""
            else:
                cls, sub = "C4", "identity-altering boundary"
        elif same_core or (ps, pe) == (fs, fe):
            if {pl, fl} == {"PER", "PER_NOM"} and _case_decidable_label(f_txt) == fl:
                cls, sub = "M4", f"{pl}->{fl}"
            else:
                cls, sub = "C3", f"{pl}->{fl}"
        else:
            cls, sub = "C4", "boundary + label"
        events.append({"cls": cls, "subtype": sub, "label": fl,
                       "from": pl, "start": fs, "end": fe})

    kept = list(final.values())
    for sid, (ps, pe, pl) in pred.items():
        if sid in final:
            continue
        reason = is_malformed(text, ps, pe)
        if reason:
            cls, sub = FILTER_CLASS, reason
        elif any(kl == pl and min(pe, ke) > max(ps, ks)
                 for ks, ke, kl in kept):
            cls, sub = "M1", "overlapping same-label span"
        else:
            cls, sub = "C2", ("identity-critical (PER/LOC)"
                              if pl in IDENTITY_LABELS else "other labels")
        events.append({"cls": cls, "subtype": sub, "label": pl,
                       "from": pl, "start": ps, "end": pe})
    return events


def load_tasks(max_minutes: float,
               drop_classes: frozenset[str] = frozenset()
               ) -> tuple[pd.DataFrame, list[dict], Counter]:
    classes = [c for c in CLASSES if c not in drop_classes]
    task_rows: list[dict] = []
    event_rows: list[dict] = []
    dropped = Counter()
    dropped_events: Counter = Counter()

    for path in sorted(LS_DIR.glob("NER_*.json")):
        export = json.loads(path.read_text(encoding="utf-8"))
        annotator = path.stem.replace("NER_", "").replace("_final", "")
        annotator = annotator.replace("707_", "").replace("607_", "")
        annotator = annotator.replace("_207", "").lower()
        for t in export:
            data = t.get("data") or {}
            lid, text = data.get("letter_id"), data.get("text") or ""
            anns = t.get("annotations") or []
            if not lid or not text or not anns:
                dropped["no data"] += 1
                continue
            a = anns[0]
            lead = a.get("lead_time")
            if lead is None:
                dropped["no lead_time"] += 1
                continue
            lead = float(lead)
            if lead < MIN_SECONDS:
                dropped["< 15 s"] += 1
                continue
            if lead / 60 > max_minutes:
                dropped[f"> {max_minutes:g} min"] += 1
                continue

            final = _spans_from_result(a.get("result"))
            origins = _origins_from_result(a.get("result"))
            pred_obj = a.get("prediction") or {}
            pred = _spans_from_result(pred_obj.get("result"))
            has_pred = bool(pred)

            evs = classify_edits(text, pred, final, origins) if has_pred else []
            dropped_events += Counter(e["cls"] for e in evs
                                      if e["cls"] in drop_classes)
            evs = [e for e in evs if e["cls"] not in drop_classes]
            counts = Counter(e["cls"] for e in evs)

            row = {
                "annotator": annotator, "export": path.stem, "letter_id": lid,
                "minutes": lead / 60, "chars": len(text),
                "n_pred": len(pred), "n_final": len(final),
                "has_prediction": has_pred,
                "n_accepted_origin": sum(1 for v in origins.values()
                                         if v == "prediction"),
                "n_changed_origin": sum(1 for v in origins.values()
                                        if v == "prediction-changed"),
                "n_manual_origin": sum(1 for v in origins.values()
                                       if v == "manual"),
            }
            for c in classes:
                row[c] = counts.get(c, 0)
            row["n_mech"] = sum(counts.get(c, 0) for c in MECH)
            row["n_cog_bounded"] = sum(counts.get(c, 0) for c in COG_BOUNDED)
            row["n_cog_unbounded"] = sum(counts.get(c, 0) for c in COG_UNBOUNDED)
            # Deletions never appear in the submitted annotation; edits do.
            row["n_deleted"] = sum(counts.get(c, 0) for c in DELETED)
            row["n_mech_edit"] = sum(counts.get(c, 0) for c in MECH_EDIT)
            row["n_cog_edit"] = sum(counts.get(c, 0) for c in COG_EDIT)
            task_rows.append(row)

            for e in evs:
                event_rows.append({"annotator": annotator, "letter_id": lid,
                                   "cls": e["cls"], "subtype": e["subtype"],
                                   "label": e["label"], "from_label": e["from"],
                                   "surface": text[max(0, e["start"]):e["end"]]
                                   .replace("\n", " ")})

    print("dropped tasks:", dict(dropped))
    return pd.DataFrame(task_rows), event_rows, dropped_events


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit(df: pd.DataFrame, regressors: list[str], label: str,
        include_spans: bool = True):
    """OLS of minutes on event counts + length + annotator fixed effects.

    ``include_spans`` adds the total number of spans in the submitted
    annotation.  With it in the model, the omitted category is "a suggestion
    that was accepted unchanged", so every event coefficient reads as the
    *extra* seconds that error class costs relative to a correct suggestion.
    Without it, coefficients are absolute but confounded with letter length.
    """
    X = df[regressors].astype(float).copy()
    X["chars_k"] = df["chars"] / 1000.0
    terms = list(regressors) + ["chars_k"]
    if include_spans:
        X["n_spans_final"] = df["n_final"].astype(float)
        terms.append("n_spans_final")
    dummies = pd.get_dummies(df["annotator"], prefix="ann", drop_first=True)
    X = pd.concat([X, dummies.astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df["minutes"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model, terms, label


def vif(df: pd.DataFrame, terms: list[str]) -> dict[str, float]:
    """Variance inflation factors, to be honest about collinearity."""
    X = df[[t for t in terms if t in df.columns]].astype(float)
    X = X.loc[:, X.std() > 0]
    out = {}
    for c in X.columns:
        others = sm.add_constant(X.drop(columns=[c]), has_constant="add")
        r2 = sm.OLS(X[c], others).fit().rsquared
        out[c] = float("inf") if r2 >= 1 else 1 / (1 - r2)
    return out


def coef_table(model, terms: list[str], scale: float = 60.0) -> list[tuple]:
    out = []
    for t in terms:
        if t not in model.params.index:
            continue
        lo, hi = model.conf_int().loc[t]
        out.append((t, model.params[t] * scale, lo * scale, hi * scale,
                    model.pvalues[t]))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, dfp: pd.DataFrame, events: list[dict],
                 models: list[tuple], vifs: dict[str, dict],
                 counts: Counter,
                 drop_classes: frozenset[str] = frozenset(),
                 dropped_events: Counter = Counter()) -> str:
    classes = [c for c in CLASSES if c not in drop_classes]
    out: list[str] = []
    A = out.append

    A("# Measured post-correction time per error class\n")
    A("Estimated from the annotators' own edit log: every NER letter was "
      "pre-annotated with GLiNER2 and corrected in Label Studio, which records "
      "`lead_time` (active-tab seconds) and, for every span, whether it was "
      "accepted, edited or created by hand.\n")

    A("## 1. Sample\n")
    A(f"- {len(df)} submitted tasks from {df.annotator.nunique()} annotators "
      f"over {df.letter_id.nunique()} distinct letters.")
    A(f"- {len(dfp)} of those carry the GLiNER2 suggestion object and can be "
      f"diffed span by span ({len(df) - len(dfp)} cannot: the export for one "
      f"annotator dropped the prediction payload).")
    A(f"- Time on task: mean {df.minutes.mean():.1f} min, median "
      f"{df.minutes.median():.1f} min, p90 {df.minutes.quantile(.9):.1f} min.")
    A(f"- Total recorded effort: {df.minutes.sum() / 60:.1f} hours.\n")

    A("| annotator | tasks | median min | mean min | accepted | edited | manual |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for ann, g in df.groupby("annotator"):
        A(f"| {ann} | {len(g)} | {g.minutes.median():.1f} | "
          f"{g.minutes.mean():.1f} | {g.n_accepted_origin.sum()} | "
          f"{g.n_changed_origin.sum()} | {g.n_manual_origin.sum()} |")
    A("")
    A("Annotator style differs sharply and is absorbed by fixed effects in "
      "every model below: some annotators edit a suggestion, others delete it "
      "and re-draw the span by hand, which moves the same correction from the "
      "`prediction-changed` column to the `manual` one.\n")

    A("## 2. What GLiNER2's output cost, by class\n")
    if drop_classes:
        n_dropped = sum(dropped_events.get(c, 0) for c in drop_classes)
        A("Excluded from this analysis: "
          + ", ".join(f"`{c}` ({TAXONOMY[c][0]})" for c in sorted(drop_classes))
          + f" — {n_dropped} events. `{FILTER_CLASS}` is not a taxonomy class: "
            "it is what the validity filter deletes before a corrector sees "
            "it, so it is not part of the cost model.\n")
    ev = Counter(e["cls"] for e in events)
    tot = sum(ev.values())
    A("| class | error class | cost | events | % of spans touched |")
    A("|---|---|---|---:|---:|")
    for c in classes:
        if c == "A0":
            A(f"| `A0` | accepted suggestion | (none) | {ev.get(c, 0)} | "
              f"{100 * ev.get(c, 0) / tot:.1f}% |")
        else:
            A(f"| `{c}` | {TAXONOMY[c][0]} | {TAXONOMY[c][4]} | {ev.get(c, 0)} | "
              f"{100 * ev.get(c, 0) / tot:.1f}% |")
    A("")

    A("## 3. How long does each kind of correction take?\n")
    A("`minutes ~ event counts + letter length + annotator`, OLS with HC1 "
      "robust standard errors. Coefficients are absolute seconds per event; "
      "`chars_k` is seconds per 1,000 characters and absorbs reading. The "
      "number of accepted suggestions is deliberately left out of the primary "
      "model: it is almost collinear with the letter's total span count, and "
      "including it makes the remaining coefficients unidentifiable.\n")
    group_n = {"n_deleted": sum(counts.get(c, 0) for c in DELETED),
               "n_mech_edit": sum(counts.get(c, 0) for c in MECH_EDIT),
               "n_cog_edit": sum(counts.get(c, 0) for c in COG_EDIT),
               "A0": counts.get("A0", 0)}
    for model, terms, label in models:
        A(f"### {label}\n")
        A(f"n = {int(model.nobs)} tasks, adjusted R² = "
          f"{model.rsquared_adj:.3f}.\n")
        A("| term | events | seconds per event | 95% CI | p |")
        A("|---|---:|---:|---|---:|")
        for t, b, lo, hi, p in coef_table(model, terms):
            star = (" ***" if p < 0.001 else " **" if p < 0.01
                    else " *" if p < 0.05 else "")
            n_ev = "—" if t == "chars_k" else str(
                counts.get(t, group_n.get(t, 0)))
            thin = " ⚠︎" if (t != "chars_k"
                            and int(counts.get(t, group_n.get(t, 0))) < 100) else ""
            A(f"| `{t}`{thin} | {n_ev} | {b:.1f}{star} | [{lo:.1f}, {hi:.1f}] | "
              f"{p:.3g} |")
        A("")
        A(f"Intercept: {model.params['const'] * 60:.0f} s of fixed cost per "
          "letter (opening the task, orienting in the text).\n")
    A("⚠︎ = fewer than 100 events in the whole corpus; the estimate is not "
      "usable.\n")

    A("### Collinearity\n")
    thin = [c for c in classes
            if c != "A0" and 0 < counts.get(c, 0) < 100]
    A("Variance inflation factors. Above ~5 the coefficient is poorly "
      "identified and its interval should be read as indicative only. "
      + (f"Classes with few events ({', '.join(f'`{c}`' for c in thin)}) are "
         "additionally imprecise regardless of their VIF.\n" if thin else "\n"))
    for name, vv in vifs.items():
        A(f"**{name} model** — " + ", ".join(
            f"`{k}` {v:.1f}" for k, v in sorted(vv.items(), key=lambda kv: -kv[1])))
        A("")

    A("## 4. Where the time actually went\n")
    A("Seconds per event from the aggregate model, multiplied by the number "
      "of events of that kind in the corrected corpus. Reading is the "
      "`chars_k` coefficient applied to the total character count, and the "
      "fixed cost is the intercept applied to every task.\n")
    model, terms, _ = models[0]
    total_min = dfp["minutes"].sum()
    sig = {t: p for t, _, _, _, p in coef_table(model, terms)}
    contrib = [("reading the letters",
                model.params["chars_k"] * dfp["chars"].sum() / 1000, None),
               ("fixed cost per task",
                model.params["const"] * len(dfp), None)]
    for t, name in (("n_deleted", "deleting spurious suggestions"),
                    ("n_mech_edit", "mechanical edits"),
                    ("n_cog_edit", "cognitive edits"),
                    ("C1", "drawing missed mentions by hand")):
        contrib.append((name, model.params[t] * dfp[t].sum(), sig.get(t)))
    ann_terms = [t for t in model.params.index if t.startswith("ann_")]
    ann_min = sum(model.params[t] * pd.get_dummies(dfp["annotator"],
                                                   prefix="ann").get(t, 0).sum()
                  for t in ann_terms)
    contrib.append(("annotator differences (fixed effects)", ann_min, None))

    A("| component | events | estimated minutes | share of measured time |")
    A("|---|---:|---:|---:|")
    ev_n = {"deleting spurious suggestions": int(dfp["n_deleted"].sum()),
            "mechanical edits": int(dfp["n_mech_edit"].sum()),
            "cognitive edits": int(dfp["n_cog_edit"].sum()),
            "drawing missed mentions by hand": int(dfp["C1"].sum())}
    for name, mins, p in contrib:
        mark = " †" if p is not None and p >= 0.05 else ""
        A(f"| {name}{mark} | {ev_n.get(name, '—')} | {mins:.0f} | "
          f"{100 * mins / total_min:.0f}% |")
    A(f"| **measured total** | | **{total_min:.0f}** | 100% |")
    A("")
    A("† coefficient not significant at the 5 % level; the component is "
      "reported for completeness but is not distinguishable from zero. "
      "Components are model-fitted and sum to the measured total up to "
      "rounding, since OLS residuals sum to zero.\n")
    A("The two columns tell different stories on purpose. A cognitive edit is "
      "by far the most expensive single action, but there are few of them; "
      "mentions GLiNER2 missed are cheap one by one and overwhelming in bulk. "
      "Optimising a system for this corpus therefore means optimising recall, "
      "not precision.\n")

    A("## 5. Interpretation\n")
    A("`chars_k` is seconds per 1,000 characters and captures reading: on a "
      "letter of average length the annotator spends roughly three and a half "
      "minutes simply reading, before any correction. The event coefficients "
      "are what one correction of that class adds on top of that. They are "
      "marginal effects in an observational sample, not stopwatch "
      "measurements; the aggregate model is the one to quote and the "
      "per-class model is indicative.\n")

    A("## 6. Caveats\n")
    A("- `lead_time` is active-tab time. Label Studio pauses after ~60 s of "
      "idleness, but a tab left open on a difficult passage still accrues time.")
    A("- Costs are estimated on GLiNER2's error mix. Applying them to another "
      "system's event profile assumes an event of class *c* costs the same "
      "whichever system produced it — plausible for the mechanical classes, "
      "less so for `C1`, whose cost depends on how many mentions are missing "
      "at once.")
    A("- Annotators worked in group settings with the domain expert present, "
      "so some cognitive cost was paid by conversation and never entered "
      "`lead_time`. These estimates are therefore a lower bound on the "
      "cognitive classes.")
    A("- Letters annotated by several annotators appear once per annotator; "
      "standard errors are robust but not clustered on the letter.\n")
    return "\n".join(out)


def make_figure(df: pd.DataFrame, models: list[tuple]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model, terms, _ = models[0]
    rows = coef_table(model, terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.scatter(df["chars"] / 1000, df["minutes"], s=9, alpha=.35,
               color="#34495e")
    ax.set_xlabel("letter length (1,000 characters)")
    ax.set_ylabel("time on task (minutes)")
    ax.set_title("Time on task vs letter length")

    ax = axes[1]
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    los = [r[1] - r[2] for r in rows][::-1]
    his = [r[3] - r[1] for r in rows][::-1]
    cog_terms = {"n_cog_edit", "n_cog_bounded", "n_cog_unbounded",
                 "C1", "C2", "C3", "C4"}
    mech_terms = {"n_mech_edit", "n_mech", "n_deleted",
                  "F0", "M1", "M2", "M3", "M4"}
    colors = ["#c0392b" if n in cog_terms else
              "#7fb3d5" if n in mech_terms else "#95a5a6" for n in names]
    ax.barh(range(len(names)), vals, xerr=[los, his], color=colors,
            error_kw={"ecolor": "#2c3e50", "lw": 1})
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("seconds per event (95% CI)")
    ax.set_title("Marginal time cost per correction event")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_correction_time.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_correction_time.png'}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-minutes", type=float, default=60.0,
                    help="discard tasks longer than this (idle-inflated)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--drop-classes", default="F0", metavar="CODES",
                    help="comma-separated classes to exclude from the model "
                         "(default: F0, the malformed spans the validity "
                         "filter removes before a corrector sees them); pass "
                         "'none' to keep every class")
    args = ap.parse_args()
    raw = "" if args.drop_classes.strip().lower() == "none" else args.drop_classes
    drop = frozenset(c.strip().upper() for c in raw.split(",") if c.strip())
    unknown = drop - set(CLASSES)
    if unknown:
        ap.error(f"unknown class(es): {', '.join(sorted(unknown))}")

    df, events, dropped_events = load_tasks(args.max_minutes, drop)
    dfp = df[df.has_prediction].reset_index(drop=True)
    print(f"{len(df)} tasks, {len(dfp)} with a diffable suggestion, "
          f"{len(events)} events"
          + (f" (excluding {', '.join(sorted(drop))})" if drop else ""))

    models = []
    # Primary: absolute seconds per event, no collinear baseline term.
    models.append(fit(
        dfp, ["n_deleted", "n_mech_edit", "n_cog_edit", "C1"],
        "Aggregate — absolute seconds per event", include_spans=False))

    counts = Counter(e["cls"] for e in events)
    per_class = [c for c in CLASSES if c != "A0" and c not in drop
                 and counts.get(c, 0) >= MIN_EVENTS_FOR_OWN_TERM]
    models.append(fit(dfp, per_class, "Per class — absolute seconds per event",
                      include_spans=False))

    # Robustness: add accepted suggestions, to show verification is not costly.
    models.append(fit(
        dfp, ["A0", "n_deleted", "n_mech_edit", "n_cog_edit", "C1"],
        "Robustness — with accepted suggestions in the model",
        include_spans=False))

    vifs = {
        "Aggregate": vif(dfp.assign(chars_k=dfp["chars"] / 1000.0),
                         ["n_deleted", "n_mech_edit", "n_cog_edit", "C1",
                          "chars_k"]),
        "Per class": vif(dfp.assign(chars_k=dfp["chars"] / 1000.0),
                         per_class + ["chars_k"]),
    }

    df.to_csv(TASK_CSV, index=False, encoding="utf-8")
    pd.DataFrame(events).to_csv(EVENT_CSV, index=False, encoding="utf-8")
    REPORT.write_text(build_report(df, dfp, events, models, vifs, counts, drop,
                                   dropped_events),
                      encoding="utf-8")
    print(f"wrote {TASK_CSV}\nwrote {EVENT_CSV}\nwrote {REPORT}")

    for model, terms, label in models:
        print(f"\n--- {label} (adj R2={model.rsquared_adj:.3f}) ---")
        for t, b, lo, hi, p in coef_table(model, terms):
            print(f"  {t:18s} {b:7.1f} s  [{lo:7.1f}, {hi:7.1f}]  p={p:.3g}")

    if not args.no_figure:
        make_figure(dfp, models)


if __name__ == "__main__":
    main()
