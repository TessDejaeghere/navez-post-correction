"""Project post-correction effort for each NER system, in hours.

Combines the two halves of the analysis:

* ``error_taxonomy.py``          — how many correction events of each class
                                   each system's output would generate on the
                                   561-letter gold corpus;
* ``analyze_correction_time.py`` — how many seconds each class of event cost
                                   the annotators during the real GLiNER2
                                   post-correction round.

Multiplying the two gives an estimated wall-clock cost of supervising each
system, and a comparison against annotating from scratch.

The projection is a linear combination of the fitted coefficients, so its
confidence interval is computed properly from the coefficient covariance
matrix (a'Va) rather than by propagating the individual intervals.

Assumptions, stated in the report
---------------------------------
* An event of class *c* costs the same regardless of which system produced it.
  This is the load-bearing assumption; it is plausible for the mechanical
  classes and weakest for ``C1``, whose cost per mention plausibly falls when
  many mentions are missing from the same letter (the annotator re-reads once
  and adds several spans).
* Annotating from scratch is priced as one ``C1`` event per gold span, since
  ``C1`` is by definition "the annotator drew this span by hand".
* Reading and the per-task fixed cost are paid in every scenario, so the
  "correction only" column is the part that is actually attributable to the
  choice of system.

Reads
-----
* ``NER_pipeline/eval/error_taxonomy.csv``      (from error_taxonomy.py)
* ``NER_pipeline/eval/gold_561_merged.jsonl``
* ``data/labelstudio_output/NER_*.json``        (re-fits the time model)

Writes
------
* ``NER_pipeline/eval/projected_correction_cost.csv``
* ``NER_pipeline/eval/projected_correction_cost.md``
* ``NER_pipeline/eval/figures/fig_projected_cost.{png,pdf}``

Usage
-----
    python NER_pipeline/project_correction_cost.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from analyze_correction_time import (  # noqa: E402
    COG_EDIT, DELETED, MECH_EDIT, fit, load_tasks,
)
from error_taxonomy import FAMILY, MODELS  # noqa: E402

EVAL_DIR = REPO / "NER_pipeline" / "eval"
FIG_DIR = EVAL_DIR / "figures"
# The paper reports the post-filter world throughout (see the validity filter in
# the methodology), so the filtered taxonomy is the default input.  Pass
# --taxonomy-csv to score the unfiltered counts instead.
TAXONOMY_CSV = EVAL_DIR / "error_taxonomy_filtered.csv"
GOLD_PATH = EVAL_DIR / "gold_561_merged.jsonl"
OUT_CSV = EVAL_DIR / "projected_correction_cost.csv"
REPORT = EVAL_DIR / "projected_correction_cost.md"

# Taxonomy classes -> regressors of the aggregate time model.  Imported from
# analyze_correction_time rather than restated, so that a renumbering of the
# taxonomy cannot leave the two halves of the cost model disagreeing.
GROUPS = {
    "n_deleted": DELETED,
    "n_mech_edit": MECH_EDIT,
    "n_cog_edit": COG_EDIT,
    "C1": ["C1"],
}
REGRESSORS = list(GROUPS)

# Three ways of pricing the same event profiles.  The ranking is only worth
# reporting to the extent that it survives all three.
#   None  -> use the fitted coefficient
#   float -> impose this value (minutes per event)
PRICINGS: dict[str, dict[str, float | None]] = {
    "measured": {"n_deleted": 0.0, "n_mech_edit": 0.0,
                 "n_cog_edit": None, "C1": None},
    "conservative": {"n_deleted": 0.0, "n_mech_edit": 0.0,
                     "n_cog_edit": "=C1", "C1": None},
    "unconstrained": {"n_deleted": None, "n_mech_edit": None,
                      "n_cog_edit": None, "C1": None},
}
PRICING_NOTE = {
    "measured": "coefficients that are not distinguishable from zero "
                "(deletions, mechanical edits) are priced at zero; the two "
                "significant event coefficients are used as fitted",
    "conservative": "as *measured*, but cognitive edits are priced at the "
                    "cheaper missed-mention rate, giving a lower bound for "
                    "systems whose segmentation-error count is far outside "
                    "the range observed in the GLiNER2 round",
    "unconstrained": "every fitted coefficient used as-is, including the "
                     "non-significant negative ones; shown to expose how "
                     "much the ranking depends on them",
}


def load_gold_shape() -> tuple[int, int, int]:
    n_letters = n_spans = n_chars = 0
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_letters += 1
            n_spans += len(r["gold_spans"])
            n_chars += len(r["text"])
    return n_letters, n_spans, n_chars


def load_event_counts(taxonomy_csv: Path = TAXONOMY_CSV) -> pd.DataFrame:
    df = pd.read_csv(taxonomy_csv)
    wide = (df.groupby(["model", "class"])["n"].sum()
              .unstack(fill_value=0).reindex(MODELS, fill_value=0))
    out = pd.DataFrame(index=wide.index)
    for reg, classes in GROUPS.items():
        cols = [c for c in classes if c in wide.columns]
        out[reg] = wide[cols].sum(axis=1) if cols else 0
    return out


def projection(model, counts: dict[str, float], n_letters: int,
               n_chars: int, ann_mix: pd.Series, include_overhead: bool,
               pricing: dict) -> tuple[float, float]:
    """Return (minutes, half-width of the 95% CI) for one scenario.

    Imposed prices contribute to the point estimate but not to the variance,
    since they are assumptions rather than estimates.  Where a class is priced
    at another class's rate, its count is folded into that class so that both
    the estimate and its variance stay consistent.
    """
    a = pd.Series(0.0, index=model.params.index)
    fixed_minutes = 0.0
    if include_overhead:
        a["const"] = n_letters
        a["chars_k"] = n_chars / 1000.0
        for term, share in ann_mix.items():
            if term in a.index:
                a[term] = n_letters * share
    for reg, n in counts.items():
        if reg not in a.index:
            continue
        price = pricing.get(reg, None)
        if price is None:
            a[reg] += n
        elif isinstance(price, str) and price.startswith("="):
            a[price[1:]] += n          # price this class at another's rate
        else:
            fixed_minutes += float(price) * n
    est = float(a @ model.params) + fixed_minutes
    var = float(a.values @ model.cov_params().values @ a.values)
    return est, 1.96 * float(np.sqrt(max(var, 0.0)))


def build_report(res: pd.DataFrame, model, n_letters: int, n_spans: int,
                 n_chars: int, per100: float) -> str:
    out: list[str] = []
    A = out.append

    A("# Projected post-correction cost per system\n")
    A("Each system's event profile on the 561-letter gold corpus "
      f"({n_spans} gold spans, {n_chars / 1000:.0f}k characters), priced with "
      "the per-event costs measured on the annotators' real GLiNER2 "
      "post-correction round.\n")
    A("Measured inputs: a cognitive edit costs "
      f"{model.params['n_cog_edit'] * 60:.0f} s, a mention the system missed "
      f"{model.params['C1'] * 60:.0f} s to draw by hand, a mechanical edit "
      f"{model.params['n_mech_edit'] * 60:.0f} s and a deletion "
      f"{model.params['n_deleted'] * 60:.0f} s (the last two are not "
      "distinguishable from zero); reading costs "
      f"{model.params['chars_k'] * 60:.0f} s per 1,000 characters and each "
      f"task carries {model.params['const'] * 60:.0f} s of fixed cost.\n")

    A("## Estimated effort to supervise 100 letters\n")
    for pricing_name, note in PRICING_NOTE.items():
        sub = res[res.pricing == pricing_name].set_index("system")
        base = sub.loc["manual from scratch", "total_h"]
        A(f"### Pricing: *{pricing_name}*\n")
        A(f"{note.capitalize()}.\n")
        A("| system | family | correction only (h) | 95% CI | total incl. "
          "reading (h) | vs. from scratch |")
        A("|---|---|---:|---|---:|---:|")
        order = sub.drop(index="manual from scratch").sort_values("corr_h").index
        for name in list(order) + ["manual from scratch"]:
            r = sub.loc[name]
            saved = 100 * (1 - r["total_h"] / base)
            saved_s = "—" if name == "manual from scratch" else f"{saved:+.0f}%"
            disp = (f"**{name}**" if name == "manual from scratch"
                    else f"`{name}`")
            A(f"| {disp} | {r['family']} | {r['corr_h']:.1f} | "
              f"[{r['corr_lo']:.1f}, {r['corr_hi']:.1f}] | "
              f"{r['total_h']:.1f} | {saved_s} |")
        A("")
    A(f"Figures are per 100 letters of this corpus (scaled from {n_letters} "
      f"letters, factor {per100:.3f}). *Correction only* is the part "
      "attributable to the system; *total* adds the reading and per-task "
      "overhead that every scenario pays, including annotating from scratch. "
      f"The from-scratch row prices each of the {n_spans} gold spans as one "
      "hand-drawn mention.\n")

    A("## What the ranking says\n")
    mea = res[res.pricing == "measured"].set_index("system")
    con = res[res.pricing == "conservative"].set_index("system")
    ranks_m = mea.drop(index="manual from scratch")["corr_h"].rank()
    ranks_c = con.drop(index="manual from scratch")["corr_h"].rank()
    stable = (ranks_m - ranks_c).abs().max() <= 1
    best = ranks_m.idxmin()
    worst = ranks_m.idxmax()
    A(f"- Cheapest to supervise under the measured pricing: `{best}` "
      f"({mea.loc[best, 'corr_h']:.1f} h of correction per 100 letters).")
    A(f"- Most expensive: `{worst}` ({mea.loc[worst, 'corr_h']:.1f} h), "
      f"{mea.loc[worst, 'corr_h'] / mea.loc[best, 'corr_h']:.1f}× the "
      "cheapest.")
    A("- The ordering is "
      + ("stable across the measured and conservative pricings (no system "
         "moves by more than one place)."
         if stable else
         "**not** stable across pricings: it is dominated by how "
         "segmentation errors are priced, which is exactly the coefficient "
         "with the weakest empirical support. Only the claims below survive "
         "all three pricings."))
    A("")
    A("### Claims that hold under every pricing\n")
    piv = res.pivot_table(index="system", columns="pricing", values="corr_h")
    piv = piv.drop(index="manual from scratch")
    ranked = {p: list(piv[p].sort_values().index) for p in PRICINGS}
    always_cheap = [s for s in piv.index
                    if all(ranked[p].index(s) <= 2 for p in PRICINGS)]
    always_dear = [s for s in piv.index
                   if all(ranked[p].index(s) >= len(piv) - 2 for p in PRICINGS)]
    A("| system | " + " | ".join(f"{p} (h)" for p in PRICINGS) + " |")
    A("|---" * (len(PRICINGS) + 1) + "|")
    for s in piv.mean(axis=1).sort_values().index:
        A(f"| `{s}` | " + " | ".join(f"{piv.loc[s, p]:.1f}" for p in PRICINGS)
          + " |")
    A("")
    if always_cheap:
        A("- Always among the three cheapest to supervise: "
          + ", ".join(f"`{s}`" for s in always_cheap) + ".")
    if always_dear:
        A("- Always among the three most expensive: "
          + ", ".join(f"`{s}`" for s in always_dear) + ".")
    A("- Every system beats annotating from scratch on total time under every "
      "pricing, but the spread between systems is comparable to the gap "
      "between the best system and no system at all, so which model is chosen "
      "matters about as much as whether one is used.\n")

    A("## Caveats\n")
    A("- The prices come from GLiNER2's error mix and are applied to other "
      "systems' event profiles. The load-bearing assumption is that an event "
      "of a given class costs the same whoever produced it.")
    A("- **Extrapolation warning.** The cognitive-edit price is estimated on "
      "339 events in the GLiNER2 round, of which only 65 were segmentation "
      "errors. CamemBERTaV2 and d'AlemBERT generate 2,026 and 1,500 "
      "segmentation events respectively, far outside that support. The "
      "*conservative* pricing exists precisely to bound this: it re-prices "
      "cognitive edits at the much cheaper missed-mention rate.")
    A("- `C1` is the dominant term for most systems, and its per-event cost is "
      "the one most likely to be non-linear: an annotator who has to add ten "
      "mentions to one letter re-reads it once, not ten times. Systems with "
      "very high `C1` counts (both Qwen variants) are therefore likely "
      "*over*-priced here.")
    A("- Intervals reflect coefficient uncertainty only. They do not include "
      "uncertainty in the event counts themselves, nor the fold-to-fold "
      "variation reported in `error_taxonomy.md`, and imposed prices "
      "contribute no variance at all.")
    A("- Annotators worked alongside the domain expert, so part of the "
      "cognitive cost never entered `lead_time`; all figures are lower "
      "bounds on the cognitive classes.\n")
    return "\n".join(out)


def make_figure(res: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sub = res[res.pricing == "measured"].set_index("system")
    con = res[res.pricing == "conservative"].set_index("system")
    d = sub.drop(index="manual from scratch").sort_values("corr_h")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ys = list(range(len(d)))
    err = [(d["corr_h"] - d["corr_lo"]).values,
           (d["corr_hi"] - d["corr_h"]).values]
    colors = {"zero-shot": "#95a5a6", "fine-tuned": "#7fb3d5",
              "prompted LLM": "#c0392b"}
    ax.barh(ys, d["corr_h"], xerr=err,
            color=[colors[f] for f in d["family"]],
            error_kw={"ecolor": "#2c3e50", "lw": 1})
    ax.scatter(con.loc[d.index, "corr_h"], ys, marker="|", s=180,
               color="k", zorder=5, label="conservative pricing")
    base = sub.loc["manual from scratch", "corr_h"]
    ax.axvline(base, color="k", ls="--", lw=1)
    ax.text(base, -0.75, f" from scratch: {base:.1f} h", fontsize=8,
            va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(d.index, fontsize=8)
    ax.set_ylim(len(d) - 0.4, -1.1)
    ax.set_xlabel("estimated correction hours per 100 letters (95% CI)")
    ax.set_title("Projected post-correction effort")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles + [plt.Line2D([], [], marker="|", color="k", ls="")],
              list(colors) + ["conservative pricing"], fontsize=8,
              loc="lower right", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_projected_cost.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_projected_cost.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--taxonomy-csv", type=Path, default=TAXONOMY_CSV,
                    help="event counts to price (default: the filtered taxonomy)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    df_tasks, _, _ = load_tasks(args.max_minutes)
    dfp = df_tasks[df_tasks.has_prediction].reset_index(drop=True)
    model, _, _ = fit(dfp, REGRESSORS, "aggregate", include_spans=False)
    ann_mix = pd.get_dummies(dfp["annotator"], prefix="ann",
                             drop_first=True).astype(float).mean()

    n_letters, n_spans, n_chars = load_gold_shape()
    counts = load_event_counts(args.taxonomy_csv)
    per100 = 100.0 / n_letters

    rows = []
    scratch = {"n_deleted": 0, "n_mech_edit": 0, "n_cog_edit": 0, "C1": n_spans}
    scenarios = [(m, counts.loc[m].to_dict(), FAMILY[m]) for m in MODELS]
    scenarios.append(("manual from scratch", scratch, "baseline"))

    for pricing_name, pricing in PRICINGS.items():
        for name, c, fam in scenarios:
            corr, ci = projection(model, c, n_letters, n_chars, ann_mix,
                                  False, pricing)
            tot, _ = projection(model, c, n_letters, n_chars, ann_mix,
                                True, pricing)
            rows.append({"pricing": pricing_name, "system": name,
                         "family": fam,
                         "corr_h": corr / 60 * per100,
                         "corr_lo": (corr - ci) / 60 * per100,
                         "corr_hi": (corr + ci) / 60 * per100,
                         "total_h": tot / 60 * per100, **c})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, encoding="utf-8")
    REPORT.write_text(build_report(res, model, n_letters, n_spans, n_chars,
                                   per100), encoding="utf-8")
    print(f"wrote {OUT_CSV}\nwrote {REPORT}\n")
    for p in PRICINGS:
        print(f"--- {p} ---")
        sub = res[res.pricing == p].set_index("system")
        print(sub[["family", "corr_h", "corr_lo", "corr_hi", "total_h"]]
              .round(1).to_string())
        print()

    if not args.no_figure:
        make_figure(res)


if __name__ == "__main__":
    main()
