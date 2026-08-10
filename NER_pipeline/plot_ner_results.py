"""Paper-ready figures for the NER model comparison.

Reads
-----
* ``NER_pipeline/eval/ner_final_metrics.csv`` (from compute_ner_metrics.py)

Writes
------
* ``NER_pipeline/eval/figures/fig_overall_lenient.pdf`` (+ .png)
* ``NER_pipeline/eval/figures/fig_per_label_lenient.pdf`` (+ .png)
* ``NER_pipeline/eval/figures/fig_schema_fit.pdf`` (+ .png)

Design decisions
----------------
* Only lenient F1 in the headline figures — strict F1 penalises the prompted
  LLMs' boundary-guessing in a way that is not the story of the paper; a
  strict/lenient table stays in the CSV for the appendix.
* Colour groups by *training regime* (zero-shot / fine-tuned / prompted LLM),
  not by architecture, because that's the axis the paper argues along.
* No gridlines, no chartjunk, sans-serif, sized for 1-column and full-page
  paper layout.  Vector PDF is authoritative; PNG is for slides.

Usage
-----
    python NER_pipeline/plot_ner_results.py
    python NER_pipeline/plot_ner_results.py --csv NER_pipeline/eval/ner_final_metrics_filtered.csv

When the CSV filename contains a suffix (e.g. `_filtered`), that suffix is
appended to every figure filename so raw and filtered figures coexist.
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "NER_pipeline" / "eval" / "ner_final_metrics.csv"
FIG_DIR = REPO / "NER_pipeline" / "eval" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Display / grouping metadata
# -----------------------------------------------------------------------------
# Regime keys: "zero_shot", "fine_tuned", "prompted_llm"
MODEL_META = {
    "claude-opus-4.8":           {"display": "Claude Opus 4.8",       "regime": "prompted_llm"},
    "qwen3-235b-a22b-2507":      {"display": "Qwen3 235B-A22B",       "regime": "prompted_llm"},
    "qwen3-vl-30b-a3b-instruct": {"display": "Qwen3-VL 30B-A3B",      "regime": "prompted_llm"},
    "dalembert":                 {"display": "d'AlemBERT (ft)",       "regime": "fine_tuned"},
    "camembertav2":              {"display": "CamemBERTaV2 (ft)",     "regime": "fine_tuned"},
    "gliner2-ft":                {"display": "GLiNER2 (ft)",          "regime": "fine_tuned"},
    "gliner2":                   {"display": "GLiNER2 (zero-shot)",   "regime": "zero_shot"},
}
REGIME_COLOR = {
    "zero_shot":    "#a6a6a6",   # neutral grey
    "fine_tuned":   "#1f77b4",   # blue
    "prompted_llm": "#d62728",   # red
}
REGIME_LABEL = {
    "zero_shot":    "Zero-shot",
    "fine_tuned":   "Fine-tuned (in-corpus)",
    "prompted_llm": "Prompted LLM",
}

LABEL_ORDER = ["PER", "LOC", "ORG", "EVT", "ART", "PER_NOM"]  # freq. descending

# -----------------------------------------------------------------------------
# Data loading + aggregation
# -----------------------------------------------------------------------------
def _normal_ci(vals: np.ndarray, z: float = 1.96) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    var = float(np.var(vals, ddof=1))
    return z * sqrt(var / n)


def load_micro(df: pd.DataFrame, mode: str = "lenient") -> pd.DataFrame:
    """Return one row per model with mean F1 and 95% CI across (fold, run=0)."""
    micro = df[(df["mode"] == mode) & (df["label"] == "MICRO")].copy()
    # For fold-variance figures, use one run per fold (run=0).
    micro = micro[micro["run"] == 0]
    micro["F1"] = micro["F1"].astype(float)
    rows = []
    for model, sub in micro.groupby("model"):
        f1s = sub["F1"].to_numpy()
        rows.append({
            "model":    model,
            "n_folds":  len(f1s),
            "mean_F1":  float(f1s.mean()),
            "ci_F1":    _normal_ci(f1s),
            "min_F1":   float(f1s.min()),
            "max_F1":   float(f1s.max()),
        })
    out = pd.DataFrame(rows)
    out["display"] = out["model"].map(lambda m: MODEL_META[m]["display"])
    out["regime"]  = out["model"].map(lambda m: MODEL_META[m]["regime"])
    return out.sort_values("mean_F1", ascending=True).reset_index(drop=True)


def load_micro_both(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per model with mean strict + lenient F1 (and CIs).

    Ordered ascending by lenient F1 so the barh() top-to-bottom reading
    matches the lenient ranking used everywhere else in the paper.
    """
    lenient = load_micro(df, mode="lenient").rename(
        columns={"mean_F1": "lenient_mean", "ci_F1": "lenient_ci"})
    strict = load_micro(df, mode="strict").rename(
        columns={"mean_F1": "strict_mean", "ci_F1": "strict_ci"})
    out = lenient.merge(
        strict[["model", "strict_mean", "strict_ci"]], on="model", how="left")
    return out.sort_values("lenient_mean", ascending=True).reset_index(drop=True)


def load_run_variance(df: pd.DataFrame, variance_fold: int = 0) -> pd.DataFrame:
    """For each model with >=2 runs on `variance_fold`, return mean F1 and CI
    across the runs (strict + lenient).  Also joins the fold-variance stats
    from load_micro_both so both variance sources can be plotted together."""
    runs = df[(df["label"] == "MICRO") & (df["fold"] == variance_fold)].copy()
    runs["F1"] = runs["F1"].astype(float)
    rows = []
    for (model, mode), sub in runs.groupby(["model", "mode"]):
        f1s = sub["F1"].to_numpy()
        if len(f1s) < 2:
            continue
        rows.append({
            "model":    model,
            "mode":     mode,
            "n_runs":   len(f1s),
            "run_mean": float(f1s.mean()),
            "run_ci":   _normal_ci(f1s),
        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = long.pivot(index="model", columns="mode",
                      values=["run_mean", "run_ci", "n_runs"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    # attach fold-variance stats for the same models
    fold = load_micro_both(df)
    return wide.merge(fold, on="model", how="left")


def load_per_label(df: pd.DataFrame, mode: str = "lenient") -> pd.DataFrame:
    """Return mean per-label F1 across folds (run=0), pivoted to model × label."""
    plabel = df[(df["mode"] == mode) & (df["label"] != "MICRO")].copy()
    plabel = plabel[plabel["run"] == 0]
    plabel["F1"] = plabel["F1"].astype(float)
    agg = (plabel
           .groupby(["model", "label"], as_index=False)["F1"].mean())
    pivot = agg.pivot(index="model", columns="label", values="F1")
    # sort rows by overall mean
    pivot["_overall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_overall", ascending=False).drop(columns="_overall")
    # column order
    cols = [c for c in LABEL_ORDER if c in pivot.columns]
    return pivot[cols]


# -----------------------------------------------------------------------------
# Figure 1: overall F1 with 95% CIs, strict + lenient side by side
# -----------------------------------------------------------------------------
def plot_overall(micro: pd.DataFrame, suffix: str = "") -> None:
    """Grouped horizontal bars: strict (hatched) + lenient (solid) per model.

    Colour = training regime; hatch pattern separates strict from lenient.
    Sorted by lenient F1 so the ranking is visually intuitive.
    """
    n = len(micro)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    y = np.arange(n)
    bar_h = 0.36
    colors = [REGIME_COLOR[r] for r in micro["regime"]]

    # Lenient bars (solid, upper half of each slot)
    ax.barh(y + bar_h / 2, micro["lenient_mean"], height=bar_h,
            xerr=micro["lenient_ci"], color=colors,
            edgecolor="black", linewidth=0.5,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "black"})
    # Strict bars (hatched, lower half of each slot)
    ax.barh(y - bar_h / 2, micro["strict_mean"], height=bar_h,
            xerr=micro["strict_ci"], color=colors, alpha=0.55,
            edgecolor="black", linewidth=0.5, hatch="////",
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "black"})

    # value labels at bar end
    for i, r in micro.iterrows():
        ax.text(r["lenient_mean"] + r["lenient_ci"] + 0.008, i + bar_h / 2,
                f"{r['lenient_mean']:.2f}", va="center", fontsize=7,
                color="black")
        ax.text(r["strict_mean"] + r["strict_ci"] + 0.008, i - bar_h / 2,
                f"{r['strict_mean']:.2f}", va="center", fontsize=7,
                color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(micro["display"], fontsize=9)
    ax.set_xlabel("Micro-F1 (mean across folds, 95% CI)", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0.0, color="black", linewidth=0.5)

    # annotate n_folds where less than max
    max_n = int(micro["n_folds"].max())
    for i, r in micro.iterrows():
        if r["n_folds"] < max_n:
            ax.text(0.98, i, f"n={int(r['n_folds'])}", va="center",
                    ha="right", fontsize=7, color="dimgray",
                    transform=ax.get_yaxis_transform())

    # two-part legend: regime colour + strict/lenient pattern
    from matplotlib.patches import Patch
    regime_handles = [Patch(facecolor=REGIME_COLOR[r], edgecolor="black",
                            label=REGIME_LABEL[r])
                      for r in ("zero_shot", "fine_tuned", "prompted_llm")]
    metric_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="Lenient F1"),
        Patch(facecolor="lightgray", edgecolor="black", hatch="////",
              alpha=0.55, label="Strict F1"),
    ]
    leg1 = ax.legend(handles=regime_handles, loc="lower right",
                     fontsize=8, frameon=False, title="Regime",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=metric_handles, loc="lower right", fontsize=8,
              frameon=False, bbox_to_anchor=(1.0, 0.22),
              title="Match criterion", title_fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_overall{suffix}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_overall{suffix}.pdf / .png ({n} models, strict + lenient)")


# -----------------------------------------------------------------------------
# Figure 2: per-label lenient F1 heatmap
# -----------------------------------------------------------------------------
def plot_per_label(pivot: pd.DataFrame, suffix: str = "") -> None:
    # Re-map row indices to display names
    row_display = [MODEL_META[m]["display"] for m in pivot.index]
    data = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(np.arange(len(row_display)))
    ax.set_yticklabels(row_display, fontsize=9)
    ax.set_xlabel("Entity label", fontsize=9)

    # annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt_color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Lenient F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.spines[:].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_per_label_lenient{suffix}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_per_label_lenient{suffix}.pdf / .png "
          f"({pivot.shape[0]} models × {pivot.shape[1]} labels)")


# -----------------------------------------------------------------------------
# Figure 3: the schema-fit story (zero-shot vs. fine-tuned, same architecture)
# -----------------------------------------------------------------------------
def plot_schema_fit(micro: pd.DataFrame, suffix: str = "") -> None:
    """Paired-bar plot: for the two families where we have both zero-shot and
    fine-tuned (only GLiNER2 currently), show the delta."""
    m = micro.set_index("model")
    if not {"gliner2", "gliner2-ft"}.issubset(m.index):
        print("  (skip fig_schema_fit: need both gliner2 and gliner2-ft)")
        return

    families = [
        ("GLiNER2",  "gliner2",  "gliner2-ft"),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    x = np.arange(len(families))
    w = 0.35
    zs_vals = [m.loc[z, "mean_F1"] for _, z, _ in families]
    zs_err  = [m.loc[z, "ci_F1"]  for _, z, _ in families]
    ft_vals = [m.loc[f, "mean_F1"] for _, _, f in families]
    ft_err  = [m.loc[f, "ci_F1"]  for _, _, f in families]

    ax.bar(x - w/2, zs_vals, w, yerr=zs_err, color=REGIME_COLOR["zero_shot"],
           edgecolor="black", linewidth=0.5, label="Zero-shot",
           error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "black"})
    ax.bar(x + w/2, ft_vals, w, yerr=ft_err, color=REGIME_COLOR["fine_tuned"],
           edgecolor="black", linewidth=0.5, label="Fine-tuned",
           error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "black"})

    # delta annotations
    for i, (zs, ft) in enumerate(zip(zs_vals, ft_vals)):
        delta = ft - zs
        ax.annotate(f"+{delta:.2f}", xy=(x[i], max(zs, ft) + 0.04),
                    ha="center", fontsize=9, fontweight="bold",
                    color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([fam for fam, _, _ in families], fontsize=9)
    ax.set_ylabel("Lenient micro-F1 (mean ± 95% CI)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_schema_fit{suffix}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_schema_fit{suffix}.pdf / .png (schema-exposure paired bars)")


# -----------------------------------------------------------------------------
# Figure 4: variance decomposition (run-variance vs fold-variance, LLMs only)
# -----------------------------------------------------------------------------
def plot_run_vs_fold_variance(rv: pd.DataFrame, variance_fold: int = 0,
                              suffix: str = "") -> None:
    """For each LLM with >=2 runs on `variance_fold`, plot lenient F1 with
    two error bars: fold-variance (5 folds, 1 run each) and run-variance
    (multiple runs on `variance_fold`).  Visual argument: run << fold.
    """
    if rv.empty:
        print("  (skip fig_variance_decomp: no run-variance data)")
        return

    # Sort by fold-mean F1 ascending (like the other bar figs)
    rv = rv.sort_values("lenient_mean", ascending=True).reset_index(drop=True)
    display = [MODEL_META[m]["display"] for m in rv["model"]]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(rv))
    offset = 0.18

    # Fold variance (5 folds, run 0): mean across folds ± fold CI
    ax.errorbar(rv["lenient_mean"], y + offset,
                xerr=rv["lenient_ci"], fmt="o", markersize=6,
                color="#1f77b4", ecolor="#1f77b4", elinewidth=1.5,
                capsize=4, label="Fold variance (5 folds × 1 run)")
    # Run variance (fold `variance_fold`, N runs): mean across runs ± run CI
    ax.errorbar(rv["run_mean_lenient"], y - offset,
                xerr=rv["run_ci_lenient"], fmt="s", markersize=6,
                color="#d62728", ecolor="#d62728", elinewidth=1.5,
                capsize=4,
                label=f"Run variance (fold {variance_fold} × 3 runs)")

    # numeric CI labels at right edge
    for i, r in rv.iterrows():
        ax.text(r["lenient_mean"] + r["lenient_ci"] + 0.008, i + offset,
                f"±{r['lenient_ci']:.03f}", va="center", fontsize=7,
                color="#1f77b4")
        ax.text(r["run_mean_lenient"] + r["run_ci_lenient"] + 0.008,
                i - offset,
                f"±{r['run_ci_lenient']:.03f}", va="center", fontsize=7,
                color="#d62728")

    ax.set_yticks(y)
    ax.set_yticklabels(display, fontsize=9)
    ax.set_xlabel("Lenient micro-F1 (95% CI)", fontsize=9)
    ax.set_xlim(0.30, 0.85)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_variance_decomp{suffix}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_variance_decomp{suffix}.pdf / .png "
          f"({len(rv)} LLMs, fold vs run variance)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                    help="Path to metrics CSV produced by compute_ner_metrics.py.")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    # Derive figure suffix from CSV stem: ner_final_metrics_filtered -> _filtered
    stem = csv_path.stem
    suffix = ""
    if stem.startswith("ner_final_metrics"):
        tail = stem[len("ner_final_metrics"):]
        if tail:
            suffix = tail  # already begins with underscore

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path.relative_to(REPO)}"
          + (f"  (suffix='{suffix}')" if suffix else ""))

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype":      42,   # embed as TrueType (not Type-3) for LaTeX
        "ps.fonttype":       42,
    })

    micro = load_micro(df, mode="lenient")
    micro_both = load_micro_both(df)
    pivot = load_per_label(df, mode="lenient")
    rv = load_run_variance(df, variance_fold=0)

    plot_overall(micro_both, suffix=suffix)
    plot_per_label(pivot, suffix=suffix)
    plot_schema_fit(micro, suffix=suffix)
    plot_run_vs_fold_variance(rv, variance_fold=0, suffix=suffix)

    print(f"\nAll figures in {FIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
