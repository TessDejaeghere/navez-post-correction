"""Figures for the NER error taxonomy, one visualisation per file.

Reads the long-form counts written by ``error_taxonomy.py`` and renders four
independent, paper-ready figures.  Each is saved as both PNG and PDF so it can
be placed on its own; no titles are drawn into the figures, on the assumption
that the caption carries the framing.

Outputs (in ``NER_pipeline/eval/figures/``)
-------------------------------------------
* ``fig_taxonomy_counts.png``       raw event counts, models x error class
* ``fig_taxonomy_burden.png``       events per letter, mechanical vs cognitive
* ``fig_taxonomy_composition.png``  each system's error profile, row-normalised
* ``fig_taxonomy_unbounded.png``    the unbounded burden, with fold CIs

Usage
-----
    python NER_pipeline/plot_error_taxonomy.py
"""
from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from error_taxonomy import (  # noqa: E402
    CLASS_ORDER, FAMILY, MECHANICAL, MODELS, TAXONOMY, active_classes,
)
import figstyle  # noqa: E402

EVAL_DIR = REPO / "NER_pipeline" / "eval"
COUNTS_CSV = EVAL_DIR / "error_taxonomy.csv"
FOLDS_JSON = EVAL_DIR / "folds_writer_year.json"
GOLD = EVAL_DIR / "gold_561_merged.jsonl"
FIG_DIR = EVAL_DIR / "figures"

MECH_COL = figstyle.MECH
COG_COL = figstyle.COG
FAMILY_COL = figstyle.FAMILY

MECH_COLS = [c for c in CLASS_ORDER if c in MECHANICAL]
COG_COLS = [c for c in CLASS_ORDER if c not in MECHANICAL]


def _active_cols(wide: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the classes that actually occur into mechanical / cognitive.

    Empty classes are dropped so that the figures show only work a corrector
    performs; ``F0`` is empty whenever the counts come from filtered predictions.
    """
    active = active_classes({m: wide.loc[m].to_dict() for m in wide.index})
    return ([c for c in active if c in MECHANICAL],
            [c for c in active if c not in MECHANICAL])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(counts_csv: Path = COUNTS_CSV) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[int, int]]:
    df = pd.read_csv(counts_csv)
    wide = (df.groupby(["model", "class"])["n"].sum()
              .unstack(fill_value=0)
              .reindex(index=MODELS, columns=CLASS_ORDER, fill_value=0))
    n = sum(1 for _ in GOLD.open(encoding="utf-8"))
    payload = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    fold_size: dict[int, int] = {}
    for r in payload["letters"]:
        fold_size[int(r["fold"])] = fold_size.get(int(r["fold"]), 0) + 1
    return df, wide, n, fold_size


def _save(fig, name: str, suffix: str = "") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, FIG_DIR / f"{name}{suffix}")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (name + suffix + '.png')}")


def _clean(ax) -> None:
    figstyle.clean(ax)


# ---------------------------------------------------------------------------
# 1. Count matrix
# ---------------------------------------------------------------------------
def fig_counts(wide: pd.DataFrame, per_letter: pd.DataFrame,
               suffix: str = "") -> None:
    mech_cols, cog_cols = _active_cols(wide)
    order = per_letter.sum(axis=1).sort_values().index.tolist()
    counts = wide.loc[order, mech_cols + cog_cols]

    # A blank column separates the two cost blocks.
    gapped = np.full((len(order), len(mech_cols) + len(cog_cols) + 1), np.nan)
    gapped[:, :len(mech_cols)] = counts[mech_cols].values
    gapped[:, len(mech_cols) + 1:] = counts[cog_cols].values

    fig, ax = plt.subplots(figsize=figstyle.size(height=3.0))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    vmax = float(np.nanmax(gapped))
    # Log shading: the classes span four orders of magnitude, so a linear
    # scale would leave every mechanical column indistinguishable from zero.
    ax.imshow(np.clip(np.ma.masked_invalid(gapped), 1, None), aspect="auto",
              cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax))

    for i in range(len(order)):
        for j in range(gapped.shape[1]):
            v = gapped[i, j]
            if np.isnan(v):
                continue
            frac = np.log10(max(v, 1)) / np.log10(max(vmax, 10))
            ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=8.2,
                    color="white" if frac > 0.62 else "#22303a")

    xt = list(range(len(mech_cols))) + \
        list(range(len(mech_cols) + 1, gapped.shape[1]))
    ax.set_xticks(xt)
    ax.set_xticklabels(mech_cols + cog_cols, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([figstyle.display(m) for m in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text((len(mech_cols) - 1) / 2, -0.95, "MECHANICAL", ha="center",
            fontsize=9, fontweight="bold", color=MECH_COL)
    ax.text(len(mech_cols) + 1 + (len(cog_cols) - 1) / 2, -0.95, "COGNITIVE",
            ha="center", fontsize=9, fontweight="bold", color=COG_COL)
    ax.set_ylim(len(order) - 0.5, -1.35)
    ax.set_xlabel("correction events over the 561 gold letters, pooled over "
                  "the 5 test folds", labelpad=8, color="#555555")
    _save(fig, "fig_taxonomy_counts", suffix)


# ---------------------------------------------------------------------------
# 2. Burden per letter
# ---------------------------------------------------------------------------
def fig_burden(per_letter: pd.DataFrame, suffix: str = "") -> None:
    order = per_letter.sum(axis=1).sort_values().index.tolist()
    mech = per_letter.loc[order, MECH_COLS].sum(axis=1)
    cog = per_letter.loc[order, COG_COLS].sum(axis=1)

    fig, ax = plt.subplots(figsize=figstyle.size(height=2.9))
    fig.patch.set_facecolor("white")
    ys = list(range(len(order)))
    ax.barh(ys, mech, color=MECH_COL, height=.6, label="mechanical", zorder=3)
    ax.barh(ys, cog, left=mech, color=COG_COL, height=.6, label="cognitive",
            zorder=3)
    for y, (m, c) in enumerate(zip(mech, cog)):
        ax.text(m / 2, y, f"{m:.1f}", ha="center", va="center", fontsize=7.5,
                color="white", zorder=4)
        ax.text(m + c / 2, y, f"{c:.1f}", ha="center", va="center", fontsize=7.5,
                color="white", zorder=4)
        ax.text(m + c + 0.35, y, f"{m + c:.1f}", va="center", fontsize=8,
                color=figstyle.VALUE, fontweight="bold", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([figstyle.display(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, float((mech + cog).max()) * 1.12)
    ax.set_xlabel("correction events per letter", labelpad=6)
    _clean(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=.95,
              edgecolor=figstyle.GRID)
    _save(fig, "fig_taxonomy_burden", suffix)


# ---------------------------------------------------------------------------
# 3. Composition
# ---------------------------------------------------------------------------
def fig_composition(wide: pd.DataFrame, per_letter: pd.DataFrame,
                    suffix: str = "") -> None:
    mech_cols, cog_cols = _active_cols(wide)
    order = per_letter.sum(axis=1).sort_values().index.tolist()
    share = 100 * wide.loc[order].div(wide.loc[order].sum(axis=1), axis=0)

    blues = plt.get_cmap("Blues")(np.linspace(.42, .88, len(mech_cols)))
    reds = plt.get_cmap("Reds")(np.linspace(.40, .86, len(cog_cols)))
    colour = {**dict(zip(mech_cols, blues)), **dict(zip(cog_cols, reds))}

    fig, ax = plt.subplots(figsize=figstyle.size(height=3.2))
    fig.patch.set_facecolor("white")
    ys = list(range(len(order)))
    left = np.zeros(len(order))
    for c in mech_cols + cog_cols:
        v = share[c].values
        ax.barh(ys, v, left=left, height=.62, color=colour[c], zorder=3,
                label=f"{c} · {TAXONOMY[c][0]}")
        for y, (val, l0) in enumerate(zip(v, left)):
            if val >= 6:
                ax.text(l0 + val / 2, y, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
        left = left + v
    ax.set_yticks(ys)
    ax.set_yticklabels([figstyle.display(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the system's correction events", labelpad=6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _clean(ax)
    ax.legend(fontsize=7.2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    _save(fig, "fig_taxonomy_composition", suffix)


# ---------------------------------------------------------------------------
# 4. Unbounded burden with fold CIs
# ---------------------------------------------------------------------------
def fig_unbounded(df: pd.DataFrame, fold_size: dict[int, int],
                  suffix: str = "") -> None:
    c1 = (df[df["class"] == "C1"].groupby(["model", "fold"])["n"].sum()
            .unstack(fill_value=0).reindex(MODELS, fill_value=0))
    rates = c1.div(pd.Series(fold_size))
    mean = rates.mean(axis=1)
    half = 1.96 * rates.std(axis=1, ddof=1) / sqrt(rates.shape[1])
    order = mean.sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=figstyle.size(height=2.9))
    fig.patch.set_facecolor("white")
    ys = list(range(len(order)))
    ax.barh(ys, mean[order], height=.6,
            color=[FAMILY_COL[FAMILY[m]] for m in order],
            xerr=half[order].values,
            error_kw={"ecolor": "#2c3e50", "lw": 1.0, "capsize": 2.5}, zorder=3)
    for y, m in enumerate(order):
        ax.text(mean[m] + half[m] + 0.3, y, f"{mean[m]:.2f}", va="center",
                fontsize=8, color=figstyle.VALUE, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([figstyle.display(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, float((mean + half).max()) * 1.16)
    ax.set_xlabel("missed mentions per letter (unbounded burden), "
                  "mean over 5 folds \u00b1 95 % CI", labelpad=6)
    _clean(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in FAMILY_COL.values()]
    ax.legend(handles, list(FAMILY_COL), loc="upper right",
              frameon=True, framealpha=.95, edgecolor=figstyle.GRID)
    _save(fig, "fig_taxonomy_unbounded", suffix)


def main(counts_csv: Path = COUNTS_CSV, suffix: str = "") -> None:
    figstyle.apply()
    df, wide, n, fold_size = load(counts_csv)
    per_letter = wide / n
    fig_counts(wide, per_letter, suffix)
    fig_burden(per_letter, suffix)
    fig_composition(wide, per_letter, suffix)
    fig_unbounded(df, fold_size, suffix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts-csv", type=str, default=str(COUNTS_CSV),
                    help="long-form counts CSV written by error_taxonomy.py")
    _args = ap.parse_args()
    _csv = Path(_args.counts_csv).resolve()
    _sfx = _csv.stem.replace("error_taxonomy", "")
    main(_csv, _sfx)
