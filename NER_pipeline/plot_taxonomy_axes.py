"""Schematic of the three-axis error classification, with worked examples.

Unlike the other taxonomy figures this one plots no counts: it is the
definition of the scheme itself. The eight classes are placed on the
detectability x locality plane (the third axis, search scope, is drawn as the
chip border), and the lower panel gives one attested example per class taken
from the Navez gold standard.

Axis assignments are read from ``error_taxonomy.TAXONOMY`` so the picture
cannot drift away from the code that produces the counts.

Outputs (in ``NER_pipeline/eval/figures/``)
-------------------------------------------
* ``fig_taxonomy_axes.png`` / ``.pdf``

Usage
-----
    python NER_pipeline/plot_taxonomy_axes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))

from error_taxonomy import CLASS_ORDER, FILTER_CLASS, TAXONOMY  # noqa: E402
import figstyle  # noqa: E402

FIG_DIR = REPO / "NER_pipeline" / "eval" / "figures"

CLASSES = [c for c in CLASS_ORDER if c != FILTER_CLASS]

# Rows are read top to bottom: the further down, the less the interface shows.
DET_ROWS = ["evident", "semi", "silent"]
LOC_COLS = ["in_span", "in_text", "external"]

DET_LABEL = {
    "evident": "evident\nwrong at a glance",
    "semi": "semi\njudgeable in context",
    "silent": "silent\nnothing flags it",
}
LOC_LABEL = {
    "in_span": "in-span\nthe span string",
    "in_text": "in-text\nelsewhere in the letter",
    "external": "external\nscan, guidelines,\nprosopography",
}

# One attested example per class, drawn from eval/error_taxonomy.md. Line
# breaks inside a mention are rendered as a space; see the caption.
EXAMPLES = {
    "M1": "«l'irascible Dupavillon» PER over an existing PER span",
    "M2": "«M F Navez» for gold «Monsieur M F Navez»",
    "M3": "«mon cher Navez» for gold «Navez»",
    "M4": "«Napoléon» tagged PER_NOM where gold has PER",
    "C1": "the dateline «Paris» LOC returned by no system",
    "C2": "«Projet» tagged PER in \"il est loin de Projet a la Chose\"",
    "C3": "«Chapelle Sixtine» tagged ART where gold has LOC",
    "C4": "«Verheyden Van Reeth» one PER span over two painters",
}

CHIP_H = 0.26
CHIP_GAP = 0.07
CHIP_W = 0.88


def total_height(n: int) -> float:
    """Height of a stack of ``n`` chips in cell coordinates."""
    return n * CHIP_H + (n - 1) * CHIP_GAP


def _cost_colours(cls: str) -> tuple[str, str]:
    """(fill, edge) for a class chip, keyed by its cost tier."""
    if TAXONOMY[cls][4] == "mechanical":
        return figstyle.MECH_LIGHT, figstyle.MECH
    return figstyle.COG_LIGHT, figstyle.COG


def _chip(ax, x, y, w, h, cls, *, fontsize=6.6, with_name=True) -> None:
    fill, edge = _cost_colours(cls)
    unbounded = TAXONOMY[cls][3] == "unbounded"
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.4 if unbounded else 0.9,
        linestyle=(0, (2.2, 1.4)) if unbounded else "solid",
        facecolor=fill, edgecolor=edge, zorder=3,
    )
    ax.add_patch(box)
    label = f"{cls}  {TAXONOMY[cls][0]}" if with_name else cls
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a", zorder=4)


# ---------------------------------------------------------------------------
# Panel A: the detectability x locality plane
# ---------------------------------------------------------------------------
def panel_plane(ax) -> None:
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # Cost regions: mechanical is the (evident|semi) x (in-span|in-text) block,
    # everything silent and everything external is cognitive.
    ax.add_patch(Rectangle((0, 0), 3, 3, facecolor=figstyle.COG_LIGHT,
                           alpha=0.22, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 1), 2, 2, facecolor=figstyle.MECH_LIGHT,
                           alpha=0.45, edgecolor="none", zorder=0))
    ax.plot([0, 2, 2], [1, 1, 3], color=figstyle.SPINE, lw=1.0,
            ls=(0, (4, 2)), zorder=2)

    for v in (1, 2):
        ax.axvline(v, color="white", lw=1.4, zorder=1)
        ax.axhline(v, color="white", lw=1.4, zorder=1)

    ax.text(0.08, 2.90, "mechanical", ha="left", va="top", fontsize=7,
            style="italic", color=figstyle.MECH, zorder=4)
    ax.text(0.08, 0.10, "cognitive", ha="left", va="bottom", fontsize=7,
            style="italic", color=figstyle.COG, zorder=4)

    by_cell: dict[tuple[int, int], list[str]] = {}
    for cls in CLASSES:
        _, det, loc, _scope, _cost, _fix = TAXONOMY[cls]
        by_cell.setdefault((LOC_COLS.index(loc), DET_ROWS.index(det)), []).append(cls)

    for (col, row), members in by_cell.items():
        cx = col + 0.5
        cy = 2.5 - row
        n = len(members)
        top = cy + total_height(n) / 2
        for i, cls in enumerate(members):
            y = top - (i + 1) * CHIP_H - i * CHIP_GAP
            _chip(ax, cx - CHIP_W / 2, y, CHIP_W, CHIP_H, cls)

    # C1 is the only unbounded class: say why, rather than leaving the dashed
    # border to the legend alone.
    ax.annotate(
        "work grows with the letter,\nnot with the predictions",
        xy=(2.5, 0.5 + total_height(3) / 2),
        xytext=(2.5, 1.42), fontsize=6.2, color=figstyle.COG,
        ha="center", va="center",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=figstyle.COG,
                        shrinkA=3, shrinkB=2),
        zorder=5,
    )

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels([LOC_LABEL[c] for c in LOC_COLS], fontsize=7)
    ax.set_yticks([2.5, 1.5, 0.5])
    ax.set_yticklabels([DET_LABEL[r] for r in DET_ROWS], fontsize=7)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("locality  —  where the evidence for the repair lives",
                  fontsize=8, labelpad=8)
    ax.set_ylabel("detectability  —  what the correction interface shows",
                  fontsize=8, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Patch(facecolor=figstyle.MECH_LIGHT, edgecolor=figstyle.MECH, lw=0.9,
              label="mechanical"),
        Patch(facecolor=figstyle.COG_LIGHT, edgecolor=figstyle.COG, lw=0.9,
              label="cognitive"),
        Patch(facecolor="white", edgecolor="#444444", lw=0.9,
              label="bounded search"),
        Patch(facecolor="white", edgecolor="#444444", lw=1.4,
              linestyle=(0, (2.2, 1.4)), label="unbounded search"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, frameon=False, fontsize=7, handlelength=1.6,
              handleheight=0.9, columnspacing=1.6)


# ---------------------------------------------------------------------------
# Panel B: one attested example per class
# ---------------------------------------------------------------------------
def panel_examples(ax) -> None:
    n = len(CLASSES)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 0.9)
    ax.axis("off")

    ax.text(0.0, n + 0.55, "one attested event per class",
            fontsize=7.5, style="italic", color=figstyle.VALUE)
    ax.plot([0, 1], [n + 0.30, n + 0.30], color=figstyle.SPINE, lw=0.8)

    for i, cls in enumerate(CLASSES):
        y = n - 1 - i + 0.12
        _chip(ax, 0.0, y, 0.055, 0.62, cls, fontsize=6.6, with_name=False)
        det, loc, scope = TAXONOMY[cls][1:4]
        ax.text(0.078, y + 0.31, TAXONOMY[cls][0], fontsize=7,
                va="center", ha="left")
        ax.text(0.325, y + 0.31, f"{det} / {loc.replace('_', '-')} / {scope}",
                fontsize=6.3, va="center", ha="left", color=figstyle.TICK)
        ax.text(0.555, y + 0.31, EXAMPLES[cls], fontsize=6.4, va="center",
                ha="left", color="#1a1a1a")
        if i < n - 1:
            ax.plot([0, 1], [y - 0.09, y - 0.09], color="#eeeeee", lw=0.6)


def main() -> None:
    figstyle.apply()
    fig = plt.figure(figsize=figstyle.size(height=6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.20)
    panel_plane(fig.add_subplot(gs[0]))
    panel_examples(fig.add_subplot(gs[1]))
    figstyle.save(fig, FIG_DIR / "fig_taxonomy_axes")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_taxonomy_axes.png'}")


if __name__ == "__main__":
    main()
