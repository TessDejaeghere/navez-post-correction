"""Shared figure style for every figure that appears in the paper.

One font, one palette, one width. Import and call :func:`apply` before building
any figure::

    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=figstyle.size(height=3.0))

Why a shared module
-------------------
The figures were written by three scripts at different times and drifted apart
in three ways that are visible once they sit on the same page:

1. *Font.* Matplotlib defaults to DejaVu Sans; the paper is set in Tinos.
2. *Palette.* Two scripts used ``#2f6690``/``#a83232``, one used the matplotlib
   ``tab10`` blue/red, so the same semantic distinction had two colour schemes.
3. *Width.* Figures were 6.8, 8.4, 9.4 and 9.6 inches wide but all are included
   at ``\\linewidth``. LaTeX scales each one by a different factor, so a nominal
   9 pt tick label reached the page at anywhere between 5.5 and 7.8 pt.

The third point is the one that matters most: matching font sizes across
scripts achieves nothing unless the figures are also authored at the width they
are printed at. :data:`TEXT_WIDTH_IN` is that width.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# anthology-ch.cls sets A4 (21 cm) with 3 cm left/right margins, so
# \linewidth = 15 cm = 5.906 in. Authoring at this width means LaTeX scales the
# figure by 1.0 and point sizes below are the point sizes on the page.
TEXT_WIDTH_IN = 5.9

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Cost tier: the paper's central distinction, used by the HTR edit-class figure
# and by both taxonomy figures.
MECH = "#2f6690"        # mechanical
MECH_LIGHT = "#9dc3d8"
COG = "#a83232"         # cognitive
COG_LIGHT = "#e5a9a2"

# Training regime. Deliberately the same blue/red as the cost tier: this was
# already the convention in plot_error_taxonomy.fig_unbounded, and the two
# variables never appear in the same figure. See the note in the module docs of
# plot_ner_results if you would rather separate them.
REGIME = {
    "zero_shot": "#95a5a6",
    "fine_tuned": MECH,
    "prompted_llm": COG,
}
# Same three colours keyed by the names error_taxonomy.FAMILY uses.
FAMILY = {"zero-shot": "#95a5a6", "fine-tuned": MECH, "prompted LLM": COG}

# Chrome
GRID = "#dddddd"
SPINE = "#bbbbbb"
TICK = "#666666"
VALUE = "#333333"

# ---------------------------------------------------------------------------
# Model display names — so the same system is spelled the same way everywhere
# ---------------------------------------------------------------------------
DISPLAY = {
    "claude-opus-4.8": "Claude Opus 4.8",
    "qwen3-235b-a22b-2507": "Qwen3 235B-A22B",
    "qwen3-vl-30b-a3b-instruct": "Qwen3-VL 30B-A3B",
    "dalembert": "d'AlemBERT (ft)",
    "camembertav2": "CamemBERTaV2 (ft)",
    "gliner2-ft": "GLiNER2 (ft)",
    "gliner2": "GLiNER2 (zero-shot)",
}


def display(model: str) -> str:
    """Display name for a model id, falling back to the id itself."""
    return DISPLAY.get(model, model)


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
def _register_tinos() -> bool:
    """Register the Tinos TTFs shipped with the LaTeX template, if present.

    The code release does not ship ``overleaf/``, so this returns False there
    and the fallback chain in :func:`apply` takes over. Figures still build.
    """
    found = False
    for directory in (REPO / "overleaf" / "fonts" / "Tinos", REPO / "fonts" / "Tinos"):
        if not directory.is_dir():
            continue
        for ttf in sorted(directory.glob("Tinos-*.ttf")):
            try:
                font_manager.fontManager.addfont(str(ttf))
                found = True
            except Exception:
                pass
        if found:
            break
    return found


def apply(base: float = 8.5) -> bool:
    """Install the shared style. Returns True if Tinos itself was found.

    ``base`` is the body size in points *as printed*, because figures are
    authored at :data:`TEXT_WIDTH_IN`.
    """
    have_tinos = _register_tinos()
    matplotlib.rcParams.update({
        "font.family": "serif",
        # Liberation Serif and Times New Roman are metric-compatible with Tinos,
        # so the fallback still matches the paper closely.
        "font.serif": ["Tinos", "Liberation Serif", "Times New Roman",
                       "DejaVu Serif"],
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base + 0.5,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base,
        "legend.fontsize": base - 0.5,
        "legend.title_fontsize": base - 0.5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": TICK,
        "ytick.color": TICK,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Embed real TrueType outlines rather than Type 3, which several
        # publishers reject and which renders poorly in some PDF viewers.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Mathtext (e.g. an italic "n" in a label) otherwise falls back to
        # DejaVu Sans and silently reintroduces a second font family.
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        # Tinos has U+2212, but the ASCII hyphen avoids a tofu box if the
        # fallback font does not.
        "axes.unicode_minus": False,
    })
    return have_tinos


def size(height: float, width: float = TEXT_WIDTH_IN) -> tuple[float, float]:
    """Figure size in inches at the paper's text width."""
    return (width, height)


def clean(ax, axis: str = "x") -> None:
    """Light chrome: grid on the value axis only, minimal spines."""
    ax.grid(axis=axis, color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", colors=TICK)


def save(fig, path_stem: Path, exts: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save a figure as both PDF (for LaTeX) and PNG (for quick viewing)."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(path_stem.with_suffix(f".{ext}"), bbox_inches="tight",
                    facecolor="white")
