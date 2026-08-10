"""Bar chart of span counts per label in the deduplicated gold pool.

Gold pool = single-annotator letters (union across the 4 annotators) +
A2's version of the 10 shared IAA letters.

Writes: NER_pipeline/IAA_annots/figures_707/label_counts.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LS_OUT = REPO / "data" / "labelstudio_output"
OUT_DIR = REPO / "NER_pipeline" / "IAA_annots" / "figures_707"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "A1": LS_OUT / "NER_707_A1_final.json",
    "A2":    LS_OUT / "NER_707_A2_final.json",
    "A3":  LS_OUT / "NER_707_A3_final.json",
    "A4":  LS_OUT / "NER_707_A4_final.json",
}


def is_annot(t):
    return any(not a.get("was_cancelled") for a in (t.get("annotations") or []))


def spans(t):
    for a in (t.get("annotations") or []):
        if a.get("was_cancelled"):
            continue
        out = []
        for r in a.get("result") or []:
            if r.get("type") != "labels":
                continue
            v = r.get("value") or {}
            labs = v.get("labels") or []
            if not labs:
                continue
            try:
                s, e = int(v["start"]), int(v["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e > s:
                out.append((s, e, labs[0]))
        return out
    return []


counts = Counter()
n_letters = 0
for name, p in FILES.items():
    tasks = [t for t in json.loads(p.read_text(encoding="utf-8")) if is_annot(t)]
    for t in tasks:
        d = t.get("data") or {}
        assignment = d.get("assignment")
        if assignment == "IAA" and name != "A2":
            continue  # dedup: keep only A2's IAA version
        n_letters += 1
        for _, _, lbl in spans(t):
            counts[lbl] += 1

# Order: proper-name classes first, then nominal, then rare
ORDER = ["PER", "LOC", "ORG", "EVT", "ART", "PER_NOM", "ART_NOM"]
labels = [l for l in ORDER if l in counts] + [l for l in counts if l not in ORDER]
values = [counts[l] for l in labels]

# Colour by "family": proper-name = steel blue, nominal = orange, event = grey
def colour(lbl):
    if lbl.endswith("_NOM"):
        return "#e9a04b"   # muted orange
    if lbl == "EVT":
        return "#8a8a8a"   # grey
    return "#3a6ea5"       # steel blue

colours = [colour(l) for l in labels]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.4)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
            f"{v:,}", ha="center", va="bottom", fontsize=9)

total = sum(values)
ax.set_ylabel("Number of spans")
ax.set_title(
    f"Span counts per label in the Navez gold pool\n"
    f"({n_letters} letters, {total:,} spans; IAA overlap deduplicated to A2's version)",
    fontsize=11,
)
ax.set_ylim(0, max(values) * 1.12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#3a6ea5", edgecolor="black", label="Proper-name / location / organisation"),
    Patch(facecolor="#e9a04b", edgecolor="black", label="Nominal reference (_NOM)"),
    Patch(facecolor="#8a8a8a", edgecolor="black", label="Event"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=9, frameon=False)

fig.tight_layout()
out = OUT_DIR / "label_counts.png"
fig.savefig(out, dpi=200)
print(f"wrote {out.relative_to(REPO)}")
