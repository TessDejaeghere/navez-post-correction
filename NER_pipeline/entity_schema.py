"""
Entity schema for zero-shot NER over 19th-century French letters
(Navez correspondence corpus).

This is the single source of truth for the annotation scheme. Edit the
``ENTITY_SCHEMA`` dict to change labels, descriptions, or thresholds and
re-run ``gliner_predict.py`` — the rest of the pipeline updates automatically.

Label codes must match the Label Studio <View> schema in
``data/labelstudio_upload/*_ls_import.json`` (currently: PER, PER_NOM, ART,
ART_NOM, EVT, ORG, LOC).
"""

from __future__ import annotations

# Descriptions are written in English (the language GLiNER2 was trained on)
# but explicitly reference 19th-century French letter conventions and give
# French-language anchor examples — these cues steer the zero-shot model much
# better than abstract category labels.
ENTITY_SCHEMA: dict[str, dict] = {
    "PER": {
        "description": (
            "Proper personal name of a real individual: first name, surname, "
            "or a name with a formal title that is capitalised as part of the "
            "name. Examples from 19th-century French letters: 'François', "
            "'Monsieur David', 'le baron de Stassart', 'Madame Récamier'. "
            "DO NOT tag indirect references such as 'mon père' or 'le ministre'."
        ),
        "threshold": 0.45,
    },
    "PER_NOM": {
        "description": (
            "Nominal / indirect reference to a specific person via kinship, "
            "social role, profession, or honorific — without giving the proper "
            "name. Examples in French: 'ma nièce', 'mon cher maître', "
            "'le ministre', 'l'épouse', 'notre ami', 'son frère'. "
            "DO NOT tag proper names (those are PER)."
        ),
        "threshold": 0.50,
    },
    "ART": {
        "description": (
            "Explicit title of a named artwork — painting, sculpture, "
            "literary work, or musical piece — usually italicised or "
            "capitalised. Examples: 'la Tête de Vierge', 'Agar dans le "
            "désert', 'le Sacre de Napoléon', 'le Serment des Horaces'. "
            "DO NOT tag generic terms like 'tableau' or 'portrait' on their own."
        ),
        "threshold": 0.50,
    },
    "ART_NOM": {
        "description": (
            "Nominal / indirect reference to a specific physical artwork or "
            "creation (a particular painting, sketch, statue, manuscript). "
            "Examples in French: 'mon grand tableau', 'la toile', "
            "'ce portrait', 'cette esquisse', 'l'ouvrage'. "
            "DO NOT tag generic discussions of art such as 'j'aime la peinture'."
        ),
        "threshold": 0.55,
    },
    "EVT": {
        "description": (
            "A specific named or referenced event: art salon, exhibition, "
            "ceremony, public competition, or institutional gathering. "
            "Examples: 'le Salon de 1814', 'l'exposition de Gand', "
            "'la distribution des prix', 'le concours de Rome'."
        ),
        "threshold": 0.50,
    },
    "ORG": {
        "description": (
            "Formal institution, academy, state body, ministry, or recognised "
            "artistic faction. Examples: 'l'Académie de Bruxelles', "
            "'l'Institut', 'l'École des beaux-arts', 'le gouvernement', "
            "'la Société des beaux-arts'."
        ),
        "threshold": 0.50,
    },
    "LOC": {
        "description": (
            "Geographic location: city, country, region, street, building, or "
            "any specific place. Examples: 'Rome', 'Bruxelles', "
            "'la rue de l'Hôpital', 'Paris', 'l'Italie', 'le Louvre'."
        ),
        "threshold": 0.45,
    },
}


def build_schema(extractor) -> object:
    """Build a GLiNER2 schema object from ``ENTITY_SCHEMA``.

    Pass any GLiNER2 extractor instance — ``extractor.create_schema()`` is
    called to produce the schema and every label is added with its
    description, ``dtype='list'``, and per-label threshold.
    """
    schema = extractor.create_schema()
    schema.entities(
        {
            label: {
                "description": cfg["description"],
                "dtype": "list",
                "threshold": cfg["threshold"],
            }
            for label, cfg in ENTITY_SCHEMA.items()
        }
    )
    return schema


def label_set() -> list[str]:
    return list(ENTITY_SCHEMA.keys())
