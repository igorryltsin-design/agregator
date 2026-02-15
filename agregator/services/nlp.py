"""Russian morphology and NLP helpers.

Extracted from ``app.py`` to provide a single, testable module for:
- tokenization via *razdel*
- lemmatization via *pymorphy2*
- synonym expansion
"""

from __future__ import annotations

import re
from typing import List, Set

# ---------------------------------------------------------------------------
# Optional dependencies — graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from razdel import tokenize as _ru_tokenize
except Exception:  # pragma: no cover
    _ru_tokenize = None

try:
    import pymorphy2

    _morph = pymorphy2.MorphAnalyzer()
except Exception:  # pragma: no cover
    _morph = None

# ---------------------------------------------------------------------------
# Synonym dictionary
# ---------------------------------------------------------------------------

RU_SYNONYMS: dict[str, list[str]] = {
    "машина": ["автомобиль", "авто", "тачка", "car"],
    "изображение": [
        "картинка",
        "фото",
        "фотография",
        "image",
        "picture",
        "снимок",
        "скриншот",
        "screenshot",
    ],
    "вуз": ["университет", "институт", "универ"],
    "статья": ["публикация", "paper", "publication", "пейпер"],
    "диссертация": ["дисс", "thesis", "dissertation"],
    "журнал": ["journal", "magazine", "сборник"],
    "номер": ["выпуск", "issue"],
    "страница": ["стр", "страницы", "pages", "page"],
    "год": ["лет", "г"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ru_tokens(text: str) -> List[str]:
    """Tokenize *text* using razdel (fallback to regex split)."""
    s = (text or "").lower()
    if _ru_tokenize:
        try:
            return [t.text for t in _ru_tokenize(s)]
        except Exception:
            pass
    return re.sub(r"[^\w\d]+", " ", s).strip().split()


def lemma(word: str) -> str:
    """Return the normal form of *word* (pymorphy2 or lowercased)."""
    if _morph:
        try:
            p = _morph.parse(word)
            if p:
                return p[0].normal_form
        except Exception:
            pass
    return word.lower()


def lemmas(text: str) -> List[str]:
    """Return list of lemmatized tokens for *text*."""
    return [lemma(w) for w in ru_tokens(text)]


def expand_synonyms(word_lemmas: List[str]) -> Set[str]:
    """Expand a list of lemmas with known Russian synonyms."""
    out = set(word_lemmas)
    for lem in list(word_lemmas):
        for syn in RU_SYNONYMS.get(lem, []):
            out.add(lemma(syn))
    return out
