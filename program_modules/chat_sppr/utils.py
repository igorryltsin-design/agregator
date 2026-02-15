"""Utility helpers for chat processing."""

from __future__ import annotations

import hashlib
import random
import re
import statistics
from datetime import datetime, timezone
from typing import Iterable, List

STOPWORDS = {"и", "в", "на", "с", "по", "для", "из", "не", "что", "как", "к"}


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9]{3,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    overlap = len(set(tokenize(a)) & set(tokenize(b)))
    union = len(set(tokenize(a)) | set(tokenize(b))) or 1
    return overlap / union


def heuristic_answer(question: str, references: Iterable[str]) -> str:
    base = normalize_text(question)
    ref = next(iter(references), "")
    return f"Вопрос: {base}. Согласно источнику: {ref[:200]}..."


def message_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def rolling_average(values: Iterable[float], window: int = 5) -> float:
    data = list(values)[-window:]
    return statistics.fmean(data) if data else 0.0


def random_ack() -> str:
    phrases = [
        "Понял запрос, проверяю материалы.",
        "Уточняю детали в базе знаний.",
        "Обновляю контекст беседы.",
    ]
    return random.choice(phrases)
