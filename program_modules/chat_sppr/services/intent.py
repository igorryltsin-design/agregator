"""Intent detection and suggestion service."""

from __future__ import annotations

import logging
import random
from typing import Dict, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import IntentRepository, TurnRepository
from .. import utils

LOGGER = logging.getLogger("chat.intent")


class IntentService:
    def __init__(self, config: AppConfig):
        self.config = config

    def detect(self, session: Session, session_id: int, question: str) -> dict:
        repo = IntentRepository(session)
        history_repo = TurnRepository(session)
        turns = history_repo.history(session_id, limit=10)
        score = utils.fuzzy_similarity(question, " ".join(turn.content for turn in turns if turn.role == "user"))
        intent = self._classify(question)
        metadata = {"similarity": score}
        repo.record(session_id, intent, score, metadata)
        LOGGER.debug("Intent %s detected with score %.2f", intent, score)
        return {"intent": intent, "confidence": score}

    def suggestions(self, question: str) -> List[str]:
        base = [
            "Попросить аналитическое резюме",
            "Запросить источники, подтверждающие выводы",
            "Построить временную линию событий",
        ]
        tokens = utils.tokenize(question)
        if "отчет" in tokens:
            base.append("Сформировать отчёт по текущей теме")
        return base[: self.config.intents.suggestions]

    def _classify(self, text: str) -> str:
        tokens = utils.tokenize(text)
        if "источник" in tokens or "документ" in tokens:
            return "request_sources"
        if "сценарий" in tokens:
            return "scenario_analysis"
        if "обоснуй" in text.lower():
            return "justify_answer"
        return random.choice(["general_query", "status_check", "explain_terms"])
