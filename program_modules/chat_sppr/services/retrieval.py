"""Retrieval stub service linking chat to knowledge references."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import AppConfig
from .. import models, utils

LOGGER = logging.getLogger("chat.retrieval")


class RetrievalService:
    def __init__(self, config: AppConfig):
        self.config = config

    def retrieve(self, session: Session, question: str) -> List[Dict]:
        since = datetime.utcnow() - timedelta(hours=self.config.retrieval.freshness_hours)
        stmt = (
            select(models.ChatSession)
            .where(models.ChatSession.updated_at >= since)
            .order_by(models.ChatSession.updated_at.desc())
            .limit(50)
        )
        sessions = list(session.scalars(stmt))
        references = []
        for chat_session in sessions:
            for turn in chat_session.turns[-3:]:
                similarity = utils.fuzzy_similarity(question, turn.content)
                if similarity >= self.config.retrieval.decay:
                    references.append(
                        {
                            "session_id": chat_session.id,
                            "turn_id": turn.id,
                            "snippet": turn.content[:200],
                            "score": similarity,
                        }
                    )
        if not references:
            references.append(
                {
                    "session_id": 0,
                    "turn_id": 0,
                    "snippet": random.choice(self.config.retrieval.fallback_answers),
                    "score": 0.1,
                }
            )
        references.sort(key=lambda item: item["score"], reverse=True)
        return references[: self.config.retrieval.top_k]
