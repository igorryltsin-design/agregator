"""Response generation helper."""

from __future__ import annotations

import logging
from typing import Dict, List

from ..config import AppConfig
from .. import utils

LOGGER = logging.getLogger("chat.responder")


class ResponseService:
    def __init__(self, config: AppConfig):
        self.config = config

    def compose(self, question: str, references: List[Dict]) -> Dict:
        text = utils.heuristic_answer(question, [ref["snippet"] for ref in references])
        tokens = len(utils.tokenize(text))
        confidence = min(1.0, sum(ref["score"] for ref in references) / max(len(references), 1))
        LOGGER.debug("Response composed with confidence %.2f", confidence)
        return {
            "text": text,
            "tokens": tokens,
            "confidence": confidence,
            "references": [ref.get("turn_id", 0) for ref in references],
        }
